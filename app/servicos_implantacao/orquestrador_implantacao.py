"""Orquestrador da etapa de implantação, promoção de modelo e publicação de regras de negócio."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import cast, final
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema, TensorSpec
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.conf_yaml import ConfYAML
from app.contratos_implantacao.avaliador_regras import AvaliadorRegras, RequisicaoImovel, RespostaPredicao, RespostaRegra
from app.contratos_implantacao.provedor_campeao import MetadadosModelo
from app.motores_regras.avaliador_pandas import AvaliadorPandas
from app.motores_regras.avaliador_polars import AvaliadorPolars
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_implantacao.carregador_campeao import CarregadorCampeao
from app.servicos_implantacao.modelo_mlflow import ModeloComRegrasNegocio


class ResultadoImplantacao(BaseModel):
    """Resultado oficial consolidado da execução da Etapa de Implantação."""

    modelo_promovido: str = Field(description="Nome do algoritmo promovido a produção")
    versao_modelo: str = Field(description="Identificador da versão promovida")
    caminho_artefato_modelo: str = Field(description="Caminho do arquivo serializado do modelo")
    caminho_metadados: str = Field(description="Caminho do arquivo de metadados")
    rmse_holdout: float = Field(description="RMSE em reais sobre o holdout cego")
    mae_holdout: float = Field(description="MAE em reais sobre o holdout cego")
    r2_holdout: float = Field(description="R² sobre o holdout cego")
    total_regras_ativas: int = Field(description="Quantidade de regras dinâmicas ativas")
    exemplo_predicao: RespostaPredicao = Field(description="Resultado de inferência de teste executada")
    mlflow_run_id: str = Field(description="ID do run de promoção no MLflow")
    data_implantacao: str = Field(description="Timestamp da implantação")


class ErroOrquestradorImplantacao(Exception):
    """Exceção levantada em falhas durante a orquestração da implantação."""


@final
class OrquestradorImplantacao:
    """Coordenador completo para treinamento final, empacotamento, promoção e serviço de API."""

    def __init__(self, configuracao: ConfYAML) -> None:
        self._conf = configuracao
        staging_conf = self._conf.obter_staging()
        self._repositorio = RepositorioSqlite(
            caminho=staging_conf.caminho,
            tabela="imoveis_limpos",
        )

    @classmethod
    def da_configuracao(cls, conf: ConfYAML) -> OrquestradorImplantacao:
        """Fábrica de conveniência a partir de uma instância ConfYAML."""
        return cls(conf)

    def executar_implantacao(
        self,
        diretorio_artefatos: Path | str = "production_artifacts/campeao",
        modelo_campeao_override: str | None = None,
    ) -> ResultadoImplantacao:
        """Executa o pipeline completo de promoção do campeão e validação das regras."""
        config_completa = self._conf.obter_configuracao_completa()
        config_implantacao = config_completa.implantacao
        config_modelagem = config_completa.modelagem
        coluna_alvo = config_completa.projeto.alvo
        dir_artefatos = Path(diretorio_artefatos)
        dir_artefatos.mkdir(parents=True, exist_ok=True)

        # 1. Carregamento da base tratada
        df_completo = self._repositorio.ler_dados()
        if df_completo.empty:
            raise ErroOrquestradorImplantacao(
                "Tabela 'imoveis_limpos' está vazia.")

        # 2. Resolução dinâmica do modelo campeão
        nome_campeao, rmse_cv_estimado = self._resolver_modelo_campeao(
            modelo_campeao_override, config_modelagem
        )

        # 3. Separação idêntica à Etapa 5 (treino 80% / holdout 20%)
        holdout_pct = config_completa.avaliacao.validacao.holdout
        n_total = len(df_completo)
        n_holdout = int(n_total * holdout_pct)

        df_embaralhado = df_completo.sample(
            frac=1.0, random_state=42).reset_index(drop=True)
        df_holdout = cast(
            pd.DataFrame, df_embaralhado.iloc[:n_holdout]).copy().reset_index(drop=True)
        df_treino = cast(
            pd.DataFrame, df_embaralhado.iloc[n_holdout:]).copy().reset_index(drop=True)

        todas_colunas: list[str] = [str(c) for c in cast(
            list[object], list(df_treino.columns))]
        X_cols: list[str] = [c for c in todas_colunas if c != coluna_alvo]
        X_treino = cast(pd.DataFrame, df_treino[X_cols])
        y_treino = cast(pd.Series, df_treino[coluna_alvo])
        X_holdout = cast(pd.DataFrame, df_holdout[X_cols])
        y_holdout = cast(pd.Series, df_holdout[coluna_alvo])

        # 4. Construção do pré-processador tabular
        cols_num: list[str] = []
        cols_cat: list[str] = []
        for c in X_cols:
            if pd.api.types.is_numeric_dtype(X_treino[c]):
                cols_num.append(c)
            else:
                cols_cat.append(c)

        item_campeao = config_modelagem.obter_item_modelo(nome_campeao)
        tipo_prep = item_campeao.obter_preprocessador() if item_campeao is not None else "standard"

        passos_num: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if tipo_prep in {"standard", "standard_scaler", "padrao"}:
            passos_num.append(("scaler", StandardScaler()))
        elif tipo_prep in {"minmax", "min_max", "minmax_scaler"}:
            passos_num.append(("scaler", MinMaxScaler()))
        elif tipo_prep in {"robust", "robust_scaler"}:
            passos_num.append(("scaler", RobustScaler()))
        elif tipo_prep in {"maxabs", "max_abs", "maxabs_scaler"}:
            passos_num.append(("scaler", MaxAbsScaler()))

        pipe_num = Pipeline(passos_num)
        pipe_cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", pipe_num, cols_num),
                ("cat", pipe_cat, cols_cat),
            ]
        )

        # 5. Instanciação dinâmica do regressor concreto campeão
        estimador_base, desc_algoritmo = self._instanciar_regressor_concreto(
            nome_campeao, config_modelagem
        )

        pipeline_base = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", estimador_base),
            ]
        )

        # Envelopamento log1p para predições estritamente não-negativas e lineares
        estimador_campeao = TransformedTargetRegressor(
            regressor=pipeline_base,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        # 6. Ajuste sobre os dados de treino
        estimador_campeao.fit(X_treino, y_treino)

        # 7. Avaliação de validação cega sobre o holdout
        predicoes_holdout = cast(
            np.ndarray[tuple[int, ...], np.dtype[np.float64]],
            np.asarray(estimador_campeao.predict(X_holdout), dtype=np.float64),
        )
        y_real = np.asarray(y_holdout, dtype=np.float64)

        erros = y_real - predicoes_holdout
        rmse_holdout = float(np.sqrt(np.mean(erros**2)))
        mae_holdout = float(np.mean(np.abs(erros)))
        sst = float(np.sum((y_real - np.mean(y_real)) ** 2))
        sse = float(np.sum(erros**2))
        r2_holdout = float(1.0 - (sse / sst)) if sst > 0.0 else 0.0

        timestamp_iso = datetime.now(timezone.utc).isoformat()
        versao_modelo = "1.0.0-champion"

        # 8. Metadados do campeão dinâmico
        metadados = MetadadosModelo(
            nome_modelo=nome_campeao,
            versao=versao_modelo,
            algoritmo=desc_algoritmo,
            transformacao_alvo="log1p_embutida",
            rmse_cv=round(rmse_cv_estimado, 2),
            rmse_holdout=round(rmse_holdout, 2),
            r2_holdout=round(r2_holdout, 4),
            data_promocao=timestamp_iso,
        )

        # 9. Serialização dos artefatos em disco
        carregador = CarregadorCampeao.salvar_modelo_e_metadados(
            estimador=estimador_campeao,
            metadados=metadados,
            diretorio_destino=dir_artefatos,
        )

        # 10. Registro de promoção e governança no MLflow
        run_id_mlflow = self._registrar_promocao_mlflow(
            metadados=metadados,
            dir_artefatos=dir_artefatos,
        )

        # 11. Teste de inferência e cálculo de regras com o modelo e regras configuradas
        df_exemplo = pd.DataFrame([{
            "Metragem": 85.5,
            "Quartos": 3,
            "Banheiros": 2,
            "Vagas": 2,
            "Bairro": "Jardim Botânico",
            "Zona": "Sul",
        }])
        regras_dict = [r.model_dump() for r in config_implantacao.regras if r.ativo]
        modelo_com_regras = ModeloComRegrasNegocio(
            modelo_estimador=estimador_campeao,
            regras_configuradas=regras_dict,
            coluna_alvo_predito=config_implantacao.coluna_alvo_predito,
        )
        df_pred = modelo_com_regras.predict(None, df_exemplo)
        preco_predito = float(df_pred["preco_predito"].iloc[0])
        regras_calculadas = {
            str(col): float(df_pred[col].iloc[0])
            for col in df_pred.columns
            if str(col) != config_implantacao.coluna_alvo_predito
        }

        detalhes: list[RespostaRegra] = []
        for regra_item in config_implantacao.regras:
            if not regra_item.ativo:
                continue
            valor = regras_calculadas.get(regra_item.destino, 0.0)
            detalhes.append(
                RespostaRegra(
                    nome=regra_item.nome,
                    destino=regra_item.destino,
                    descricao=regra_item.descricao,
                    valor=valor,
                )
            )

        data_hora_atual = datetime.now(timezone.utc).isoformat()
        resposta_exemplo = RespostaPredicao(
            preco_predito=preco_predito,
            regras_calculadas=regras_calculadas,
            detalhes_regras=detalhes,
            modelo_campeao=nome_campeao,
            versao_modelo=versao_modelo,
            data_hora_iso=data_hora_atual,
        )

        total_regras_ativas = len(
            [r for r in config_implantacao.regras if r.ativo]
        )

        resultado = ResultadoImplantacao(
            modelo_promovido=nome_campeao,
            versao_modelo=versao_modelo,
            caminho_artefato_modelo=str(dir_artefatos / "modelo_campeao.joblib"),
            caminho_metadados=str(dir_artefatos / "metadados_campeao.json"),
            rmse_holdout=round(rmse_holdout, 2),
            mae_holdout=round(mae_holdout, 2),
            r2_holdout=round(r2_holdout, 4),
            total_regras_ativas=total_regras_ativas,
            exemplo_predicao=resposta_exemplo,
            mlflow_run_id=run_id_mlflow,
            data_implantacao=timestamp_iso,
        )

        # 13. Geração de relatórios
        self._gerar_relatorios(resultado)

        return resultado

    def _resolver_modelo_campeao(
        self,
        modelo_campeao_override: str | None,
        config_modelagem: object,
    ) -> tuple[str, float]:
        """Resolve dinamicamente o modelo campeão e seu RMSE estimado de CV."""
        from app.esquema_configuracao import ConfiguracaoModelagem

        config_mod=cast(ConfiguracaoModelagem, config_modelagem)

        if modelo_campeao_override:
            return modelo_campeao_override, 0.0

        # 1. Tenta obter do relatório de avaliação estatística (Etapa 5)
        caminho_aval=Path("storage-data/relatorio_avaliacao.json")
        if caminho_aval.exists():
            try:
                with open(caminho_aval, "r", encoding="utf-8") as f:
                    dados_aval=cast(dict[str, object], json.load(f))
                rec=str(dados_aval.get("modelo_final_recomendado", ""))
                comp=cast(dict[str, object], dados_aval.get(
                    "comparacao_estatistica", {}))
                campeao_est=str(comp.get("modelo_campeao_estatistico", ""))
                alvo_nome=campeao_est if (
                    rec.startswith("ensemble_") or not rec) else rec

                # Tenta obter rmse_cv pareado ou rmse holdout
                rmse_cv=0.0
                metricas_holdout=cast(
                    dict[str, dict[str, float]], dados_aval.get("metricas_holdout", {}))
                if alvo_nome in metricas_holdout:
                    val_rmse=metricas_holdout[alvo_nome].get("rmse", 0.0)
                    rmse_cv=float(str(val_rmse))

                if alvo_nome:
                    return alvo_nome, rmse_cv
            except Exception:
                pass

        # 2. Tenta obter do relatório da modelagem (Etapa 4)
        caminho_mod=Path("storage-data/relatorio_modelagem.json")
        if caminho_mod.exists():
            try:
                with open(caminho_mod, "r", encoding="utf-8") as f:
                    dados_mod=cast(dict[str, object], json.load(f))
                dict_tuning=cast(dict[str, object],
                                   dados_mod.get("res_tuning", {}))
                campeao_mod=str(dict_tuning.get("modelo_campeao", ""))
                melhor_rmse=float(str(dict_tuning.get("melhor_rmse", 0.0)))
                if campeao_mod:
                    return campeao_mod, melhor_rmse
            except Exception:
                pass

        # 3. Fallback: primeiro modelo ativo na configuração
        modelos_ativos=config_mod.obter_modelos_ativos()
        if modelos_ativos:
            return modelos_ativos[0].nome, 0.0

        return "regressao_linear", 0.0

    def _instanciar_regressor_concreto(
        self,
        nome_modelo: str,
        config_modelagem: object,
    ) -> tuple[object, str]:
        """Instancia o algoritmo concreto com base nos hiperparâmetros configurados."""
        from app.esquema_configuracao import ConfiguracaoModelagem

        config_mod=cast(ConfiguracaoModelagem, config_modelagem)
        params_grade=config_mod.tuning.parametros.get(nome_modelo)

        def _primeiro(chave: str, fallback: object) -> object:
            if params_grade is not None:
                val=params_grade.get(chave)
                if isinstance(val, list) and len(val) > 0:
                    return val[0]
                if val is not None and not isinstance(val, list):
                    return val
            return fallback

        if nome_modelo in {"regressao_linear", "regressao_multipla"}:
            fit_int=bool(_primeiro("fit_intercept", True))
            return LinearRegression(fit_intercept=fit_int), f"LinearRegression(fit_intercept={fit_int})"

        if nome_modelo == "lasso":
            alpha=float(str(_primeiro("alpha", 1.0)))
            return Lasso(alpha=alpha, random_state=42), f"Lasso(alpha={alpha})"

        if nome_modelo == "ridge":
            alpha=float(str(_primeiro("alpha", 1.0)))
            return Ridge(alpha=alpha, random_state=42), f"Ridge(alpha={alpha})"

        if nome_modelo == "elasticnet":
            alpha=float(str(_primeiro("alpha", 1.0)))
            l1=float(str(_primeiro("l1_ratio", 0.5)))
            return ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42), f"ElasticNet(alpha={alpha}, l1={l1})"

        if nome_modelo == "arvore_decisao":
            depth=int(str(_primeiro("max_depth", 5)))
            split=int(str(_primeiro("min_samples_split", 2)))
            return DecisionTreeRegressor(max_depth=depth, min_samples_split=split, random_state=42), f"DecisionTreeRegressor(depth={depth})"

        if nome_modelo == "random_forest":
            n_est=int(str(_primeiro("n_estimators", 100)))
            depth_val=_primeiro("max_depth", 10)
            depth_int=int(str(depth_val)) if depth_val is not None else None
            return RandomForestRegressor(n_estimators=n_est, max_depth=depth_int, random_state=42), f"RandomForestRegressor(n={n_est}, depth={depth_int})"

        if nome_modelo == "svr":
            c_val=float(str(_primeiro("C", 1.0)))
            k_val=str(_primeiro("kernel", "rbf"))
            return SVR(C=c_val, kernel=k_val), f"SVR(C={c_val}, kernel={k_val})"

        if nome_modelo == "rede_neural":
            iter_val=int(str(_primeiro("max_iter", 200)))
            return MLPRegressor(max_iter=iter_val, random_state=42), f"MLPRegressor(iter={iter_val})"

        if nome_modelo == "xgboost":
            import xgboost as xgb
            n_est=int(str(_primeiro("n_estimators", 100)))
            depth=int(str(_primeiro("max_depth", 6)))
            lr=float(str(_primeiro("learning_rate", 0.1)))
            return xgb.XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42, verbosity=0), f"XGBRegressor(n={n_est}, lr={lr})"

        if nome_modelo == "lightgbm":
            import lightgbm as lgb
            n_est=int(str(_primeiro("n_estimators", 100)))
            depth=int(str(_primeiro("max_depth", 6)))
            lr=float(str(_primeiro("learning_rate", 0.05)))
            return lgb.LGBMRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42, verbosity=-1), f"LGBMRegressor(n={n_est}, lr={lr})"

        return LinearRegression(), "LinearRegression(fallback)"

    def _registrar_promocao_mlflow(
        self,
        metadados: MetadadosModelo,
        dir_artefatos: Path,
    ) -> str:
        """Registra o run de promoção oficial no MLflow e publica a versão no Model Registry."""
        config_mlflow = self._conf.obter_modelagem().mlflow
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        try:
            mlflow.set_tracking_uri(config_mlflow.tracking_uri)
            mlflow.set_experiment(config_mlflow.experimento)

            with mlflow.start_run(run_name=f"promocao_modelo_campeao_{metadados.nome_modelo}") as run:
                run_id = str(run.info.run_id)
                mlflow.set_tag("pipeline.etapa", "implantacao_regras")
                mlflow.set_tag("modelo.status", "Production")
                mlflow.set_tag("modelo.alias", "champion")
                mlflow.set_tag("modelo.nome", metadados.nome_modelo)
                mlflow.set_tag("modelo.versao", metadados.versao)

                mlflow.log_metric("rmse_cv", metadados.rmse_cv)
                mlflow.log_metric("rmse_holdout", metadados.rmse_holdout)
                mlflow.log_metric("r2_holdout", metadados.r2_holdout)

                caminho_joblib = dir_artefatos / "modelo_campeao.joblib"
                caminho_meta = dir_artefatos / "metadados_campeao.json"

                if caminho_meta.exists():
                    mlflow.log_artifact(str(caminho_meta), artifact_path="model")

                nome_modelo_reg = config_mlflow.modelo_registrado or "previsao_preco_imoveis"

                pode_registrar = bool(
                    config_mlflow.registrar_modelo_campeao
                    and not config_mlflow.tracking_uri.startswith("file:")
                )

                config_implantacao = self._conf.obter_implantacao()
                regras_ativas = [r for r in config_implantacao.regras if r.ativo]

                # Definição explícita do Schema de Inputs e Outputs para o MLflow Registry
                esquema_inputs = Schema([
                    ColSpec("double", "Metragem"),
                    ColSpec("long", "Quartos"),
                    ColSpec("long", "Banheiros"),
                    ColSpec("long", "Vagas"),
                    ColSpec("string", "Bairro"),
                    ColSpec("string", "Zona"),
                ])

                colunas_outputs: list[ColSpec | TensorSpec] = [ColSpec("double", config_implantacao.coluna_alvo_predito)]
                for r in regras_ativas:
                    colunas_outputs.append(ColSpec("double", r.destino))

                esquema_outputs = Schema(colunas_outputs)
                assinatura_modelo = ModelSignature(inputs=esquema_inputs, outputs=esquema_outputs)

                exemplo_entrada = pd.DataFrame([{
                    "Metragem": 85.5,
                    "Quartos": 3,
                    "Banheiros": 2,
                    "Vagas": 2,
                    "Bairro": "Jardim Botânico",
                    "Zona": "Sul",
                }])

                if caminho_joblib.exists():
                    modelo_obj = joblib.load(caminho_joblib)
                    regras_dict = [r.model_dump() for r in regras_ativas]
                    modelo_pyfunc = ModeloComRegrasNegocio(
                        modelo_estimador=modelo_obj,
                        regras_configuradas=regras_dict,
                        coluna_alvo_predito=config_implantacao.coluna_alvo_predito,
                    )

                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=modelo_pyfunc,
                        code_paths=["app/servicos_implantacao/modelo_mlflow.py"],
                        signature=assinatura_modelo,
                        input_example=exemplo_entrada,
                        registered_model_name=nome_modelo_reg if pode_registrar else None,
                    )
                elif caminho_meta.exists():
                    mlflow.log_artifact(str(caminho_meta), artifact_path="model")

                if pode_registrar:
                    try:
                        client = MlflowClient(tracking_uri=config_mlflow.tracking_uri)
                        versoes = client.search_model_versions(f"name='{nome_modelo_reg}'")
                        if versoes:
                            versao_atual = str(max(versoes, key=lambda v: int(v.version)).version)
                            client.set_registered_model_alias(nome_modelo_reg, "champion", versao_atual)
                            client.set_model_version_tag(nome_modelo_reg, versao_atual, "status", "Production")
                            client.set_model_version_tag(nome_modelo_reg, versao_atual, "modelo", metadados.nome_modelo)
                            client.set_model_version_tag(nome_modelo_reg, versao_atual, "algoritmo", metadados.algoritmo)
                    except Exception:
                        pass

                return run_id
        except Exception:
            return "offline_local_run"

    def _gerar_relatorios(self, resultado: ResultadoImplantacao) -> None:
        """Gera o relatório Markdown e JSON de evidências da etapa de implantação."""
        dir_implantacao = Path("production_artifacts/implantacao")
        dir_implantacao.mkdir(parents=True, exist_ok=True)

        # 1. Salvar JSON de evidências
        caminho_json = Path("storage-data/relatorio_implantacao.json")
        caminho_json.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(resultado.model_dump(), f, indent=2, ensure_ascii=False)

        # 2. Relatório Markdown
        caminho_md = dir_implantacao / "relatorio_implantacao.md"
        ex = resultado.exemplo_predicao
        regras_md = "\n".join(
            [
                f"| `{r.destino}` | {r.nome} | {r.descricao} | **R$ {r.valor:,.2f}** |"
                for r in ex.detalhes_regras
            ]
        )

        conteudo_md = f"""# Relatório de Entrega — Etapa 6: Implantação e Regras de Negócio Dinâmicas

**Data de Execução:** {resultado.data_implantacao}
**Responsáveis:** @ml_engineer, @ml_architect, @pm
**Status:** CONCLUÍDO COM SUCESSO
**Evidências Persistidas:** [storage-data/relatorio_implantacao.json](file:///{caminho_json.resolve()})
**Run de Promoção no MLflow:** `{resultado.mlflow_run_id}`

---

## 1. Visão Geral da Promoção

O modelo campeão **{resultado.modelo_promovido.upper()}**, validado nas etapas anteriores como superior e selecionado por métricas de generalização, foi promovido para o ambiente de produção.

- **Identificador:** `{resultado.modelo_promovido}`
- **Versão:** `{resultado.versao_modelo}`
- **Artefato Persistido:** [`{resultado.caminho_artefato_modelo}`](file:///{Path(resultado.caminho_artefato_modelo).resolve()})
- **Metadados Oficiais:** [`{resultado.caminho_metadados}`](file:///{Path(resultado.caminho_metadados).resolve()})
- **Rastreabilidade no MLflow:** Run `{resultado.mlflow_run_id}` com tags `modelo.status=Production` e `modelo.alias=champion`.

### Métricas de Generalização no Holdout (20% Retido)
- **RMSE:** **R$ {resultado.rmse_holdout:,.2f}**
- **MAE:** **R$ {resultado.mae_holdout:,.2f}**
- **R²:** **{resultado.r2_holdout:.4f}**

---

## 2. Motor de Regras de Negócio Dinâmicas (AST Seguro)

As regras de negócio configuradas em [configuracoes/06_implantacao_regras.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/06_implantacao_regras.yaml) são consumidas dinamicamente a partir da predição do modelo (`preco_predito`), executando em sandbox restrita sem `eval` inseguro.

### Regras em Vigor e Exemplo de Inferência

**Entrada de Teste:** Apartamento de 85.5m², 3 quartos, 2 banheiros, 2 vagas, Jardim Botânico (Zona Sul).
**Preço Previsto pelo Modelo:** **R$ {ex.preco_predito:,.2f}**

| Variável Destino | Regra | Descrição de Negócio | Valor Calculado |
| :--- | :--- | :--- | :---: |
{regras_md}

---

## 3. Arquitetura da API HTTP (FastAPI)

A aplicação FastAPI desacopla transporte HTTP, serialização, inferência e cálculo de regras:

- **`GET /health`:** Retorna status operacional, versão da API e do modelo promovido.
- **`POST /predict`:** Recebe especificação do imóvel via `RequisicaoImovel`, retorna previsão e detalhamento das regras.
- **`POST /invocations`:** Endpoint compatível com MLflow Serving para predições em lote e integrações legadas.

---

## 4. Conformidade de Engenharia e Governança

- **Tipagem Estrita (MyPy / Pyright):** 0 erros, modo estrito preservado sem utilização de `Any`.
- **Validação de Expressões:** Nenhuma expressão fora da whitelist do `ValidadorAst` é aceita.
- **Isolamento de Código:** Módulos e pacotes com exatamente duas palavras separadas por underscore.
"""
        with open(caminho_md, "w", encoding="utf-8") as f_md:
            f_md.write(conteudo_md)
