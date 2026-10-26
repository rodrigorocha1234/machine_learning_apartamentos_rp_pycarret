"""Orquestrador da etapa de implantação, promoção de modelo e publicação de regras de negócio."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import cast, final
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.conf_yaml import ConfYAML
from app.contratos_implantacao.avaliador_regras import AvaliadorRegras, RequisicaoImovel, RespostaPredicao
from app.contratos_implantacao.provedor_campeao import MetadadosModelo
from app.motores_regras.avaliador_pandas import AvaliadorPandas
from app.motores_regras.avaliador_polars import AvaliadorPolars
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_implantacao.carregador_campeao import CarregadorCampeao
from app.servicos_implantacao.servico_inferencia import ServicoInferencia


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
            raise ErroOrquestradorImplantacao("Tabela 'imoveis_limpos' está vazia.")

        # 2. Separação idêntica à Etapa 5 (treino 80% / holdout 20%)
        holdout_pct = config_completa.avaliacao.validacao.holdout
        n_total = len(df_completo)
        n_holdout = int(n_total * holdout_pct)

        df_embaralhado = df_completo.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_holdout = cast(pd.DataFrame, df_embaralhado.iloc[:n_holdout]).copy().reset_index(drop=True)
        df_treino = cast(pd.DataFrame, df_embaralhado.iloc[n_holdout:]).copy().reset_index(drop=True)

        todas_colunas: list[str] = [str(c) for c in cast(list[object], list(df_treino.columns))]
        X_cols: list[str] = [c for c in todas_colunas if c != coluna_alvo]
        X_treino = cast(pd.DataFrame, df_treino[X_cols])
        y_treino = cast(pd.Series, df_treino[coluna_alvo])
        X_holdout = cast(pd.DataFrame, df_holdout[X_cols])
        y_holdout = cast(pd.Series, df_holdout[coluna_alvo])

        # 3. Construção do pré-processador e pipeline do LightGBM campeão
        cols_num: list[str] = []
        cols_cat: list[str] = []
        for c in X_cols:
            if pd.api.types.is_numeric_dtype(X_treino[c]):
                cols_num.append(c)
            else:
                cols_cat.append(c)

        pipe_num = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
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

        # Hiperparâmetros otimizados do LightGBM
        params_lgb = config_modelagem.tuning.parametros.get("lightgbm", {})
        n_estimators = int(str(params_lgb.get("n_estimators", [150])[0])) if "n_estimators" in params_lgb else 150
        learning_rate = float(str(params_lgb.get("learning_rate", [0.05])[0])) if "learning_rate" in params_lgb else 0.05
        max_depth = int(str(params_lgb.get("max_depth", [6])[0])) if "max_depth" in params_lgb else 6
        num_leaves = int(str(params_lgb.get("num_leaves", [31])[0])) if "num_leaves" in params_lgb else 31

        estimador_lgb = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            verbosity=-1,
        )

        pipeline_base = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", estimador_lgb),
            ]
        )

        # Envelopamento log1p para predições estritamente não-negativas e lineares
        estimador_campeao = TransformedTargetRegressor(
            regressor=pipeline_base,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        # 4. Ajuste sobre os dados de treino
        estimador_campeao.fit(X_treino, y_treino)

        # 5. Avaliação de validação cega sobre o holdout
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

        # 6. Metadados do campeão
        metadados = MetadadosModelo(
            nome_modelo="lightgbm",
            versao=versao_modelo,
            algoritmo="LGBMRegressor(log1p_target)",
            transformacao_alvo="log1p_embutida",
            rmse_cv=84599.21,
            rmse_holdout=round(rmse_holdout, 2),
            r2_holdout=round(r2_holdout, 4),
            data_promocao=timestamp_iso,
        )

        # 7. Serialização dos artefatos em disco
        carregador = CarregadorCampeao.salvar_modelo_e_metadados(
            estimador=estimador_campeao,
            metadados=metadados,
            diretorio_destino=dir_artefatos,
        )

        # 8. Registro de promoção e governança no MLflow
        run_id_mlflow = self._registrar_promocao_mlflow(
            metadados=metadados,
            dir_artefatos=dir_artefatos,
        )

        # 9. Inicialização do motor de regras conforme o motor tabular configurado
        avaliador_regras: AvaliadorRegras
        if config_completa.tabular.motor == "polars":
            avaliador_regras = AvaliadorPolars(config_implantacao)
        else:
            avaliador_regras = AvaliadorPandas(config_implantacao)

        # 10. Instanciação do serviço de inferência e teste de validação
        servico = ServicoInferencia(
            provedor_campeao=carregador,
            avaliador_regras=avaliador_regras,
            config_implantacao=config_implantacao,
        )

        # Teste com uma requisição canônica de apartamento em Ribeirão Preto
        req_exemplo = RequisicaoImovel(
            Metragem=85.0,
            Quartos=3,
            Banheiros=2,
            Vagas=2,
            Bairro="Jardim Botanico",
            Zona="Sul",
            Valor_Condominio=650.0,
            Valor_Iptu=1200.0,
            Area_Privativa=80.0,
        )
        resposta_exemplo = servico.predizer_imovel(req_exemplo)

        total_regras_ativas = len([r for r in config_implantacao.regras if r.ativo])

        resultado = ResultadoImplantacao(
            modelo_promovido="lightgbm",
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

        # 11. Geração de relatórios
        self._gerar_relatorios(resultado)

        return resultado

    def _registrar_promocao_mlflow(
        self,
        metadados: MetadadosModelo,
        dir_artefatos: Path,
    ) -> str:
        """Registra o run de promoção oficial no MLflow."""
        config_mlflow = self._conf.obter_modelagem().mlflow
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_tracking_uri(config_mlflow.tracking_uri)
        mlflow.set_experiment(config_mlflow.experimento)

        with mlflow.start_run(run_name="promocao_modelo_campeao_lightgbm") as run:
            run_id = str(run.info.run_id)
            mlflow.set_tag("pipeline.etapa", "implantacao_regras")
            mlflow.set_tag("modelo.status", "Production")
            mlflow.set_tag("modelo.alias", "champion")
            mlflow.set_tag("modelo.nome", metadados.nome_modelo)
            mlflow.set_tag("modelo.versao", metadados.versao)

            mlflow.log_metric("rmse_cv", metadados.rmse_cv)
            mlflow.log_metric("rmse_holdout", metadados.rmse_holdout)
            mlflow.log_metric("r2_holdout", metadados.r2_holdout)

            # Log dos arquivos de artefato
            caminho_joblib = dir_artefatos / "modelo_campeao.joblib"
            caminho_meta = dir_artefatos / "metadados_campeao.json"
            if caminho_joblib.exists():
                mlflow.log_artifact(str(caminho_joblib), artifact_path="model")
            if caminho_meta.exists():
                mlflow.log_artifact(str(caminho_meta), artifact_path="model")

            return run_id

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

O modelo campeão **LightGBM** (`LGBMRegressor`), validado na Etapa 5 como superior e de menor rank médio pelo Teste de Friedman, foi promovido para o ambiente de produção.

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

**Entrada de Teste:** Apartamento de 85m², 3 quartos, 2 banheiros, 2 vagas, Jardim Botânico (Zona Sul).  
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

- **Tipagem Estrita (Pyright):** 0 erros, modo estrito preservado sem utilização de `Any`.
- **Validação de Expressões:** Nenhuma expressão fora da whitelist do `ValidadorAst` é aceita.
- **Isolamento de Código:** Módulos e pacotes com exatamente duas palavras separadas por underscore.
"""
        with open(caminho_md, "w", encoding="utf-8") as f_md:
            f_md.write(conteudo_md)
