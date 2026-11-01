"""Implementação do motor de modelagem e tuning baseado no Scikit-Learn nativo."""

from __future__ import annotations

import time
from typing import Literal, Protocol, TypeAlias, cast, final, override
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score  # pyright: ignore[reportUnknownVariableType]
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
import xgboost as xgb

from app.contratos_modelagem.motor_treinador import (
    MotorTreinador,
    ResultadoTreinoModelo,
    ResultadoTuning,
)
from app.esquema_configuracao import ConfiguracaoModelagem, ValorParametro
from app.motores_modelagem.fabrica_folds import FabricaFolds
from app.servicos_observabilidade.publicador_modelos_grafana import (
    PublicadorModelosGrafana,
    formatar_nome_modelo,
)

TipoEscaladorSklearn: TypeAlias = (
    StandardScaler | MinMaxScaler | RobustScaler | MaxAbsScaler | Literal["passthrough"]
)


class EstimadorTreinado(Protocol):
    """Protocolo estrutural para estimadores treinados com método predict."""

    def predict(
        self, X: pd.DataFrame
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | pd.Series | list[float]: ...


class ErroTreinadorSklearn(Exception):
    """Exceção levantada em falhas durante o treinamento ou tuning com Scikit-Learn."""


@final
class TreinadorSklearn(MotorTreinador):
    """Motor de treinamento e busca de hiperparâmetros nativo via Scikit-Learn Pipelines."""

    def __init__(self, fabrica_folds: FabricaFolds | None = None) -> None:
        self._fabrica_folds = fabrica_folds or FabricaFolds()

    @override
    def obter_nome_motor(self) -> str:
        return "sklearn"

    @staticmethod
    def _extrair_parametros_estimador(
        estimador: object, chaves_grade: list[str]
    ) -> dict[str, ValorParametro]:
        """Extrai os hiperparâmetros concretos do estimador interno alinhados à grade de tuning."""
        obj: object = estimador
        if hasattr(obj, "regressor_"):
            obj = getattr(obj, "regressor_")
        elif hasattr(obj, "regressor"):
            obj = getattr(obj, "regressor")

        if hasattr(obj, "named_steps"):
            named_steps = getattr(obj, "named_steps")
            if isinstance(named_steps, dict) and "regressor" in named_steps:
                obj = named_steps["regressor"]

        if hasattr(obj, "get_params"):
            get_params_fn = getattr(obj, "get_params")
            if callable(get_params_fn):
                todos_params = cast(dict[str, ValorParametro], get_params_fn())
                return {k: todos_params[k] for k in chaves_grade if k in todos_params}
        return {}

    @override
    def treinar_e_ajustar(
        self,
        dados_treino: pd.DataFrame,
        coluna_alvo: str,
        config: ConfiguracaoModelagem,
        folds: int = 3,
    ) -> ResultadoTuning:
        if dados_treino.empty:
            raise ErroTreinadorSklearn("Conjunto de dados de treino está vazio.")
        if coluna_alvo not in dados_treino.columns:
            raise ErroTreinadorSklearn(f"Coluna alvo '{coluna_alvo}' ausente no DataFrame.")

        X = dados_treino.drop(columns=[coluna_alvo])
        y = cast(pd.Series, dados_treino[coluna_alvo])

        # 1. Identificação de colunas numéricas e categóricas
        cols_numericas: list[str] = []
        cols_categoricas: list[str] = []
        for col in X.columns:
            c = str(col)
            if pd.api.types.is_numeric_dtype(X[c]):
                cols_numericas.append(c)
            else:
                cols_categoricas.append(c)

        cv = self._fabrica_folds.criar_divisor(config.validacao_cruzada, folds_fallback=folds)

        modelos_treinados: list[ResultadoTreinoModelo] = []

        for item_modelo in config.obter_modelos_ativos():
            inicio_modelo = time.perf_counter()
            tipo_prep = item_modelo.obter_preprocessador()
            preprocessor = self._construir_preprocessador(
                cols_numericas, cols_categoricas, config, tipo_preprocessador=tipo_prep
            )
            params_iniciais = getattr(item_modelo, "parametros", {}) or {}
            estimador_base = self._instanciar_estimador(item_modelo.nome, parametros=params_iniciais)
            pipeline_estimador = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("regressor", estimador_base),
                ]
            )

            # Envelopamento opcional de transformação do alvo (log1p anti-skewness)
            if config.preprocessamento.transformar_alvo:
                estimador_final = TransformedTargetRegressor(
                    regressor=pipeline_estimador,
                    func=np.log1p,
                    inverse_func=np.expm1,
                )
                prefixo_param = "regressor__regressor__"
            else:
                estimador_final = pipeline_estimador
                prefixo_param = "regressor__"

            # Grade de busca
            grade_configurada = config.tuning.obter_grade_modelo(item_modelo.nome)
            grade_sklearn: dict[str, list[ValorParametro]] = {
                f"{prefixo_param}{k}": v
                for k, v in grade_configurada.items()
            }

            nome_formatado = formatar_nome_modelo(item_modelo.nome)
            if config.tuning.ativo and grade_sklearn:
                msg_tuning = f"realizando tuning de parâmetros {nome_formatado}"
                print(f"  → Realizando tuning de parâmetros {nome_formatado}...")
                PublicadorModelosGrafana.registrar_log(item_modelo.nome, msg_tuning)
                tuning_ativo = True
                mapa_scorers: dict[str, str] = {
                    "RMSE": "neg_root_mean_squared_error",
                    "MAE": "neg_mean_absolute_error",
                    "R2": "r2",
                    "RMSLE": "neg_mean_squared_log_error",
                }
                scoring_sklearn = mapa_scorers.get(
                    config.tuning.otimizar_metrica, "neg_root_mean_squared_error"
                )

                if config.tuning.metodo == "grid":
                    busca = GridSearchCV(
                        estimator=estimador_final,
                        param_grid=grade_sklearn,
                        scoring=scoring_sklearn,
                        cv=cv,
                        n_jobs=1,
                    )
                else:
                    n_iter = min(config.tuning.iteracoes, len(grade_sklearn) * 10)
                    n_iter = max(n_iter, 2)
                    busca = RandomizedSearchCV(
                        estimator=estimador_final,
                        param_distributions=grade_sklearn,
                        n_iter=n_iter,
                        scoring=scoring_sklearn,
                        cv=cv,
                        random_state=42,
                        n_jobs=1,
                    )

                try:
                    busca.fit(X, y)
                    melhor_estimador = cast(EstimadorTreinado, busca.best_estimator_)
                    val_score = float(busca.best_score_)
                    melhor_score = -val_score if "neg_" in scoring_sklearn else val_score
                    melhores_params_raw = cast(dict[str, ValorParametro], busca.best_params_)
                    melhores_params: dict[str, ValorParametro] = {
                        str(k).removeprefix(prefixo_param): v
                        for k, v in melhores_params_raw.items()
                    }

                    if config.tuning.manter_melhor:
                        scores_base_raw = cross_val_score(
                            estimador_final, X, y, scoring=scoring_sklearn, cv=cv, n_jobs=1
                        )
                        scores_base: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(
                            scores_base_raw, dtype=np.float64
                        )
                        score_base_medio = float(np.mean(scores_base))
                        score_busca = val_score
                        # Maior é melhor em scikit-learn scorers (ex: -100 > -120 ou R2 0.85 > 0.80)
                        if score_base_medio > score_busca:
                            estimador_final.fit(X, y)
                            melhor_estimador = cast(EstimadorTreinado, estimador_final)
                            melhor_score = (
                                -score_base_medio if "neg_" in scoring_sklearn else score_base_medio
                            )
                            # Extrai os parâmetros padrão do estimador vencedor para rastreabilidade completa
                            melhores_params = self._extrair_parametros_estimador(
                                melhor_estimador, list(grade_configurada.keys())
                            )
                except Exception as erro_fit:
                    raise ErroTreinadorSklearn(
                        f"Falha no tuning do modelo '{item_modelo.nome}': {erro_fit}"
                    ) from erro_fit
            else:
                msg_tuning = f"treinando modelo {nome_formatado} (tuning desativado)"
                print(f"  → Treinando modelo {nome_formatado} (tuning desativado)...")
                PublicadorModelosGrafana.registrar_log(item_modelo.nome, msg_tuning)
                tuning_ativo = False
                # Treinamento direto sem busca
                try:
                    estimador_final.fit(X, y)
                    melhor_estimador = cast(EstimadorTreinado, estimador_final)
                    preds_direto: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(
                        melhor_estimador.predict(X), dtype=np.float64
                    )
                    y_direto: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(
                        y.to_numpy(), dtype=np.float64
                    )
                    residuos_direto = y_direto - preds_direto
                    melhor_score = float(np.sqrt(np.mean(residuos_direto**2)))
                    melhores_params = self._extrair_parametros_estimador(
                        melhor_estimador, list(grade_configurada.keys())
                    )
                except Exception as erro_fit_direto:
                    raise ErroTreinadorSklearn(
                        f"Falha no treino direto do modelo '{item_modelo.nome}': {erro_fit_direto}"
                    ) from erro_fit_direto

            duracao = time.perf_counter() - inicio_modelo
            print(f"  ✓ {nome_formatado.capitalize()} concluído em {duracao:.2f}s | RMSE: R$ {melhor_score:,.2f}")

            # Cálculo de métricas adicionais
            preds_finais: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(
                melhor_estimador.predict(X), dtype=np.float64
            )
            y_arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(
                y.to_numpy(), dtype=np.float64
            )
            diferenca = y_arr - preds_finais
            mae = float(np.mean(np.abs(diferenca)))
            variancia_total = float(np.var(y_arr))
            r2 = (
                float(1.0 - (float(np.mean(diferenca**2)) / variancia_total))
                if variancia_total > 0
                else 0.0
            )

            res_modelo = ResultadoTreinoModelo(
                nome_modelo=item_modelo.nome,
                framework="sklearn",
                melhores_parametros=melhores_params,
                metricas={
                    "RMSE": round(melhor_score, 2),
                    "MAE": round(mae, 2),
                    "R2": round(r2, 4),
                },
                duracao_segundos=round(duracao, 4),
                tuning_ativo=tuning_ativo,
                mensagem_tuning=msg_tuning,
            )
            modelos_treinados.append(res_modelo)

        if not modelos_treinados:
            raise ErroTreinadorSklearn("Nenhum modelo ativo foi treinado com sucesso.")

        # Determinação do modelo campeão (menor RMSE)
        modelos_ordenados = sorted(
            modelos_treinados, key=lambda m: m.metricas.get("RMSE", float("inf"))
        )
        campeao = modelos_ordenados[0]

        return ResultadoTuning(
            metodo_tuning=config.tuning.metodo,
            iteracoes=config.tuning.iteracoes,
            modelos_treinados=modelos_treinados,
            modelo_campeao=campeao.nome_modelo,
            melhor_rmse=campeao.metricas.get("RMSE", 0.0),
        )

    @staticmethod
    def _obter_escalador(tipo_preprocessador: str) -> TipoEscaladorSklearn:
        """Mapeia o nome do pré-processador para o scaler do scikit-learn ou passthrough."""
        tipo = tipo_preprocessador.lower().strip()
        if tipo in {"standard", "standard_scaler", "padrao"}:
            return StandardScaler()
        if tipo in {"minmax", "min_max", "minmax_scaler"}:
            return MinMaxScaler()
        if tipo in {"robust", "robust_scaler"}:
            return RobustScaler()
        if tipo in {"maxabs", "max_abs", "maxabs_scaler"}:
            return MaxAbsScaler()
        if tipo in {"nenhum", "none", "passthrough", "sem_escala", "sem_scaler"}:
            return "passthrough"
        return StandardScaler()

    def _construir_preprocessador(
        self,
        cols_numericas: list[str],
        cols_categoricas: list[str],
        config: ConfiguracaoModelagem,
        tipo_preprocessador: str = "standard",
    ) -> ColumnTransformer:
        estrategia_num = config.imputacao.estrategia_numerica
        estrategia_cat = config.imputacao.estrategia_categorica

        if estrategia_num == "mais_frequente":
            est_num = "most_frequent"
        elif estrategia_num == "mediana":
            est_num = "median"
        elif estrategia_num == "media":
            est_num = "mean"
        else:
            est_num = "constant"

        if estrategia_cat == "mais_frequente":
            est_cat = "most_frequent"
        else:
            est_cat = "constant"

        passos_numericos: list[tuple[str, TransformerMixin | TipoEscaladorSklearn]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=est_num,
                    fill_value=config.imputacao.valor_constante_numerico
                    if est_num == "constant"
                    else None,
                ),
            ),
        ]

        escalador = self._obter_escalador(tipo_preprocessador)
        if escalador != "passthrough":
            passos_numericos.append(("scaler", escalador))

        pipeline_numerico = Pipeline(passos_numericos)

        pipeline_categorico = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=est_cat,
                        fill_value=config.imputacao.valor_constante_categorico
                        if est_cat == "constant"
                        else None,
                    ),
                ),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", pipeline_numerico, cols_numericas),
                ("cat", pipeline_categorico, cols_categoricas),
            ],
            remainder="drop",
        )

    def _instanciar_estimador(
        self,
        nome_modelo: str,
        parametros: dict[str, ValorParametro] | None = None,
    ) -> BaseEstimator:
        """Instancia o modelo concreto do catálogo de regressão sem hiperparâmetros fixos."""
        params = dict(parametros) if parametros else {}

        if nome_modelo in {"regressao_linear", "regressao_multipla"}:
            return LinearRegression(**params)
        if nome_modelo == "lasso":
            return Lasso(**params)
        if nome_modelo == "ridge":
            return Ridge(**params)
        if nome_modelo == "elasticnet":
            return ElasticNet(**params)
        if nome_modelo == "arvore_decisao":
            return DecisionTreeRegressor(**params)
        if nome_modelo == "random_forest":
            return RandomForestRegressor(**params)
        if nome_modelo == "svr":
            return SVR(**params)
        if nome_modelo == "rede_neural":
            return MLPRegressor(**params)
        if nome_modelo == "xgboost":
            if "verbosity" not in params:
                params["verbosity"] = 0
            return xgb.XGBRegressor(**params)
        if nome_modelo == "lightgbm":
            if "verbose" not in params:
                params["verbose"] = -1
            return lgb.LGBMRegressor(**params)  # type: ignore[arg-type]

        raise ErroTreinadorSklearn(f"Modelo não suportado no motor sklearn: '{nome_modelo}'.")
