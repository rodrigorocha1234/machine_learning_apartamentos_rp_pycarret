"""Implementação do motor de modelagem e tuning baseado no Scikit-Learn nativo."""

from __future__ import annotations

import time
from typing import Protocol, cast, final, override
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score  # pyright: ignore[reportUnknownVariableType]
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

from app.contratos_modelagem.motor_treinador import (
    MotorTreinador,
    ResultadoTreinoModelo,
    ResultadoTuning,
)
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.fabrica_folds import FabricaFolds


class EstimadorTreinado(Protocol):
    """Protocolo estrutural para estimadores treinados com método predict."""

    def predict(self, X: pd.DataFrame) -> object: ...


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
        for col in cast(list[object], list(X.columns)):
            c = str(col)
            if pd.api.types.is_numeric_dtype(X[c]):
                cols_numericas.append(c)
            else:
                cols_categoricas.append(c)

        preprocessor = self._construir_preprocessador(cols_numericas, cols_categoricas, config)
        cv = self._fabrica_folds.criar_divisor(config.validacao_cruzada, folds_fallback=folds)

        modelos_treinados: list[ResultadoTreinoModelo] = []

        for item_modelo in config.modelos:
            if not item_modelo.ativo:
                continue

            inicio_modelo = time.perf_counter()
            estimador_base = self._instanciar_estimador(item_modelo.nome)
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
            grade_configurada = config.tuning.parametros.get(item_modelo.nome, {})
            grade_sklearn: dict[str, list[object]] = {
                f"{prefixo_param}{k}": cast(list[object], v)
                for k, v in grade_configurada.items()
            }

            if config.tuning.ativo and grade_sklearn:
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
                    val_score = cast(object, busca.best_score_)
                    melhor_score = -float(str(val_score)) if "neg_" in scoring_sklearn else float(str(val_score))
                    melhores_params_raw = cast(dict[str, object], busca.best_params_)
                    melhores_params: dict[str, object] = {
                        str(k).removeprefix(prefixo_param): v
                        for k, v in melhores_params_raw.items()
                    }

                    if config.tuning.manter_melhor:
                        scores_base_raw: object = cast(
                            object,
                            cross_val_score(
                                estimador_final, X, y, scoring=scoring_sklearn, cv=cv, n_jobs=1
                            ),
                        )
                        scores_base: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(
                            scores_base_raw, dtype=np.float64
                        )
                        score_base_medio = float(np.mean(scores_base))
                        score_busca = float(str(val_score))
                        # Maior é melhor em scikit-learn scorers (ex: -100 > -120 ou R2 0.85 > 0.80)
                        if score_base_medio > score_busca:
                            estimador_final.fit(X, y)
                            melhor_estimador = cast(EstimadorTreinado, estimador_final)
                            melhor_score = (
                                -score_base_medio if "neg_" in scoring_sklearn else score_base_medio
                            )
                            melhores_params = {}
                except Exception as erro_fit:
                    raise ErroTreinadorSklearn(
                        f"Falha no tuning do modelo '{item_modelo.nome}': {erro_fit}"
                    ) from erro_fit
            else:
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
                    melhores_params = {}
                except Exception as erro_fit_direto:
                    raise ErroTreinadorSklearn(
                        f"Falha no treino direto do modelo '{item_modelo.nome}': {erro_fit_direto}"
                    ) from erro_fit_direto

            duracao = time.perf_counter() - inicio_modelo

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

    def _construir_preprocessador(
        self,
        cols_numericas: list[str],
        cols_categoricas: list[str],
        config: ConfiguracaoModelagem,
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

        pipeline_numerico = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=est_num,
                        fill_value=config.imputacao.valor_constante_numerico
                        if est_num == "constant"
                        else None,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

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

    def _instanciar_estimador(self, nome_modelo: str) -> object:
        """Instancia o modelo concreto do catálogo de regressão."""
        if nome_modelo in {"regressao_linear", "regressao_multipla"}:
            return LinearRegression()
        if nome_modelo == "lasso":
            return Lasso(random_state=42)
        if nome_modelo == "ridge":
            return Ridge(random_state=42)
        if nome_modelo == "elasticnet":
            return ElasticNet(random_state=42)
        if nome_modelo == "arvore_decisao":
            return RandomForestRegressor(n_estimators=1, max_depth=5, random_state=42)
        if nome_modelo == "random_forest":
            return RandomForestRegressor(n_estimators=50, random_state=42)
        if nome_modelo == "svr":
            return SVR()
        if nome_modelo == "rede_neural":
            return MLPRegressor(max_iter=200, random_state=42)
        if nome_modelo == "xgboost":
            return xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        if nome_modelo == "lightgbm":
            return lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)

        raise ErroTreinadorSklearn(f"Modelo não suportado no motor sklearn: '{nome_modelo}'.")
