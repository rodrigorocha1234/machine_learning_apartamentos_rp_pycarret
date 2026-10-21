from __future__ import annotations
from typing import final, override
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from app.contratos_dados.motor_treino import ModeloPreditor, MotorTreino, ValorMetricaParametro

class ErroMotorTreino(Exception):
    """Exceção para falhas nos motores de treino."""
    pass

@final
class MotorSklearn(MotorTreino):
    """Trilha EXCLUSIVAMENTE scikit-learn com suporte completo aos 10 modelos configuráveis."""

    _DEFAULTS_MODELOS: list[dict[str, str | bool]] = [
        {"nome": "regressao_linear", "sklearn_class": "LinearRegression", "ativo": True},
        {"nome": "lasso", "sklearn_class": "Lasso", "ativo": True},
        {"nome": "ridge", "sklearn_class": "Ridge", "ativo": True},
        {"nome": "elasticnet", "sklearn_class": "ElasticNet", "ativo": True},
        {"nome": "arvore_decisao", "sklearn_class": "DecisionTreeRegressor", "ativo": True},
        {"nome": "random_forest", "sklearn_class": "RandomForestRegressor", "ativo": True},
        {"nome": "svr", "sklearn_class": "SVR", "ativo": True},
        {"nome": "rede_neural", "sklearn_class": "MLPRegressor", "ativo": True},
        {"nome": "xgboost", "sklearn_class": "XGBRegressor", "ativo": True},
        {"nome": "lightgbm", "sklearn_class": "LGBMRegressor", "ativo": True},
    ]

    def __init__(
        self,
        semente: int = 42,
        folds: int = 3,
        metodo_folds: str = "kfold",
        embaralhar_folds: bool = True,
        colunas_numericas: list[str] | None = None,
        colunas_categoricas: list[str] | None = None,
        preprocessamento_por_modelo: dict[str, dict[str, str]] | None = None,
        modelos_configurados: list[dict[str, str | bool]] | None = None,
        parametros_tuning: dict[str, dict[str, list[str | int | float | bool]]] | None = None,
        metodo_tuning: str = "random",
    ) -> None:
        self._semente = semente
        self._folds = folds
        self._metodo_folds = metodo_folds
        self._embaralhar_folds = embaralhar_folds
        self._colunas_numericas = colunas_numericas
        self._colunas_categoricas = colunas_categoricas
        self._preprocessamento_por_modelo = preprocessamento_por_modelo or {}
        self._modelos_configurados = modelos_configurados or self._DEFAULTS_MODELOS
        self._parametros_tuning = parametros_tuning or {}
        self._metodo_tuning = metodo_tuning.lower().strip()
        self._melhor_estimador: ModeloPreditor | None = None
        self._nome_modelo_campeao: str = "ridge"

    def _resolver_colunas(self, x: pd.DataFrame) -> tuple[list[str], list[str]]:
        col_num = self._colunas_numericas
        if col_num is None:
            col_num = x.select_dtypes(include=["number"]).columns.tolist()
        else:
            col_num = [c for c in col_num if c in x.columns]

        col_cat = self._colunas_categoricas
        if col_cat is None:
            col_cat = x.select_dtypes(exclude=["number"]).columns.tolist()
        else:
            col_cat = [c for c in col_cat if c in x.columns]

        return col_num, col_cat

    def _instanciar_regressor(self, modelo_nome: str) -> BaseEstimator:
        nome_normalizado = modelo_nome.lower().strip()

        if nome_normalizado in ("regressao_linear", "linearregression", "lr"):
            return LinearRegression()
        if nome_normalizado in ("lasso",):
            return Lasso(alpha=0.1, random_state=self._semente)
        if nome_normalizado in ("ridge",):
            return Ridge(alpha=1.0, random_state=self._semente)
        if nome_normalizado in ("elasticnet", "en"):
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self._semente)
        if nome_normalizado in ("arvore_decisao", "decisiontreeregressor", "dt"):
            return DecisionTreeRegressor(random_state=self._semente)
        if nome_normalizado in ("random_forest", "randomforestregressor", "rf"):
            return RandomForestRegressor(n_estimators=50, random_state=self._semente)
        if nome_normalizado in ("svr", "svm"):
            return SVR()
        if nome_normalizado in ("rede_neural", "mlpregressor", "mlp"):
            return MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=self._semente)
        if nome_normalizado in ("xgboost", "xgbregressor"):
            return XGBRegressor(n_estimators=50, random_state=self._semente)
        if nome_normalizado in ("lightgbm", "lgbmregressor"):
            return LGBMRegressor(n_estimators=50, random_state=self._semente, verbose=-1)

        raise ErroMotorTreino(f"Modelo não suportado no motor Sklearn: '{modelo_nome}'")

    def _criar_pipeline(self, modelo_tipo: str, colunas_num: list[str], colunas_cat: list[str]) -> Pipeline:
        config_modelo = self._preprocessamento_por_modelo.get(modelo_tipo, {})
        tipo_escalonador = config_modelo.get("escalonador", "standard")
        
        transformadores: list[tuple[str, Pipeline, list[str]]] = []

        if colunas_num:
            passos_num: list[tuple[str, BaseEstimator]] = [
                ("imputer", SimpleImputer(strategy="median"))
            ]
            if tipo_escalonador == "standard":
                passos_num.append(("scaler", StandardScaler()))
            transformadores.append(("num", Pipeline(passos_num), colunas_num))

        if colunas_cat:
            passos_cat: list[tuple[str, BaseEstimator]] = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
            transformadores.append(("cat", Pipeline(passos_cat), colunas_cat))

        transformador = ColumnTransformer(transformers=transformadores, remainder="drop")
        regressor = self._instanciar_regressor(modelo_tipo)

        return Pipeline([
            ("preprocessamento", transformador),
            ("modelo", regressor)
        ])

    @override
    def avaliar_modelos(self, dados: pd.DataFrame, alvo: str) -> pd.DataFrame:
        if alvo not in dados.columns:
            raise ErroMotorTreino(f"Alvo '{alvo}' ausente no DataFrame.")

        x = dados.drop(columns=[alvo])
        y = dados[alvo].to_numpy()
        colunas_num, colunas_cat = self._resolver_colunas(x)

        modelos_ativos = [m for m in self._modelos_configurados if m.get("ativo", True)]
        if not modelos_ativos:
            modelos_ativos = [
                {"nome": "ridge", "ativo": True},
                {"nome": "random_forest", "ativo": True}
            ]

        kfold = KFold(n_splits=self._folds, shuffle=self._embaralhar_folds, random_state=self._semente)
        erros_por_modelo: dict[str, list[float]] = {str(m.get("nome")): [] for m in modelos_ativos}

        for indices_treino, indices_val in kfold.split(x):
            x_treino, x_val = x.iloc[indices_treino], x.iloc[indices_val]
            y_treino, y_val = y[indices_treino], y[indices_val]

            for mod in modelos_ativos:
                nome_mod = str(mod.get("nome"))
                pipe = self._criar_pipeline(nome_mod, colunas_num, colunas_cat)
                pipe.fit(x_treino, y_treino)
                pred = pipe.predict(x_val)
                rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
                erros_por_modelo[nome_mod].append(rmse)

        return pd.DataFrame(erros_por_modelo)

    @override
    def ajustar_parametros(
        self,
        dados: pd.DataFrame,
        alvo: str,
        iteracoes: int,
        modelo_nome: str | None = None,
        metodo_tuning: str | None = None,
    ) -> dict[str, ValorMetricaParametro]:
        if alvo not in dados.columns:
            raise ErroMotorTreino(f"Alvo '{alvo}' ausente no DataFrame.")

        nome_alvo = modelo_nome or self._nome_modelo_campeao or "ridge"
        self._nome_modelo_campeao = nome_alvo
        metodo_escolhido = (metodo_tuning or self._metodo_tuning).lower().strip()

        x = dados.drop(columns=[alvo])
        y = dados[alvo]
        colunas_num, colunas_cat = self._resolver_colunas(x)

        pipeline = self._criar_pipeline(nome_alvo, colunas_num, colunas_cat)
        grid_config = self._parametros_tuning.get(nome_alvo, {})

        grid_params = {f"modelo__{k}": v for k, v in grid_config.items()}
        if not grid_params:
            if nome_alvo == "ridge":
                grid_params = {"modelo__alpha": [0.1, 1.0, 10.0]}
            elif nome_alvo == "lasso":
                grid_params = {"modelo__alpha": [0.01, 0.1, 1.0]}
            else:
                grid_params = {}

        if grid_params:
            if metodo_escolhido in ("grid", "grid_search"):
                busca = GridSearchCV(
                    pipeline,
                    grid_params,
                    cv=self._folds,
                    scoring="neg_root_mean_squared_error",
                )
            elif metodo_escolhido in ("halving_grid", "halving_grid_search"):
                busca = HalvingGridSearchCV(
                    pipeline,
                    grid_params,
                    cv=self._folds,
                    scoring="neg_root_mean_squared_error",
                    random_state=self._semente,
                )
            elif metodo_escolhido in ("halving_random", "halving_random_search"):
                total_combinacoes = int(np.prod([len(v) for v in grid_params.values()]))
                n_candidates = min(iteracoes, total_combinacoes)
                busca = HalvingRandomSearchCV(
                    pipeline,
                    grid_params,
                    n_candidates=n_candidates,
                    cv=self._folds,
                    scoring="neg_root_mean_squared_error",
                    random_state=self._semente,
                )
            elif metodo_escolhido in ("random", "random_search", "randomized"):
                total_combinacoes = int(np.prod([len(v) for v in grid_params.values()]))
                n_iter = min(iteracoes, total_combinacoes)
                busca = RandomizedSearchCV(
                    pipeline,
                    grid_params,
                    n_iter=n_iter,
                    cv=self._folds,
                    scoring="neg_root_mean_squared_error",
                    random_state=self._semente,
                )
            else:
                raise ErroMotorTreino(
                    f"Método de tuning '{metodo_escolhido}' não suportado no motor Sklearn. "
                    "Válidos: 'random', 'grid', 'halving_random', 'halving_grid'."
                )

            busca.fit(x, y)
            self._melhor_estimador = busca.best_estimator_
            return {
                "modelo": nome_alvo,
                "melhor_score": float(busca.best_score_),
                "melhores_parametros": busca.best_params_
            }
        else:
            pipeline.fit(x, y)
            self._melhor_estimador = pipeline
            return {
                "modelo": nome_alvo,
                "melhor_score": 0.0,
                "melhores_parametros": {}
            }

    @override
    def treinar_final(
        self,
        dados: pd.DataFrame,
        alvo: str,
        modelo_nome: str | None = None
    ) -> ModeloPreditor:
        if alvo not in dados.columns:
            raise ErroMotorTreino(f"Alvo '{alvo}' ausente no DataFrame.")

        nome_alvo = modelo_nome or self._nome_modelo_campeao or "ridge"
        x = dados.drop(columns=[alvo])
        y = dados[alvo]
        colunas_num, colunas_cat = self._resolver_colunas(x)

        if self._melhor_estimador is None:
            pipeline = self._criar_pipeline(nome_alvo, colunas_num, colunas_cat)
            pipeline.fit(x, y)
            self._melhor_estimador = pipeline
        else:
            if hasattr(self._melhor_estimador, "fit"):
                self._melhor_estimador.fit(x, y)

        return self._melhor_estimador
