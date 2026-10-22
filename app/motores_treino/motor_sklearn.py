from __future__ import annotations
from typing import final, override
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from app.contratos_dados.motor_treino import ModeloPreditor, MotorTreino, ValorMetricaParametro
from app.tipos_enumerados import (
    EstrategiaFold,
    MetodoTransformacao,
    MetodoTuning,
    TipoCodificador,
    TipoEscalonador,
    TipoModelo,
)

class ErroMotorTreino(Exception):
    """Exceção para falhas nos motores de treino."""
    pass

@final
class MotorSklearn(MotorTreino):
    """Trilha EXCLUSIVAMENTE scikit-learn com suporte completo aos 10 modelos configuráveis."""

    _DEFAULTS_MODELOS: list[dict[str, str | bool]] = [
        {"nome": TipoModelo.REGRESSAO_LINEAR.value, "sklearn_class": "LinearRegression", "ativo": True},
        {"nome": TipoModelo.LASSO.value, "sklearn_class": "Lasso", "ativo": True},
        {"nome": TipoModelo.RIDGE.value, "sklearn_class": "Ridge", "ativo": True},
        {"nome": TipoModelo.ELASTICNET.value, "sklearn_class": "ElasticNet", "ativo": True},
        {"nome": TipoModelo.ARVORE_DECISAO.value, "sklearn_class": "DecisionTreeRegressor", "ativo": True},
        {"nome": TipoModelo.RANDOM_FOREST.value, "sklearn_class": "RandomForestRegressor", "ativo": True},
        {"nome": TipoModelo.SVR.value, "sklearn_class": "SVR", "ativo": True},
        {"nome": TipoModelo.REDE_NEURAL.value, "sklearn_class": "MLPRegressor", "ativo": True},
        {"nome": TipoModelo.XGBOOST.value, "sklearn_class": "XGBRegressor", "ativo": True},
        {"nome": TipoModelo.LIGHTGBM.value, "sklearn_class": "LGBMRegressor", "ativo": True},
    ]

    def __init__(
        self,
        semente: int = 42,
        folds: int = 3,
        metodo_folds: str = EstrategiaFold.KFOLD.value,
        embaralhar_folds: bool = True,
        colunas_numericas: list[str] | None = None,
        colunas_categoricas: list[str] | None = None,
        preprocessamento_por_modelo: dict[str, dict[str, str]] | None = None,
        modelos_configurados: list[dict[str, str | bool]] | None = None,
        parametros_tuning: dict[str, dict[str, list[str | int | float | bool]]] | None = None,
        metodo_tuning: str = MetodoTuning.RANDOM.value,
        transformar_alvo: bool = False,
        metodo_transformacao_alvo: str = MetodoTransformacao.LOG1P.value,
        transformar_atributos: bool = False,
        metodo_transformacao_atributos: str = MetodoTransformacao.LOG1P.value,
        colunas_transformacao_atributos: list[str] | None = None,
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
        self._transformar_alvo = transformar_alvo
        self._metodo_transformacao_alvo = metodo_transformacao_alvo.lower().strip()
        self._transformar_atributos = transformar_atributos
        self._metodo_transformacao_atributos = metodo_transformacao_atributos.lower().strip()
        self._colunas_transformacao_atributos = colunas_transformacao_atributos
        self._melhor_estimador: ModeloPreditor | None = None
        self._nome_modelo_campeao: str = TipoModelo.RIDGE.value


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
        modelo_enum = TipoModelo.de_string(modelo_nome)

        if modelo_enum == TipoModelo.REGRESSAO_LINEAR:
            return LinearRegression()
        if modelo_enum == TipoModelo.LASSO:
            return Lasso(alpha=0.1, random_state=self._semente)
        if modelo_enum == TipoModelo.RIDGE:
            return Ridge(alpha=1.0, random_state=self._semente)
        if modelo_enum == TipoModelo.ELASTICNET:
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self._semente)
        if modelo_enum == TipoModelo.ARVORE_DECISAO:
            return DecisionTreeRegressor(random_state=self._semente)
        if modelo_enum == TipoModelo.RANDOM_FOREST:
            return RandomForestRegressor(n_estimators=50, random_state=self._semente)
        if modelo_enum == TipoModelo.SVR:
            return SVR()
        if modelo_enum == TipoModelo.REDE_NEURAL:
            return MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=self._semente)
        if modelo_enum == TipoModelo.XGBOOST:
            return XGBRegressor(n_estimators=50, random_state=self._semente)
        if modelo_enum == TipoModelo.LIGHTGBM:
            return LGBMRegressor(n_estimators=50, random_state=self._semente, verbose=-1)

        raise ErroMotorTreino(f"Modelo não suportado no motor Sklearn: '{modelo_nome}'")

    def _criar_pipeline(self, modelo_tipo: str, colunas_num: list[str], colunas_cat: list[str]) -> Pipeline:
        config_modelo = self._preprocessamento_por_modelo.get(modelo_tipo, {})
        valor_escalonador_raw = config_modelo.get("escalonador", TipoEscalonador.STANDARD.value)
        tipo_escalonador = TipoEscalonador.de_string(valor_escalonador_raw)
        
        transformadores: list[tuple[str, Pipeline, list[str]]] = []

        if colunas_num:
            if self._transformar_atributos:
                colunas_alvo_log = (
                    [c for c in colunas_num if any(c.lower() == ca.lower() for ca in self._colunas_transformacao_atributos)]
                    if self._colunas_transformacao_atributos
                    else list(colunas_num)
                )
                colunas_sem_log = [c for c in colunas_num if c not in colunas_alvo_log]

                metodo_transf_attr = MetodoTransformacao.de_string(self._metodo_transformacao_atributos)
                func_transf = np.log if metodo_transf_attr == MetodoTransformacao.LOG else np.log1p
                inv_transf = np.exp if metodo_transf_attr == MetodoTransformacao.LOG else np.expm1

                if colunas_alvo_log:
                    passos_num_log: list[tuple[str, BaseEstimator]] = [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(func=func_transf, inverse_func=inv_transf, check_inverse=False)),
                    ]
                    if tipo_escalonador == TipoEscalonador.STANDARD:
                        passos_num_log.append(("scaler", StandardScaler()))
                    transformadores.append(("num_log", Pipeline(passos_num_log), colunas_alvo_log))

                if colunas_sem_log:
                    passos_num_sem_log: list[tuple[str, BaseEstimator]] = [
                        ("imputer", SimpleImputer(strategy="median"))
                    ]
                    if tipo_escalonador == TipoEscalonador.STANDARD:
                        passos_num_sem_log.append(("scaler", StandardScaler()))
                    transformadores.append(("num_direto", Pipeline(passos_num_sem_log), colunas_sem_log))
            else:
                passos_num: list[tuple[str, BaseEstimator]] = [
                    ("imputer", SimpleImputer(strategy="median"))
                ]
                if tipo_escalonador == TipoEscalonador.STANDARD:
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

        regressor_final: BaseEstimator
        if self._transformar_alvo:
            metodo_alvo = MetodoTransformacao.de_string(self._metodo_transformacao_alvo)
            if metodo_alvo in (MetodoTransformacao.YEO_JOHNSON, MetodoTransformacao.BOX_COX):
                metodo_pt = "yeo-johnson" if metodo_alvo == MetodoTransformacao.YEO_JOHNSON else "box-cox"
                regressor_final = TransformedTargetRegressor(
                    regressor=regressor,
                    transformer=PowerTransformer(method=metodo_pt)
                )
            elif metodo_alvo == MetodoTransformacao.LOG:
                regressor_final = TransformedTargetRegressor(
                    regressor=regressor,
                    func=np.log,
                    inverse_func=np.exp
                )
            else:
                regressor_final = TransformedTargetRegressor(
                    regressor=regressor,
                    func=np.log1p,
                    inverse_func=np.expm1
                )
        else:
            regressor_final = regressor

        return Pipeline([
            ("preprocessamento", transformador),
            ("modelo", regressor_final)
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
        prefixo_modelo = "modelo__regressor__" if self._transformar_alvo else "modelo__"

        grid_params = {f"{prefixo_modelo}{k}": v for k, v in grid_config.items()}
        if not grid_params:
            if nome_alvo == "ridge":
                grid_params = {f"{prefixo_modelo}alpha": [0.1, 1.0, 10.0]}
            elif nome_alvo == "lasso":
                grid_params = {f"{prefixo_modelo}alpha": [0.01, 0.1, 1.0]}
            else:
                grid_params = {}


        if grid_params:
            try:
                tuning_enum = MetodoTuning.de_string(metodo_escolhido)
            except ValueError as erro:
                raise ErroMotorTreino(
                    f"Método de tuning '{metodo_escolhido}' não suportado no motor Sklearn: {erro}"
                ) from erro

            if tuning_enum == MetodoTuning.GRID:
                busca = GridSearchCV(
                    pipeline,
                    grid_params,
                    cv=self._folds,
                    scoring="neg_root_mean_squared_error",
                )
            elif tuning_enum == MetodoTuning.HALVING and ("grid" in metodo_escolhido):
                busca = HalvingGridSearchCV(
                    pipeline,
                    grid_params,
                    cv=self._folds,
                    scoring="neg_root_mean_squared_error",
                    random_state=self._semente,
                )
            elif tuning_enum == MetodoTuning.HALVING:
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
            elif tuning_enum == MetodoTuning.RANDOM:
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
