"""Executor de validação cruzada pareada para comparação justa de modelos."""

from __future__ import annotations

from typing import Protocol, cast, final
import numpy as np
import pandas as pd
from sklearn.base import clone  # pyright: ignore[reportUnknownVariableType]
from sklearn.model_selection import KFold, RepeatedKFold

from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo
from app.esquema_configuracao import ConfiguracaoAvaliacao, ConfiguracaoModelagem
from app.motores_modelagem.treinador_sklearn import TreinadorSklearn
from app.servicos_modelagem.rastreador_mlflow import RastreadorMlflow


class EstimadorComPredict(Protocol):
    """Protocolo estrutural para estimador com métodos fit e predict."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...
    def predict(self, X: pd.DataFrame) -> object: ...


class ErroValidadorCruzado(Exception):
    """Exceção levantada em falhas durante a validação cruzada pareada."""


@final
class ValidadorCruzado:
    """Validador cruzado pareado que assegura avaliação nos mesmos splits de dados."""

    def __init__(
        self,
        config_avaliacao: ConfiguracaoAvaliacao,
        config_modelagem: ConfiguracaoModelagem,
        coluna_alvo: str = "Valor_da_Venda",
        rastreador_mlflow: RastreadorMlflow | None = None,
    ) -> None:
        self._config_av = config_avaliacao
        self._config_mod = config_modelagem
        self._coluna_alvo = coluna_alvo
        self._treinador_aux = TreinadorSklearn()
        self._rastreador_mlflow = rastreador_mlflow

    def executar_cv_pareada(
        self,
        dados_treino: pd.DataFrame,
        modelos_tuning: list[ResultadoTreinoModelo],
    ) -> tuple[dict[str, list[float]], dict[str, object]]:
        """Executa validação cruzada pareada com RepeatedKFold em todos os modelos selecionados."""
        if dados_treino.empty:
            raise ErroValidadorCruzado("DataFrame de treino para validação cruzada está vazio.")
        if self._coluna_alvo not in dados_treino.columns:
            raise ErroValidadorCruzado(f"Coluna alvo '{self._coluna_alvo}' ausente no DataFrame.")
        if not modelos_tuning:
            raise ErroValidadorCruzado("Lista de modelos do tuning está vazia.")

        X = dados_treino.drop(columns=[coluna_alvo for coluna_alvo in [self._coluna_alvo]])
        y = cast(pd.Series, dados_treino[self._coluna_alvo])

        # 1. Configuração do esquema de folds pareados
        cfg_cv = self._config_av.validacao.cv_externa
        if cfg_cv.metodo == "repeated_kfold":
            cv = RepeatedKFold(
                n_splits=cfg_cv.splits,
                n_repeats=cfg_cv.repeticoes,
                random_state=42,
            )
        else:
            cv = KFold(
                n_splits=cfg_cv.splits,
                shuffle=True,
                random_state=42,
            )

        splits_brutos = cast(list[tuple[object, object]], list(cv.split(X, y)))  # pyright: ignore[reportUnknownArgumentType]
        splits_indices: list[
            tuple[
                np.ndarray[tuple[int, ...], np.dtype[np.int64]],
                np.ndarray[tuple[int, ...], np.dtype[np.int64]],
            ]
        ] = [
            (
                cast(np.ndarray[tuple[int, ...], np.dtype[np.int64]], item_tr),
                cast(np.ndarray[tuple[int, ...], np.dtype[np.int64]], item_va),
            )
            for item_tr, item_va in splits_brutos
        ]

        # 2. Construção dos estimadores parametrizados
        estimadores_base: dict[str, object] = {}
        for mod_res in modelos_tuning:
            pipeline_mod = self._construir_pipeline_modelo(mod_res.nome_modelo, mod_res.melhores_parametros, X)
            estimadores_base[mod_res.nome_modelo] = pipeline_mod

        # 3. Execução pareada dos folds com rastreamento no MLflow
        id_run_pai: str | None = None
        if self._rastreador_mlflow is not None:
            try:
                id_run_pai = self._rastreador_mlflow.iniciar_run_pai_cv()
            except Exception:
                id_run_pai = None

        tecnica_prep = (
            self._config_mod.preprocessamento.metodo_transformacao_alvo
            if self._config_mod.preprocessamento.transformar_alvo
            else "padrao"
        )
        params_por_modelo: dict[str, dict[str, object]] = {
            m.nome_modelo: m.melhores_parametros for m in modelos_tuning
        }

        matriz_resultados: dict[str, list[float]] = {m: [] for m in estimadores_base}
        estimadores_finais_treinados: dict[str, object] = {}

        for nome_modelo, estimador in estimadores_base.items():
            scores_modelo: list[float] = []

            for fold_idx, (idx_treino, idx_val) in enumerate(splits_indices, start=1):
                X_tr = cast(pd.DataFrame, X.iloc[idx_treino])
                y_tr = cast(pd.Series, y.iloc[idx_treino])
                X_va = cast(pd.DataFrame, X.iloc[idx_val])
                y_va = cast(pd.Series, y.iloc[idx_val])

                # Clona o pipeline para garantir independência total de estado
                est_clone = cast(EstimadorComPredict, clone(estimador))
                est_clone.fit(X_tr, y_tr)

                preds_val = np.asarray(est_clone.predict(X_va), dtype=np.float64)
                y_val_arr = np.asarray(y_va.to_numpy(), dtype=np.float64)
                rmse_split = float(np.sqrt(np.mean((y_val_arr - preds_val) ** 2)))
                mae_split = float(np.mean(np.abs(y_val_arr - preds_val)))
                scores_modelo.append(rmse_split)

                # Salva resultado por fold no MLflow (filho aninhado)
                if self._rastreador_mlflow is not None and id_run_pai is not None:
                    try:
                        self._rastreador_mlflow.registrar_run_filho_fold(
                            nome_modelo=nome_modelo,
                            tecnica_preprocessamento=tecnica_prep,
                            fold=fold_idx,
                            metricas={"rmse": rmse_split, "mae": mae_split},
                            parametros=params_por_modelo.get(nome_modelo, {}),
                            id_run_pai=id_run_pai,
                        )
                    except Exception:
                        pass

            matriz_resultados[nome_modelo] = scores_modelo

            # Ajusta modelo final com todo o conjunto de treino
            est_completo = cast(EstimadorComPredict, clone(estimador))
            est_completo.fit(X, y)
            estimadores_finais_treinados[nome_modelo] = est_completo

        # Finaliza o run pai de validação cruzada no MLflow
        if self._rastreador_mlflow is not None and id_run_pai is not None:
            try:
                self._rastreador_mlflow.finalizar_run_pai_cv(matriz_resultados, id_run_pai)
            except Exception:
                pass

        return matriz_resultados, estimadores_finais_treinados

    def _construir_pipeline_modelo(
        self, nome_modelo: str, melhores_params: dict[str, object], X_exemplo: pd.DataFrame
    ) -> object:
        """Constrói pipeline completo com pré-processamento e hiperparâmetros configurados."""
        cols_numericas: list[str] = []
        cols_categoricas: list[str] = []
        for col in cast(list[object], list(X_exemplo.columns)):
            c = str(col)
            if pd.api.types.is_numeric_dtype(X_exemplo[c]):
                cols_numericas.append(c)
            else:
                cols_categoricas.append(c)

        preprocessor = self._treinador_aux._construir_preprocessador(  # pyright: ignore[reportPrivateUsage]
            cols_numericas, cols_categoricas, self._config_mod
        )
        estimador_base = self._treinador_aux._instanciar_estimador(nome_modelo)  # pyright: ignore[reportPrivateUsage]

        # Aplica os melhores parâmetros no estimador base
        if melhores_params:
            fn_set = getattr(estimador_base, "set_params", None)
            if callable(fn_set):
                try:
                    fn_set(**melhores_params)
                except Exception:
                    pass

        from sklearn.pipeline import Pipeline
        from sklearn.compose import TransformedTargetRegressor

        pipeline_estimador = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", estimador_base),
            ]
        )

        if self._config_mod.preprocessamento.transformar_alvo:
            return TransformedTargetRegressor(
                regressor=pipeline_estimador,
                func=np.log1p,
                inverse_func=np.expm1,
            )
        return pipeline_estimador
