"""Implementação do motor de modelagem e tuning baseado no PyCaret nativo."""

from __future__ import annotations

import time
from typing import cast, final, override
import pandas as pd
from pycaret.regression import create_model, pull, setup, tune_model  # pyright: ignore[reportUnknownVariableType]

from app.contratos_modelagem.motor_treinador import (
    MotorTreinador,
    ResultadoTreinoModelo,
    ResultadoTuning,
)
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.fabrica_folds import FabricaFolds


class ErroTreinadorPycaret(Exception):
    """Exceção levantada em falhas durante o treinamento ou tuning com PyCaret."""


@final
class TreinadorPycaret(MotorTreinador):
    """Motor de modelagem e busca de hiperparâmetros nativo via PyCaret Regression."""

    def __init__(self, fabrica_folds: FabricaFolds | None = None) -> None:
        self._fabrica_folds = fabrica_folds or FabricaFolds()

    @override
    def obter_nome_motor(self) -> str:
        return "pycaret"

    @override
    def treinar_e_ajustar(
        self,
        dados_treino: pd.DataFrame,
        coluna_alvo: str,
        config: ConfiguracaoModelagem,
        folds: int = 3,
    ) -> ResultadoTuning:
        if dados_treino.empty:
            raise ErroTreinadorPycaret("DataFrame de entrada para PyCaret está vazio.")
        if coluna_alvo not in dados_treino.columns:
            raise ErroTreinadorPycaret(f"Coluna alvo '{coluna_alvo}' ausente no DataFrame.")

        # 1. Configuração do ambiente PyCaret Setup nativo
        est_num = config.imputacao.estrategia_numerica
        est_cat = config.imputacao.estrategia_categorica
        pycaret_num_imp = "median" if est_num == "mediana" else ("mean" if est_num == "media" else "zero")
        pycaret_cat_imp = "mode" if est_cat == "mais_frequente" else "missing"

        divisor_folds = self._fabrica_folds.criar_divisor(
            config.validacao_cruzada, folds_fallback=folds
        )

        try:
            setup(
                data=dados_treino,
                target=coluna_alvo,
                numeric_imputation=pycaret_num_imp,
                categorical_imputation=pycaret_cat_imp,
                transform_target=config.preprocessamento.transformar_alvo,
                fold_strategy=divisor_folds,
                session_id=42,
                verbose=False,
                html=False,
            )
        except Exception as erro_setup:
            raise ErroTreinadorPycaret(f"Falha ao inicializar pycaret.regression.setup: {erro_setup}") from erro_setup

        modelos_treinados: list[ResultadoTreinoModelo] = []

        for item_modelo in config.modelos:
            if not item_modelo.ativo:
                continue

            inicio_modelo = time.perf_counter()
            pycaret_id = item_modelo.pycaret_id
            if not pycaret_id:
                continue

            try:
                # 2. Criação do modelo nativo no PyCaret
                modelo_base: object = create_model(pycaret_id, verbose=False)

                # 3. Tuning nativo com busca configurada no YAML
                grade = config.tuning.parametros.get(item_modelo.nome, {})
                grade_pycaret: dict[str, list[object]] = {
                    k: cast(list[object], v) for k, v in grade.items()
                }

                if config.tuning.ativo and grade_pycaret:
                    search_lib = "scikit-learn"
                    search_algo = "random"
                    if config.tuning.metodo == "grid":
                        search_algo = "grid"
                    elif config.tuning.metodo == "bayesian":
                        try:
                            import skopt  # type: ignore  # noqa: F401
                            search_lib = "scikit-optimize"
                            search_algo = "bayesian"
                        except ImportError:
                            search_lib = "scikit-learn"
                            search_algo = "random"
                    elif config.tuning.metodo == "optuna":
                        try:
                            import optuna  # type: ignore  # noqa: F401
                            search_lib = "optuna"
                            search_algo = "tpe"
                        except ImportError:
                            search_lib = "scikit-learn"
                            search_algo = "random"

                    modelo_ajustado: object = tune_model(
                        modelo_base,
                        n_iter=config.tuning.iteracoes,
                        optimize=config.tuning.otimizar_metrica,
                        custom_grid=grade_pycaret,
                        search_library=search_lib,
                        search_algorithm=search_algo,
                        choose_better=config.tuning.manter_melhor,
                        early_stopping=config.tuning.early_stopping,
                        verbose=False,
                    )
                else:
                    modelo_ajustado = modelo_base

                duracao = time.perf_counter() - inicio_modelo

                # 4. Extração de métricas de CV via pull() nativo do PyCaret
                df_metricas: pd.DataFrame = pull()  # pyright: ignore[reportUnknownVariableType]
                rmse_val = 0.0
                mae_val = 0.0
                r2_val = 0.0

                if "Mean" in df_metricas.index:
                    linha_media = cast(pd.Series, df_metricas.loc["Mean"])
                    linha_dict = cast(dict[str, object], linha_media.to_dict())
                    if "RMSE" in linha_dict:
                        try:
                            rmse_val = float(str(linha_dict["RMSE"]))
                        except (ValueError, TypeError):
                            pass
                    if "MAE" in linha_dict:
                        try:
                            mae_val = float(str(linha_dict["MAE"]))
                        except (ValueError, TypeError):
                            pass
                    if "R2" in linha_dict:
                        try:
                            r2_val = float(str(linha_dict["R2"]))
                        except (ValueError, TypeError):
                            pass

                # Extrai parâmetros se disponíveis no estimador
                params_dict: dict[str, object] = {}
                if hasattr(modelo_ajustado, "get_params"):
                    get_params_fn = getattr(modelo_ajustado, "get_params")
                    if callable(get_params_fn):
                        todos_params = cast(dict[str, object], get_params_fn())
                        # Mantém apenas os parâmetros que estavam na grade
                        for k in grade_pycaret:
                            if k in todos_params:
                                params_dict[k] = todos_params[k]

                res_modelo = ResultadoTreinoModelo(
                    nome_modelo=item_modelo.nome,
                    framework="pycaret",
                    melhores_parametros=params_dict,
                    metricas={
                        "RMSE": round(rmse_val, 2),
                        "MAE": round(mae_val, 2),
                        "R2": round(r2_val, 4),
                    },
                    duracao_segundos=round(duracao, 4),
                )
                modelos_treinados.append(res_modelo)

            except Exception as erro_modelo:
                raise ErroTreinadorPycaret(
                    f"Falha ao treinar modelo PyCaret '{item_modelo.nome}' ({pycaret_id}): {erro_modelo}"
                ) from erro_modelo

        if not modelos_treinados:
            raise ErroTreinadorPycaret("Nenhum modelo ativo foi treinado com sucesso no PyCaret.")

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
