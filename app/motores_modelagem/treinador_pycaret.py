"""Implementação do motor de modelagem e tuning baseado no PyCaret nativo."""

from __future__ import annotations

import time
from typing import cast, final, override
import pandas as pd
from pycaret.regression import create_model, pull, setup, tune_model  # pyright: ignore[reportUnknownVariableType]
from sklearn.base import BaseEstimator

from app.contratos_modelagem.motor_treinador import (
    MotorTreinador,
    ResultadoTreinoModelo,
    ResultadoTuning,
)
from app.esquema_configuracao import ConfiguracaoModelagem, ValorParametro
from app.motores_modelagem.fabrica_folds import FabricaFolds, GeradorFolds
from app.servicos_observabilidade.publicador_modelos_grafana import (
    PublicadorModelosGrafana,
    formatar_nome_modelo,
)


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

    @staticmethod
    def _mapear_preprocessador_pycaret(tipo_preprocessador: str) -> tuple[bool, str]:
        """Mapeia o tipo de pré-processador do YAML para (normalize, normalize_method) do PyCaret."""
        tipo = tipo_preprocessador.lower().strip()
        if tipo in {"standard", "standard_scaler", "padrao"}:
            return True, "zscore"
        if tipo in {"robust", "robust_scaler"}:
            return True, "robust"
        if tipo in {"minmax", "min_max", "minmax_scaler"}:
            return True, "minmax"
        if tipo in {"maxabs", "max_abs", "maxabs_scaler"}:
            return True, "maxabs"
        if tipo in {"nenhum", "none", "passthrough", "sem_escala", "sem_scaler"}:
            return False, "zscore"
        return True, "zscore"

    def _configurar_ambiente_pycaret(
        self,
        dados_treino: pd.DataFrame,
        coluna_alvo: str,
        config: ConfiguracaoModelagem,
        divisor_folds: GeradorFolds,
        tipo_preprocessador: str,
    ) -> None:
        """Inicializa ou reconfigura o pycaret.regression.setup com o pré-processador configurado."""
        est_num = config.imputacao.estrategia_numerica
        est_cat = config.imputacao.estrategia_categorica
        pycaret_num_imp = "median" if est_num == "mediana" else ("mean" if est_num == "media" else "zero")
        pycaret_cat_imp = "mode" if est_cat == "mais_frequente" else "missing"

        normalize, normalize_method = self._mapear_preprocessador_pycaret(tipo_preprocessador)

        try:
            setup(
                data=dados_treino,
                target=coluna_alvo,
                numeric_imputation=pycaret_num_imp,
                categorical_imputation=pycaret_cat_imp,
                transform_target=config.preprocessamento.transformar_alvo,
                normalize=normalize,
                normalize_method=normalize_method,
                fold_strategy=divisor_folds,
                session_id=42,
                verbose=False,
                html=False,
            )
        except Exception as erro_setup:
            raise ErroTreinadorPycaret(
                f"Falha ao inicializar pycaret.regression.setup com preprocessador '{tipo_preprocessador}': {erro_setup}"
            ) from erro_setup

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

        divisor_folds = self._fabrica_folds.criar_divisor(
            config.validacao_cruzada, folds_fallback=folds
        )

        modelos_treinados: list[ResultadoTreinoModelo] = []
        ultimo_prep: str | None = None

        for item_modelo in config.obter_modelos_ativos():
            pycaret_id = item_modelo.pycaret_id
            if not pycaret_id:
                continue

            tipo_prep = item_modelo.obter_preprocessador()
            if tipo_prep != ultimo_prep:
                self._configurar_ambiente_pycaret(
                    dados_treino=dados_treino,
                    coluna_alvo=coluna_alvo,
                    config=config,
                    divisor_folds=divisor_folds,
                    tipo_preprocessador=tipo_prep,
                )
                ultimo_prep = tipo_prep

            inicio_modelo = time.perf_counter()
            params_iniciais = dict(item_modelo.parametros) if item_modelo.parametros else {}

            try:
                # 2. Criação do modelo nativo no PyCaret
                if params_iniciais:
                    modelo_base: BaseEstimator = create_model(pycaret_id, **params_iniciais, verbose=False)
                else:
                    modelo_base = create_model(pycaret_id, verbose=False)

                # 3. Tuning nativo com busca configurada no YAML
                grade = config.tuning.obter_grade_modelo(item_modelo.nome)
                grade_pycaret: dict[str, list[ValorParametro]] = {
                    k: v for k, v in grade.items()
                }

                nome_formatado = formatar_nome_modelo(item_modelo.nome)
                if config.tuning.ativo and grade_pycaret:
                    msg_tuning = f"realizando tuning de parâmetros {nome_formatado}"
                    print(f"  → Realizando tuning de parâmetros {nome_formatado}...")
                    PublicadorModelosGrafana.registrar_log(item_modelo.nome, msg_tuning)
                    tuning_ativo = True
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

                    modelo_ajustado: BaseEstimator = tune_model(
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
                    msg_tuning = f"treinando modelo {nome_formatado} (tuning desativado)"
                    print(f"  → Treinando modelo {nome_formatado} (tuning desativado)...")
                    PublicadorModelosGrafana.registrar_log(item_modelo.nome, msg_tuning)
                    tuning_ativo = False
                    modelo_ajustado = modelo_base

                duracao = time.perf_counter() - inicio_modelo

                # 4. Extração de métricas de CV via pull() nativo do PyCaret
                df_metricas: pd.DataFrame = pull()  # pyright: ignore[reportUnknownVariableType]
                rmse_val = 0.0
                mae_val = 0.0
                r2_val = 0.0

                if "Mean" in df_metricas.index:
                    linha_media = cast(pd.Series, df_metricas.loc["Mean"])
                    linha_dict = dict(linha_media.to_dict())
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
                params_dict: dict[str, ValorParametro] = {}
                if hasattr(modelo_ajustado, "get_params"):
                    get_params_fn = getattr(modelo_ajustado, "get_params")
                    if callable(get_params_fn):
                        todos_params = cast(dict[str, ValorParametro], get_params_fn())
                        # Mantém apenas os parâmetros que estavam na grade
                        for k in grade_pycaret:
                            if k in todos_params:
                                params_dict[k] = todos_params[k]

                print(f"  ✓ {nome_formatado.capitalize()} concluído em {duracao:.2f}s | RMSE: R$ {rmse_val:,.2f}")

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
                    tuning_ativo=tuning_ativo,
                    mensagem_tuning=msg_tuning,
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
