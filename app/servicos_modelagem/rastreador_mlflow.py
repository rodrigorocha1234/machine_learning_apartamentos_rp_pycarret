"""Serviço de rastreamento hierárquico e governança de experimentos no MLflow."""

from __future__ import annotations

import os
from typing import cast, final
import mlflow
from mlflow.entities import Run

from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo, ResultadoTuning
from app.esquema_configuracao import ConfiguracaoMlflow


class ErroRastreadorMlflow(Exception):
    """Exceção levantada em falhas de comunicação ou registro com o MLflow."""


@final
class RastreadorMlflow:
    """Gerenciador desacoplado da hierarquia de runs (pai/filhos) e artefatos no MLflow."""

    def __init__(self, config_mlflow: ConfiguracaoMlflow) -> None:
        self._config = config_mlflow
        self._configurar_ambiente()

    def _configurar_ambiente(self) -> None:
        """Configura a URI de tracking e silencia mensagens verbosas do cliente."""
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_tracking_uri(self._config.tracking_uri)

    def iniciar_run_pai(self, metodo_tuning: str, motor: str) -> str:
        """Inicia e retorna o run pai hierárquico tuning_de_parametros_{metodo_tuning}."""
        try:
            mlflow.set_experiment(self._config.experimento)
            nome_run_pai = f"tuning_de_parametros_{metodo_tuning}"

            run_pai = mlflow.start_run(run_name=nome_run_pai)
            mlflow.set_tag("pipeline.etapa", "modelagem_regressao")
            mlflow.set_tag("pipeline.motor", motor)
            mlflow.set_tag("pipeline.metodo_tuning", metodo_tuning)

            run_obj = cast(Run, run_pai)
            return str(run_obj.info.run_id)
        except Exception as erro:
            raise ErroRastreadorMlflow(
                f"Falha ao iniciar run pai no MLflow em '{self._config.tracking_uri}': {erro}"
            ) from erro

    def registrar_run_filho(
        self,
        resultado_modelo: ResultadoTreinoModelo,
        tecnica_preprocessamento: str,
        id_run_pai: str,
    ) -> str:
        """Registra um run filho modelo_{nome_modelo}_{tecnica_preprocessamento} aninhado."""
        nome_run_filho = (
            f"modelo_{resultado_modelo.nome_modelo}_{tecnica_preprocessamento}"
        )
        try:
            with mlflow.start_run(run_name=nome_run_filho, nested=True) as run_filho:
                run_obj = cast(Run, run_filho)
                run_id_filho = str(run_obj.info.run_id)

                # Tags de governança
                mlflow.set_tag("parent_run_id", id_run_pai)
                mlflow.set_tag("modelo.nome", resultado_modelo.nome_modelo)
                mlflow.set_tag("modelo.framework", resultado_modelo.framework)
                mlflow.set_tag("modelo.preprocessamento", tecnica_preprocessamento)

                # Parâmetros
                for param_nome, param_valor in resultado_modelo.melhores_parametros.items():
                    mlflow.log_param(param_nome, str(param_valor)[:250])

                # Métricas
                for metrica_nome, metrica_valor in resultado_modelo.metricas.items():
                    mlflow.log_metric(metrica_nome, float(metrica_valor))

                mlflow.log_metric("duracao_segundos", resultado_modelo.duracao_segundos)

                return run_id_filho
        except Exception as erro:
            raise ErroRastreadorMlflow(
                f"Falha ao registrar run filho para '{resultado_modelo.nome_modelo}': {erro}"
            ) from erro

    def finalizar_run_pai(self, resultado_tuning: ResultadoTuning, id_run_pai: str) -> None:
        """Finaliza o run pai registrando métricas agregadas e o modelo campeão."""
        try:
            mlflow.set_tag("modelo.campeao", resultado_tuning.modelo_campeao)
            mlflow.log_metric("melhor_rmse", resultado_tuning.melhor_rmse)
            mlflow.log_metric("total_modelos_treinados", float(len(resultado_tuning.modelos_treinados)))
            mlflow.end_run()
        except Exception as erro:
            raise ErroRastreadorMlflow(
                f"Falha ao finalizar run pai '{id_run_pai}' no MLflow: {erro}"
            ) from erro

    def iniciar_run_pai_cv(self) -> str:
        """Inicia e retorna o run pai hierárquico 'validacao_cruzada'."""
        try:
            mlflow.set_experiment(self._config.experimento)
            run_pai = mlflow.start_run(run_name="validacao_cruzada")
            mlflow.set_tag("pipeline.etapa", "validacao_cruzada")

            run_obj = cast(Run, run_pai)
            return str(run_obj.info.run_id)
        except Exception as erro:
            raise ErroRastreadorMlflow(
                f"Falha ao iniciar run pai de validação cruzada no MLflow: {erro}"
            ) from erro

    def registrar_run_filho_fold(
        self,
        nome_modelo: str,
        tecnica_preprocessamento: str,
        fold: int,
        metricas: dict[str, float],
        parametros: dict[str, object] | None = None,
        id_run_pai: str | None = None,
    ) -> str:
        """Registra um run filho por fold aninhado com o padrão:
        metodo_{nome_do_modelo_regressao}_prepocessamento_{nome_da_tecnica_preprocessamento}_fold_{numero_do_fold}
        """
        nome_run_filho = (
            f"metodo_{nome_modelo}_prepocessamento_{tecnica_preprocessamento}_fold_{fold}"
        )
        try:
            with mlflow.start_run(run_name=nome_run_filho, nested=True) as run_filho:
                run_obj = cast(Run, run_filho)
                run_id_filho = str(run_obj.info.run_id)

                if id_run_pai:
                    mlflow.set_tag("parent_run_id", id_run_pai)
                mlflow.set_tag("modelo.nome", nome_modelo)
                mlflow.set_tag("modelo.preprocessamento", tecnica_preprocessamento)
                mlflow.set_tag("modelo.fold", str(fold))

                mlflow.log_param("fold", fold)
                if parametros:
                    for param_nome, param_valor in parametros.items():
                        mlflow.log_param(param_nome, str(param_valor)[:250])

                for metrica_nome, metrica_valor in metricas.items():
                    mlflow.log_metric(metrica_nome, float(metrica_valor))

                return run_id_filho
        except Exception as erro:
            raise ErroRastreadorMlflow(
                f"Falha ao registrar run filho fold {fold} para '{nome_modelo}': {erro}"
            ) from erro

    def finalizar_run_pai_cv(
        self,
        matriz_resultados: dict[str, list[float]],
        id_run_pai: str,
    ) -> None:
        """Finaliza o run pai de validação cruzada registrando métricas agregadas."""
        try:
            for mod_nome, scores in matriz_resultados.items():
                if scores:
                    media_rmse = float(sum(scores) / len(scores))
                    mlflow.log_metric(f"media_rmse_{mod_nome}", media_rmse)
            mlflow.log_metric("total_modelos_avaliados", float(len(matriz_resultados)))
            mlflow.end_run()
        except Exception as erro:
            raise ErroRastreadorMlflow(
                f"Falha ao finalizar run pai de CV '{id_run_pai}' no MLflow: {erro}"
            ) from erro
