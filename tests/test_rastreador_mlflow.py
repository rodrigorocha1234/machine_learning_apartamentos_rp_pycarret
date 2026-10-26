"""Testes automatizados para RastreadorMlflow com hierarquia pai/filho."""

from __future__ import annotations

import tempfile
import mlflow
import pytest

from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo, ResultadoTuning
from app.esquema_configuracao import ConfiguracaoMlflow
from app.servicos_modelagem.rastreador_mlflow import RastreadorMlflow


def test_rastreador_mlflow_ciclo_completo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"file://{tmp_dir}/mlruns"
        config_mlflow = ConfiguracaoMlflow(
            tracking_uri=tracking_uri,
            experimento="teste_hierarquia_etapa4",
        )
        rastreador = RastreadorMlflow(config_mlflow)

        # 1. Inicia run pai
        id_run_pai = rastreador.iniciar_run_pai(metodo_tuning="random", motor="sklearn")
        assert isinstance(id_run_pai, str)
        assert len(id_run_pai) > 0

        # Verifica dados do run pai
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run_pai = client.get_run(id_run_pai)
        assert run_pai.data.tags.get("mlflow.runName") == "tuning_de_parametros_random"
        assert run_pai.data.tags.get("pipeline.motor") == "sklearn"

        # 2. Registra run filho
        res_modelo = ResultadoTreinoModelo(
            nome_modelo="ridge",
            framework="sklearn",
            melhores_parametros={"alpha": 1.0},
            metricas={"RMSE": 15000.0, "MAE": 11000.0, "R2": 0.85},
            duracao_segundos=0.5,
        )
        id_run_filho = rastreador.registrar_run_filho(
            resultado_modelo=res_modelo,
            tecnica_preprocessamento="padrao",
            id_run_pai=id_run_pai,
        )
        assert isinstance(id_run_filho, str)

        run_filho = client.get_run(id_run_filho)
        assert run_filho.data.tags.get("mlflow.parentRunId") == id_run_pai
        assert run_filho.data.tags.get("mlflow.runName") == "modelo_ridge_padrao"
        assert run_filho.data.metrics.get("RMSE") == 15000.0

        # 3. Finaliza run pai
        res_tuning = ResultadoTuning(
            metodo_tuning="random",
            iteracoes=5,
            modelos_treinados=[res_modelo],
            modelo_campeao="ridge",
            melhor_rmse=15000.0,
        )
        rastreador.finalizar_run_pai(res_tuning, id_run_pai)

        run_pai_final = client.get_run(id_run_pai)
        assert run_pai_final.info.status == "FINISHED"
        assert run_pai_final.data.tags.get("modelo.campeao") == "ridge"
        assert run_pai_final.data.metrics.get("melhor_rmse") == 15000.0


def test_rastreador_mlflow_validacao_cruzada_por_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"file://{tmp_dir}/mlruns"
        config_mlflow = ConfiguracaoMlflow(
            tracking_uri=tracking_uri,
            experimento="teste_cv_mlflow",
        )
        rastreador = RastreadorMlflow(config_mlflow)

        # 1. Inicia run pai validacao_cruzada
        id_run_pai = rastreador.iniciar_run_pai_cv()
        assert isinstance(id_run_pai, str)
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run_pai = client.get_run(id_run_pai)
        assert run_pai.data.tags.get("mlflow.runName") == "validacao_cruzada"
        assert run_pai.data.tags.get("pipeline.etapa") == "validacao_cruzada"

        # 2. Registra run filho de fold
        id_filho = rastreador.registrar_run_filho_fold(
            nome_modelo="ridge",
            tecnica_preprocessamento="log1p",
            fold=1,
            metricas={"rmse": 12000.0, "mae": 9000.0},
            parametros={"alpha": 1.0},
            id_run_pai=id_run_pai,
        )
        assert isinstance(id_filho, str)
        run_filho = client.get_run(id_filho)
        assert (
            run_filho.data.tags.get("mlflow.runName")
            == "metodo_ridge_prepocessamento_log1p_fold_1"
        )
        assert run_filho.data.metrics.get("rmse") == 12000.0

        # 3. Finaliza run pai de CV
        rastreador.finalizar_run_pai_cv(
            matriz_resultados={"ridge": [12000.0, 11500.0]},
            id_run_pai=id_run_pai,
        )
        run_pai_final = client.get_run(id_run_pai)
        assert run_pai_final.info.status == "FINISHED"
        assert run_pai_final.data.metrics.get("media_rmse_ridge") == 11750.0

