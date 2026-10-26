"""Testes automatizados para ValidadorCruzado."""

from __future__ import annotations

import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo
from app.esquema_configuracao import ConfiguracaoAvaliacao
from app.servicos_avaliacao.validador_cruzado import ErroValidadorCruzado, ValidadorCruzado


@pytest.fixture
def df_treino_sintetico() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area_Privativa": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
            "Quartos": [1, 1, 2, 2, 2, 3, 3, 4],
            "Bairro": ["Centro", "Centro", "Sul", "Sul", "Sul", "Norte", "Norte", "Sul"],
            "Valor_da_Venda": [
                150000.0,
                190000.0,
                230000.0,
                280000.0,
                330000.0,
                390000.0,
                460000.0,
                550000.0,
            ],
        }
    )


def test_validador_cruzado_dataframe_vazio() -> None:
    conf = ConfYAML()
    validador = ValidadorCruzado(conf.obter_avaliacao(), conf.obter_modelagem())

    with pytest.raises(ErroValidadorCruzado, match="está vazio"):
        validador.executar_cv_pareada(pd.DataFrame(), [])


def test_validador_cruzado_execucao_pareada(df_treino_sintetico: pd.DataFrame) -> None:
    conf = ConfYAML()
    cfg_av_dict = conf.obter_avaliacao().model_dump()
    cfg_av_dict["validacao"]["cv_externa"]["splits"] = 2
    cfg_av_dict["validacao"]["cv_externa"]["repeticoes"] = 2
    config_av = ConfiguracaoAvaliacao.model_validate(cfg_av_dict)

    config_mod = conf.obter_modelagem()
    validador = ValidadorCruzado(config_av, config_mod, coluna_alvo="Valor_da_Venda")

    modelos_mock = [
        ResultadoTreinoModelo(
            nome_modelo="ridge",
            framework="sklearn",
            melhores_parametros={"alpha": 1.0},
            metricas={},
            duracao_segundos=0.1,
        ),
        ResultadoTreinoModelo(
            nome_modelo="lasso",
            framework="sklearn",
            melhores_parametros={"alpha": 0.1},
            metricas={},
            duracao_segundos=0.1,
        ),
    ]

    matriz_cv, estimadores = validador.executar_cv_pareada(df_treino_sintetico, modelos_mock)

    assert "ridge" in matriz_cv
    assert "lasso" in matriz_cv
    # 2 splits * 2 repetições = 4 dobras
    assert len(matriz_cv["ridge"]) == 4
    assert len(matriz_cv["lasso"]) == 4
    assert all(rmse > 0 for rmse in matriz_cv["ridge"])
    assert len(estimadores) == 2


def test_validador_cruzado_com_rastreamento_mlflow(
    df_treino_sintetico: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile
    import mlflow
    from app.esquema_configuracao import ConfiguracaoMlflow
    from app.servicos_modelagem.rastreador_mlflow import RastreadorMlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"file://{tmp_dir}/mlruns"
        conf = ConfYAML()
        cfg_av_dict = conf.obter_avaliacao().model_dump()
        cfg_av_dict["validacao"]["cv_externa"]["splits"] = 2
        cfg_av_dict["validacao"]["cv_externa"]["repeticoes"] = 1
        config_av = ConfiguracaoAvaliacao.model_validate(cfg_av_dict)

        config_mod = conf.obter_modelagem()
        config_mlflow = ConfiguracaoMlflow(tracking_uri=tracking_uri, experimento="teste_validador_cv")
        rastreador = RastreadorMlflow(config_mlflow)

        validador = ValidadorCruzado(
            config_av,
            config_mod,
            coluna_alvo="Valor_da_Venda",
            rastreador_mlflow=rastreador,
        )

        modelos_mock = [
            ResultadoTreinoModelo(
                nome_modelo="ridge",
                framework="sklearn",
                melhores_parametros={"alpha": 1.0},
                metricas={},
                duracao_segundos=0.1,
            )
        ]

        matriz_cv, _ = validador.executar_cv_pareada(df_treino_sintetico, modelos_mock)
        assert len(matriz_cv["ridge"]) == 2

        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name("teste_validador_cv")
        assert exp is not None
        runs = client.search_runs(experiment_ids=[exp.experiment_id])

        nomes_runs = [r.data.tags.get("mlflow.runName") for r in runs]
        assert "validacao_cruzada" in nomes_runs
        assert "metodo_ridge_prepocessamento_log1p_fold_1" in nomes_runs
        assert "metodo_ridge_prepocessamento_log1p_fold_2" in nomes_runs

