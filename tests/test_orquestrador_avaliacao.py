"""Testes automatizados para OrquestradorAvaliacao."""

from __future__ import annotations

import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo, ResultadoTuning
from app.esquema_configuracao import ConfiguracaoAvaliacao
from app.servicos_avaliacao.orquestrador_avaliacao import (
    ErroOrquestradorAvaliacao,
    OrquestradorAvaliacao,
    ResultadoEtapaAvaliacao,
)


@pytest.fixture
def df_completo_teste() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area_Privativa": [45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0, 115.0, 125.0, 135.0],
            "Quartos": [1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
            "Bairro": [
                "Centro",
                "Centro",
                "Sul",
                "Sul",
                "Norte",
                "Norte",
                "Sul",
                "Leste",
                "Sul",
                "Sul",
            ],
            "Valor_da_Venda": [
                160000.0,
                210000.0,
                270000.0,
                330000.0,
                390000.0,
                460000.0,
                530000.0,
                610000.0,
                690000.0,
                780000.0,
            ],
        }
    )


def test_orquestrador_avaliacao_validacao_base_vazia() -> None:
    conf = ConfYAML()
    orquestrador = OrquestradorAvaliacao(
        config_avaliacao=conf.obter_avaliacao(),
        config_modelagem=conf.obter_modelagem(),
    )

    with pytest.raises(ErroOrquestradorAvaliacao, match="estão vazios"):
        orquestrador.executar_avaliacao_e_selecao(pd.DataFrame())


def test_orquestrador_avaliacao_fluxo_completo(df_completo_teste: pd.DataFrame) -> None:
    conf = ConfYAML()
    cfg_av_dict = conf.obter_avaliacao().model_dump()
    cfg_av_dict["validacao"]["cv_externa"]["splits"] = 2
    cfg_av_dict["validacao"]["cv_externa"]["repeticoes"] = 2
    cfg_av_dict["validacao"]["holdout"] = 0.2
    cfg_av_dict["votacao"]["ativo"] = True
    cfg_av_dict["votacao"]["tecnica"] = "media_simples"
    config_av = ConfiguracaoAvaliacao.model_validate(cfg_av_dict)

    config_mod = conf.obter_modelagem()
    orquestrador = OrquestradorAvaliacao(
        config_avaliacao=config_av,
        config_modelagem=config_mod,
        coluna_alvo="Valor_da_Venda",
    )

    tuning_mock = ResultadoTuning(
        metodo_tuning="random",
        iteracoes=2,
        modelos_treinados=[
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
            ResultadoTreinoModelo(
                nome_modelo="random_forest",
                framework="sklearn",
                melhores_parametros={"n_estimators": 50},
                metricas={},
                duracao_segundos=0.2,
            ),
        ],
        modelo_campeao="ridge",
        melhor_rmse=20000.0,
    )

    resultado = orquestrador.executar_avaliacao_e_selecao(
        dados=df_completo_teste,
        resultado_tuning=tuning_mock,
    )

    assert isinstance(resultado, ResultadoEtapaAvaliacao)
    assert resultado.total_amostras_treino == 8
    assert resultado.total_amostras_holdout == 2
    assert "ridge" in resultado.metricas_holdout
    assert "lasso" in resultado.metricas_holdout
    assert resultado.resultado_comite.ativo is True
    assert resultado.modelo_final_recomendado == "ensemble_media_simples"
    assert resultado.duracao_total_segundos > 0.0
