"""Testes automatizados para TreinadorSklearn com pipelines nativos."""

from __future__ import annotations

import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.contratos_modelagem.motor_treinador import ResultadoTuning
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.treinador_sklearn import ErroTreinadorSklearn, TreinadorSklearn


@pytest.fixture
def dados_sinteticos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area_Privativa": [50.0, 75.0, 100.0, 120.0, 45.0, 80.0, 110.0, 95.0, None, 60.0],
            "Quartos": [1, 2, 3, 3, 1, 2, 3, 2, 2, 1],
            "Bairro": [
                "Centro",
                "Sul",
                "Sul",
                "Norte",
                "Centro",
                "Leste",
                "Sul",
                "Norte",
                "Centro",
                None,
            ],
            "Valor_da_Venda": [
                200000.0,
                350000.0,
                500000.0,
                550000.0,
                180000.0,
                380000.0,
                520000.0,
                460000.0,
                290000.0,
                230000.0,
            ],
        }
    )


def test_sklearn_validacao_dataframe_vazio(dados_sinteticos: pd.DataFrame) -> None:
    _ = dados_sinteticos
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()

    with pytest.raises(ErroTreinadorSklearn, match="está vazio"):
        treinador.treinar_e_ajustar(pd.DataFrame(), "Valor_da_Venda", config_modelagem)


def test_sklearn_validacao_coluna_alvo_ausente(dados_sinteticos: pd.DataFrame) -> None:
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()

    with pytest.raises(ErroTreinadorSklearn, match="Coluna alvo 'Alvo_Inexistente' ausente"):
        treinador.treinar_e_ajustar(dados_sinteticos, "Alvo_Inexistente", config_modelagem)


def test_sklearn_treinamento_e_tuning_rapido(dados_sinteticos: pd.DataFrame) -> None:
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    # Ativa apenas ridge e random_forest para teste rápido
    for modelo in config_dict["modelos"]:
        if modelo["nome"] in ("ridge", "random_forest"):
            modelo["ativo"] = True
        else:
            modelo["ativo"] = False
    config_dict["tuning"]["iteracoes"] = 2
    config_rapida = ConfiguracaoModelagem.model_validate(config_dict)

    resultado = treinador.treinar_e_ajustar(
        dados_treino=dados_sinteticos,
        coluna_alvo="Valor_da_Venda",
        config=config_rapida,
        folds=2,
    )

    assert isinstance(resultado, ResultadoTuning)
    assert len(resultado.modelos_treinados) == 2
    assert resultado.modelo_campeao in ("ridge", "random_forest")
    assert resultado.melhor_rmse > 0.0

    for modelo_res in resultado.modelos_treinados:
        assert modelo_res.framework == "sklearn"
        assert "RMSE" in modelo_res.metricas
        assert "MAE" in modelo_res.metricas
        assert modelo_res.metricas["RMSE"] > 0


def test_sklearn_treinamento_com_grid_e_metrica_customizada(
    dados_sinteticos: pd.DataFrame,
) -> None:
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    for modelo in config_dict["modelos"]:
        modelo["ativo"] = modelo["nome"] == "ridge"

    config_dict["tuning"]["metodo"] = "grid"
    config_dict["tuning"]["otimizar_metrica"] = "MAE"
    config_dict["tuning"]["manter_melhor"] = True
    config_dict["tuning"]["parametros"] = {"ridge": {"alpha": [0.1, 1.0]}}
    config_rapida = ConfiguracaoModelagem.model_validate(config_dict)

    resultado = treinador.treinar_e_ajustar(
        dados_treino=dados_sinteticos,
        coluna_alvo="Valor_da_Venda",
        config=config_rapida,
        folds=2,
    )

    assert isinstance(resultado, ResultadoTuning)
    assert len(resultado.modelos_treinados) == 1
    assert resultado.modelo_campeao == "ridge"
    assert "MAE" in resultado.modelos_treinados[0].metricas

