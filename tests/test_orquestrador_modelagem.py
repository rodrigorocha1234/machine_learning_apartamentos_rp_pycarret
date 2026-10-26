"""Testes automatizados para OrquestradorModelagem."""

from __future__ import annotations

import tempfile
import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.contratos_modelagem.motor_treinador import ResultadoModelagem, ResultadoTuning
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.treinador_sklearn import TreinadorSklearn
from app.servicos_modelagem.orquestrador_modelagem import (
    ErroOrquestradorModelagem,
    OrquestradorModelagem,
)
from app.servicos_modelagem.rastreador_mlflow import RastreadorMlflow


@pytest.fixture
def base_imoveis_teste() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area_Privativa": [45.0, 60.0, 80.0, 110.0, 50.0, 75.0, 95.0, 130.0, 65.0, 85.0],
            "Quartos": [1, 2, 2, 3, 1, 2, 3, 4, 2, 3],
            "Bairro": [
                "Centro",
                "Sul",
                "Centro",
                "Sul",
                "Norte",
                "Leste",
                "Sul",
                "Sul",
                "Centro",
                "Norte",
            ],
            "Valor_da_Venda": [
                180000.0,
                260000.0,
                380000.0,
                550000.0,
                210000.0,
                330000.0,
                480000.0,
                720000.0,
                290000.0,
                410000.0,
            ],
        }
    )


def test_orquestrador_validacoes_base_vazia() -> None:
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()
    treinador = TreinadorSklearn()
    rastreador = RastreadorMlflow(config_modelagem.mlflow)
    orquestrador = OrquestradorModelagem(
        treinador=treinador,
        rastreador=rastreador,
        config_modelagem=config_modelagem,
        coluna_alvo="Valor_da_Venda",
    )

    with pytest.raises(ErroOrquestradorModelagem, match="Dados de entrada para modelagem estão vazios"):
        orquestrador.executar_modelagem(pd.DataFrame())


def test_orquestrador_validacoes_alvo_ausente(base_imoveis_teste: pd.DataFrame) -> None:
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()
    treinador = TreinadorSklearn()
    rastreador = RastreadorMlflow(config_modelagem.mlflow)
    orquestrador = OrquestradorModelagem(
        treinador=treinador,
        rastreador=rastreador,
        config_modelagem=config_modelagem,
        coluna_alvo="Preco_Inexistente",
    )

    with pytest.raises(ErroOrquestradorModelagem, match="Coluna alvo 'Preco_Inexistente' ausente"):
        orquestrador.executar_modelagem(base_imoveis_teste)


def test_orquestrador_fluxo_completo_anti_leakage(
    base_imoveis_teste: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    for modelo in config_dict["modelos"]:
        if modelo["nome"] in ("ridge", "lasso"):
            modelo["ativo"] = True
        else:
            modelo["ativo"] = False
    config_dict["tuning"]["iteracoes"] = 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dict["mlflow"]["tracking_uri"] = f"file://{tmp_dir}/mlruns"
        config_dict["mlflow"]["experimento"] = "teste_orquestrador"
        config_rapida = ConfiguracaoModelagem.model_validate(config_dict)

        treinador = TreinadorSklearn()
        rastreador = RastreadorMlflow(config_rapida.mlflow)

        orquestrador = OrquestradorModelagem(
            treinador=treinador,
            rastreador=rastreador,
            config_modelagem=config_rapida,
            coluna_alvo="Valor_da_Venda",
            fracao_holdout=0.2,
        )

        res_tuning, res_modelagem = orquestrador.executar_modelagem(base_imoveis_teste)

        assert isinstance(res_tuning, ResultadoTuning)
        assert isinstance(res_modelagem, ResultadoModelagem)
        assert res_modelagem.motor_utilizado == "sklearn"
        assert res_modelagem.total_modelos == 2
        assert res_modelagem.modelo_campeao in ("ridge", "lasso")
        assert len(res_modelagem.runs_filhos) == 2
        assert res_modelagem.duracao_total_segundos > 0
