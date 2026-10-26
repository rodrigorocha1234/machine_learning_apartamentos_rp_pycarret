"""Testes unitários para TreinadorPycaret."""

from __future__ import annotations

import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.motores_modelagem.treinador_pycaret import ErroTreinadorPycaret, TreinadorPycaret


def test_pycaret_nome_motor() -> None:
    treinador = TreinadorPycaret()
    assert treinador.obter_nome_motor() == "pycaret"


def test_pycaret_validacao_dataframe_vazio() -> None:
    treinador = TreinadorPycaret()
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()

    with pytest.raises(ErroTreinadorPycaret, match="DataFrame de entrada para PyCaret está vazio"):
        treinador.treinar_e_ajustar(pd.DataFrame(), "Valor_da_Venda", config_modelagem)


def test_pycaret_validacao_coluna_alvo_ausente() -> None:
    treinador = TreinadorPycaret()
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()
    df = pd.DataFrame({"Area": [50.0, 70.0], "Valor": [200000.0, 300000.0]})

    with pytest.raises(ErroTreinadorPycaret, match="Coluna alvo 'Alvo_Invalido' ausente"):
        treinador.treinar_e_ajustar(df, "Alvo_Invalido", config_modelagem)
