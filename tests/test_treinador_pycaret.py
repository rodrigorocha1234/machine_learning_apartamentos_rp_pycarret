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


def test_pycaret_mapeamento_preprocessador() -> None:
    """Verifica que as escolhas de pré-processador do YAML são mapeadas corretamente para PyCaret."""
    assert TreinadorPycaret._mapear_preprocessador_pycaret("standard") == (True, "zscore")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("padrao") == (True, "zscore")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("robust") == (True, "robust")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("minmax") == (True, "minmax")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("maxabs") == (True, "maxabs")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("nenhum") == (False, "zscore")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("passthrough") == (False, "zscore")
    assert TreinadorPycaret._mapear_preprocessador_pycaret("sem_escala") == (False, "zscore")

