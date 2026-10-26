"""Testes automatizados para FabricaTreinadores (GoF Factory Method)."""

from __future__ import annotations

import pytest

from app.conf_yaml import ConfYAML
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.fabrica_treinadores import ErroFabricaTreinadores, FabricaTreinadores
from app.motores_modelagem.treinador_pycaret import TreinadorPycaret
from app.motores_modelagem.treinador_sklearn import TreinadorSklearn


def test_fabrica_cria_treinador_sklearn() -> None:
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    config_dict["motor_treino"] = "sklearn"
    config_sklearn = ConfiguracaoModelagem.model_validate(config_dict)

    treinador = FabricaTreinadores.criar_treinador(config_sklearn)
    assert isinstance(treinador, TreinadorSklearn)
    assert treinador.obter_nome_motor() == "sklearn"


def test_fabrica_cria_treinador_pycaret() -> None:
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    config_dict["motor_treino"] = "pycaret"
    config_pycaret = ConfiguracaoModelagem.model_validate(config_dict)

    treinador = FabricaTreinadores.criar_treinador(config_pycaret)
    assert isinstance(treinador, TreinadorPycaret)
    assert treinador.obter_nome_motor() == "pycaret"


def test_fabrica_lanca_erro_para_motor_desconhecido() -> None:
    conf = ConfYAML()
    config_valida = conf.obter_modelagem()
    # Simula contorno de validação para testar salvaguarda interna da fábrica
    object.__setattr__(config_valida, "motor_treino", "desconhecido")

    with pytest.raises(ErroFabricaTreinadores, match="Motor de treino não suportado"):
        FabricaTreinadores.criar_treinador(config_valida)
