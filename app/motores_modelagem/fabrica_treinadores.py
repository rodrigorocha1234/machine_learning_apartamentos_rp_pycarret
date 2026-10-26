"""Fábrica GoF Factory Method para motores de modelagem e tuning."""

from __future__ import annotations

from typing import final
from app.contratos_modelagem.motor_treinador import MotorTreinador
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.treinador_pycaret import TreinadorPycaret
from app.motores_modelagem.treinador_sklearn import TreinadorSklearn


class ErroFabricaTreinadores(Exception):
    """Exceção levantada quando o motor de treinamento solicitado não é suportado."""


@final
class FabricaTreinadores:
    """Fábrica responsável por instanciar o motor de modelagem correto (sklearn ou pycaret)."""

    @classmethod
    def criar_treinador(cls, config: ConfiguracaoModelagem) -> MotorTreinador:
        """Instancia e retorna o motor de treinamento configurado no YAML."""
        motor = config.motor_treino.strip().lower()

        if motor == "pycaret":
            return TreinadorPycaret()

        if motor == "sklearn":
            return TreinadorSklearn()

        raise ErroFabricaTreinadores(
            f"Motor de treino não suportado: '{config.motor_treino}'. Opções válidas: pycaret, sklearn."
        )
