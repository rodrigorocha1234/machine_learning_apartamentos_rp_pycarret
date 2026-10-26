"""Fábrica GoF (Factory Method) para instanciação desacoplada de motores tabulares."""

from __future__ import annotations

from typing import final
from app.contratos_dados.motor_tabela import MotorTabela
from app.esquema_configuracao import ConfiguracaoTabular
from app.motores_tabela.motor_pandas import MotorPandas
from app.motores_tabela.motor_polars import MotorPolars


class ErroFabricaMotores(Exception):
    """Exceção levantada quando um motor tabular não suportado é requisitado."""


@final
class FabricaMotores:
    """Fábrica para instanciação de estratégias de processamento tabular."""

    @staticmethod
    def criar_motor(configuracao: ConfiguracaoTabular) -> MotorTabela:
        """Cria e retorna uma instância concreta de MotorTabela com base na configuração."""
        motor_normalizado = configuracao.motor.lower().strip()

        if motor_normalizado == "pandas":
            return MotorPandas()
        if motor_normalizado == "polars":
            return MotorPolars()

        raise ErroFabricaMotores(
            f"Motor tabular não suportado pela fábrica: '{configuracao.motor}'. "
            "Suportados atualmente: 'pandas', 'polars'."
        )
