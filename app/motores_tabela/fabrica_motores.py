from __future__ import annotations
from typing import final
from app.contratos_dados.motor_tabela import MotorTabela
from app.motores_tabela.motor_pandas import MotorPandas
from app.motores_tabela.motor_polars import MotorPolars

class ErroFabricaMotores(Exception):
    """Exceção para erros de criação de motores tabulares."""
    pass

@final
class FabricaMotores:
    """Fábrica (GoF Factory) para instanciação desacoplada de motores tabulares."""

    _MOTORES: dict[str, type[MotorTabela]] = {
        "pandas": MotorPandas,
        "polars": MotorPolars,
    }

    @classmethod
    def criar_motor(cls, identificador: str) -> MotorTabela:
        chave = identificador.strip().lower()
        classe_motor = cls._MOTORES.get(chave)
        if classe_motor is None:
            motores_suportados = list(cls._MOTORES.keys())
            raise ErroFabricaMotores(
                f"Motor tabular '{identificador}' não suportado. Opções válidas: {motores_suportados}"
            )
        return classe_motor()
