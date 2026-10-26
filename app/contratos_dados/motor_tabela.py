"""Contrato para motores de processamento e preparação tabular."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class MotorTabela(Protocol):
    """Protocolo GoF (Strategy) para motores de manipulação de dados tabulares."""

    def filtrar_dados(self, dados: pd.DataFrame, expressao: str) -> pd.DataFrame:
        """Filtra linhas com base em expressão booleana avaliada com segurança."""
        ...

    def transformar_dados(self, dados: pd.DataFrame, expressao: str, destino: str) -> pd.DataFrame:
        """Aplica transformação ou cria nova coluna a partir de expressão avaliada com segurança."""
        ...

    def selecionar_colunas(self, dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
        """Filtra e ordena o conjunto de colunas ativas."""
        ...

    def obter_nome_motor(self) -> str:
        """Retorna o identificador único do motor (ex: 'pandas', 'polars')."""
        ...
