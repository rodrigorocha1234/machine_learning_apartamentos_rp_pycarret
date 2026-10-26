"""Contrato para adaptadores de ingestão de dados tabulares."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class FonteDados(Protocol):
    """Protocolo GoF (Strategy/Adapter) para fontes primárias de dados."""

    def carregar_dados(self) -> pd.DataFrame:
        """Carrega os dados tabulares da fonte configurada e retorna um DataFrame."""
        ...

    def obter_origem(self) -> str:
        """Retorna uma descrição amigável ou caminho da fonte de dados."""
        ...
