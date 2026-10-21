from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd
import polars as pl

class ErroFonteDados(Exception):
    """Exceção base de domínio para operações e falhas em fontes de dados."""
    pass

@runtime_checkable
class FonteDados(Protocol):
    """Contrato extensível para ingestão de dados tabulares brutos (Excel, Banco SQL, API REST, etc.)."""

    def carregar_dados(self) -> pd.DataFrame | pl.DataFrame:
        """Carrega e retorna o conjunto de dados brutos como DataFrame estruturado (Pandas ou Polars)."""
        ...

    def validar_conexao(self) -> bool:
        """Verifica se a fonte de dados está acessível, válida e operacional."""
        ...

