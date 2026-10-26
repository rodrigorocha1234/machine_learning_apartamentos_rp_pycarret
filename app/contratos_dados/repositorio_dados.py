"""Contrato para repositórios de persistência intermediária (staging)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class RepositorioDados(Protocol):
    """Protocolo GoF (Repository) para armazenamento intermediário de dados."""

    def gravar_dados(self, dados: pd.DataFrame, substituir: bool = True) -> int:
        """Persiste o DataFrame no repositório e retorna o total de registros gravados."""
        ...

    def ler_dados(self) -> pd.DataFrame:
        """Lê os dados persistidos da tabela de staging e retorna um DataFrame."""
        ...

    def verificar_existencia(self) -> bool:
        """Verifica se o banco e a tabela de staging já foram materializados."""
        ...
