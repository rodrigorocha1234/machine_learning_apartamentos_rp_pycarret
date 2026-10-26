"""Repositório GoF concreto para persistência intermediária em banco de dados SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import final, override
import pandas as pd

from app.contratos_dados.repositorio_dados import RepositorioDados


class ErroRepositorioSqlite(Exception):
    """Exceção levantada quando ocorre falha de persistência ou consulta no SQLite."""


@final
class RepositorioSqlite(RepositorioDados):
    """Implementação concreta de RepositorioDados para staging em SQLite."""

    def __init__(self, caminho: str | Path, tabela: str = "imoveis_preparados") -> None:
        caminho_str = str(caminho).strip()
        if caminho_str.startswith("sqlite:///"):
            caminho_str = caminho_str.removeprefix("sqlite:///")

        self._caminho = Path(caminho_str)
        self._tabela = tabela.strip()
        if not self._tabela:
            raise ErroRepositorioSqlite("Nome da tabela de staging não pode ser vazio.")

    @property
    def caminho(self) -> Path:
        """Retorna o caminho do arquivo de banco SQLite."""
        return self._caminho

    def _obter_conexao(self) -> sqlite3.Connection:
        """Cria e retorna uma conexão SQLite garantindo que o diretório pai exista."""
        self._caminho.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._caminho)

    @override
    def gravar_dados(self, dados: pd.DataFrame, substituir: bool = True) -> int:
        if dados.empty:
            raise ErroRepositorioSqlite(
                "Não é permitido gravar um DataFrame vazio no repositório de staging."
            )

        modo = "replace" if substituir else "append"
        try:
            with self._obter_conexao() as conexao:
                dados.to_sql(
                    name=self._tabela,
                    con=conexao,
                    if_exists=modo,
                    index=False,
                )
        except Exception as erro_gravacao:
            raise ErroRepositorioSqlite(
                f"Erro ao gravar dados na tabela '{self._tabela}' do banco '{self._caminho}': {erro_gravacao}"
            ) from erro_gravacao

        return len(dados)

    @override
    def ler_dados(self) -> pd.DataFrame:
        if not self.verificar_existencia():
            raise ErroRepositorioSqlite(
                f"Tabela de staging '{self._tabela}' não existe no banco '{self._caminho}'."
            )

        try:
            with self._obter_conexao() as conexao:
                dados = pd.read_sql_query(f"SELECT * FROM {self._tabela}", con=conexao)
        except Exception as erro_leitura:
            raise ErroRepositorioSqlite(
                f"Erro ao consultar tabela '{self._tabela}' no banco '{self._caminho}': {erro_leitura}"
            ) from erro_leitura

        return dados

    @override
    def verificar_existencia(self) -> bool:
        if not self._caminho.is_file():
            return False

        try:
            with self._obter_conexao() as conexao:
                cursor = conexao.cursor()
                cursor.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (self._tabela,),
                )
                resultado = cursor.fetchone()
                return resultado is not None
        except sqlite3.Error:
            return False
