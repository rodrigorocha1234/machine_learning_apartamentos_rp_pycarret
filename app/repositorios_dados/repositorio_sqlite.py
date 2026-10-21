from dataclasses import dataclass
import sqlite3
from typing import final, override
import pandas as pd
from app.contratos_dados.repositorio_dados import RepositorioDados

class ErroRepositorio(Exception):
    """Exceção para erros de acesso ou persistência no repositório de dados."""
    pass

@final
@dataclass(frozen=True)
class RepositorioSqlite(RepositorioDados):
    caminho: str
    tabela: str

    @override
    def gravar_dados(self, dados: pd.DataFrame) -> None:
        try:
            with sqlite3.connect(self.caminho) as conexao:
                dados.to_sql(self.tabela, conexao, if_exists="replace", index=False)
        except Exception as erro:
            raise ErroRepositorio(f"Erro ao gravar dados no SQLite em '{self.caminho}': {erro}") from erro

    @override
    def ler_dados(self) -> pd.DataFrame:
        if not self.tabela.isidentifier():
            raise ErroRepositorio(f"Nome de tabela inválido: '{self.tabela}'")
        try:
            with sqlite3.connect(self.caminho) as conexao:
                return pd.read_sql_query(f'SELECT * FROM "{self.tabela}"', conexao)
        except Exception as erro:
            raise ErroRepositorio(f"Erro ao ler tabela '{self.tabela}' no SQLite: {erro}") from erro

