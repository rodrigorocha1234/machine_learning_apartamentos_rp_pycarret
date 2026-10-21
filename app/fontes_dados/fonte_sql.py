from __future__ import annotations
from dataclasses import dataclass
from typing import final, override
import pandas as pd
from sqlalchemy import create_engine, text
from app.contratos_dados.fonte_dados import ErroFonteDados, FonteDados

class ErroFonteSql(ErroFonteDados):
    """Exceção levantada em operações de consulta a banco de dados relacional SQL."""
    pass

@final
@dataclass(frozen=True)
class FonteSql(FonteDados):
    """Adapter para ingestão de dados a partir de bancos relacionais (Postgres, SQLite, MySQL)."""

    url_conexao: str
    consulta_sql: str
    parametros: dict[str, str | int | float] | None = None

    @override
    def validar_conexao(self) -> bool:
        try:
            engine = create_engine(self.url_conexao)
            with engine.connect() as conexao:
                conexao.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @override
    def carregar_dados(self) -> pd.DataFrame:
        try:
            engine = create_engine(self.url_conexao)
            params = self.parametros or {}
            with engine.connect() as conexao:
                df = pd.read_sql_query(sql=text(self.consulta_sql), con=conexao, params=params)
            return df
        except Exception as erro:
            raise ErroFonteSql(
                f"Falha ao executar consulta SQL na base '{self.url_conexao}': {erro}"
            ) from erro
