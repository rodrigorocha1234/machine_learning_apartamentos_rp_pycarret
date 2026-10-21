from __future__ import annotations
from dataclasses import dataclass
import os
from typing import final, override
import pandas as pd
from app.contratos_dados.fonte_dados import ErroFonteDados, FonteDados


class ErroFonteExcel(ErroFonteDados):
    """Exceção levantada em operações de leitura de planilhas Excel."""
    pass


@final
@dataclass(frozen=True)
class FonteExcel(FonteDados):
    """Adapter para ingestão de dados a partir de arquivos XLSX / Excel."""

    caminho: str
    aba: str | int = 0

    @override
    def validar_conexao(self) -> bool:
        return os.path.exists(self.caminho) and os.path.isfile(self.caminho)

    @override
    def carregar_dados(self) -> pd.DataFrame:
        if not self.validar_conexao():
            raise ErroFonteExcel(
                f"Arquivo Excel não encontrado no caminho especificado: '{self.caminho}'")
        try:
            return pd.read_excel(self.caminho, sheet_name=self.aba)
        except Exception as erro:
            raise ErroFonteExcel(
                f"Falha ao carregar dados do Excel '{self.caminho}', aba '{self.aba}': {erro}") from erro
