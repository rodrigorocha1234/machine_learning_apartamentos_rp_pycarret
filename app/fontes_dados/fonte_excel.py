"""Adaptador para ingestão de dados históricos a partir de planilhas Excel (XLSX/XLS)."""

from __future__ import annotations

from pathlib import Path
from typing import final, override
import pandas as pd

from app.contratos_dados.fonte_dados import FonteDados


class ErroFonteExcel(Exception):
    """Exceção levantada quando ocorre erro de validação ou leitura na fonte Excel."""


@final
class FonteExcel(FonteDados):
    """Implementação concreta de FonteDados para arquivos XLSX via Pandas/OpenPyXL."""

    def __init__(self, caminho: str | Path, aba: int | str = 0) -> None:
        self._caminho = Path(caminho)
        self._aba = aba

    @override
    def carregar_dados(self) -> pd.DataFrame:
        if not self._caminho.exists():
            raise ErroFonteExcel(
                f"Arquivo de dados Excel não encontrado: '{self._caminho.resolve()}'"
            )

        if self._caminho.suffix.lower() not in {".xlsx", ".xls"}:
            raise ErroFonteExcel(
                f"Extensão inválida para fonte Excel: '{self._caminho.suffix}'. Esperado '.xlsx' ou '.xls'."
            )

        try:
            dados = pd.read_excel(
                self._caminho,
                sheet_name=self._aba,
                engine="openpyxl",
            )
        except Exception as erro_leitura:
            raise ErroFonteExcel(
                f"Falha ao processar arquivo Excel '{self._caminho}': {erro_leitura}"
            ) from erro_leitura

        if dados.empty:
            raise ErroFonteExcel(
                f"A planilha Excel '{self._caminho}' (aba '{self._aba}') está vazia."
            )

        return dados

    @override
    def obter_origem(self) -> str:
        return f"Excel(arquivo='{self._caminho}', aba='{self._aba}')"
