from __future__ import annotations
from dataclasses import dataclass
import os
from typing import final, override
import pandas as pd
import polars as pl
from app.configuracao_projeto.leitor_configuracao import LeitorConfiguracao
from app.contratos_dados.fonte_dados import ErroFonteDados, FonteDados


class ErroFonteExcel(ErroFonteDados):
    """Exceção levantada em operações de leitura de planilhas Excel."""
    pass


@final
@dataclass(frozen=True)
class FonteExcel(FonteDados):
    """Adapter para ingestão de dados a partir de arquivos XLSX / Excel com suporte a Pandas e Polars."""

    caminho: str
    aba: str | int = 0
    motor: str | None = None
    caminho_configuracao: str = "configuracoes/projeto.yaml"

    @override
    def validar_conexao(self) -> bool:
        return os.path.exists(self.caminho) and os.path.isfile(self.caminho)

    def _obter_motor_selecionado(self) -> str:
        if self.motor is not None and self.motor.strip():
            return self.motor.strip().lower()

        if os.path.exists(self.caminho_configuracao):
            try:
                leitor = LeitorConfiguracao()
                configuracao = leitor.carregar_configuracao(self.caminho_configuracao)
                config_tabular = configuracao.get("tabular", {})
                if isinstance(config_tabular, dict) and "motor" in config_tabular:
                    motor_configurado = str(config_tabular["motor"]).strip().lower()
                    if motor_configurado:
                        return motor_configurado
            except Exception:
                pass
        return "pandas"

    def _carregar_com_polars(self) -> pl.DataFrame:
        try:
            if isinstance(self.aba, int):
                return pl.read_excel(self.caminho, sheet_id=self.aba + 1, engine="openpyxl")
            return pl.read_excel(self.caminho, sheet_name=str(self.aba), engine="openpyxl")
        except Exception:
            df_pandas = pd.read_excel(self.caminho, sheet_name=self.aba)
            return pl.from_pandas(df_pandas)

    @override
    def carregar_dados(self, motor: str | None = None) -> pd.DataFrame | pl.DataFrame:
        if not self.validar_conexao():
            raise ErroFonteExcel(
                f"Arquivo Excel não encontrado no caminho especificado: '{self.caminho}'")

        motor_efetivo = motor.strip().lower() if motor and motor.strip() else self._obter_motor_selecionado()

        try:
            if motor_efetivo == "polars":
                return self._carregar_com_polars()
            return pd.read_excel(self.caminho, sheet_name=self.aba)
        except Exception as erro:
            raise ErroFonteExcel(
                f"Falha ao carregar dados do Excel '{self.caminho}', aba '{self.aba}' com motor '{motor_efetivo}': {erro}") from erro

