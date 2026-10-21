from __future__ import annotations
from dataclasses import dataclass
from typing import final, override
import pandas as pd
import requests
from app.contratos_dados.fonte_dados import ErroFonteDados, FonteDados

class ErroFonteApi(ErroFonteDados):
    """Exceção levantada em operações de consumo de API REST."""
    pass

@final
@dataclass(frozen=True)
class FonteApi(FonteDados):
    """Adapter para ingestão de dados a partir de serviços web e APIs REST."""

    url_endpoint: str
    metodo_http: str = "GET"
    cabecalhos: dict[str, str] | None = None
    parametros: dict[str, str | int | float] | None = None
    chave_dados: str | None = None
    timeout_segundos: float = 30.0

    @override
    def validar_conexao(self) -> bool:
        try:
            resposta = requests.head(
                self.url_endpoint,
                headers=self.cabecalhos,
                timeout=min(self.timeout_segundos, 5.0)
            )
            return resposta.status_code < 500
        except Exception:
            return False

    @override
    def carregar_dados(self) -> pd.DataFrame:
        try:
            resposta = requests.request(
                method=self.metodo_http.upper(),
                url=self.url_endpoint,
                headers=self.cabecalhos,
                params=self.parametros,
                timeout=self.timeout_segundos
            )
            resposta.raise_for_status()
            conteudo_json = resposta.json()

            if self.chave_dados and isinstance(conteudo_json, dict):
                if self.chave_dados not in conteudo_json:
                    raise ErroFonteApi(f"Chave '{self.chave_dados}' não encontrada na resposta da API.")
                itens = conteudo_json[self.chave_dados]
            else:
                itens = conteudo_json

            if not isinstance(itens, list):
                if isinstance(itens, dict):
                    itens = [itens]
                else:
                    raise ErroFonteApi("A resposta da API deve conter uma lista ou dicionário de registros.")

            return pd.DataFrame(itens)
        except Exception as erro:
            raise ErroFonteApi(f"Falha ao consumir dados da API '{self.url_endpoint}': {erro}") from erro
