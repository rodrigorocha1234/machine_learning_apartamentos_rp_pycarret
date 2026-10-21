from __future__ import annotations
from typing import final
from app.configuracao_projeto.leitor_configuracao import ValorConfiguracao
from app.contratos_dados.fonte_dados import FonteDados
from app.fontes_dados.fonte_api import FonteApi
from app.fontes_dados.fonte_excel import FonteExcel
from app.fontes_dados.fonte_sql import FonteSql

class ErroFabricaFontes(Exception):
    """Exceção levantada na instanciação de provedores de fontes de dados."""
    pass

@final
class FabricaFontes:
    """Fábrica GoF para desacoplamento e instanciação polimórfica de fontes de dados."""

    @staticmethod
    def criar_fonte(config_fonte: dict[str, ValorConfiguracao]) -> FonteDados:
        tipo = str(config_fonte.get("tipo", "xlsx")).lower().strip()

        if tipo in ("xlsx", "excel"):
            caminho = str(config_fonte.get("caminho", "storage-data/imoveis_apartamentos_rp.xlsx"))
            aba_raw = config_fonte.get("aba", 0)
            aba: str | int = int(aba_raw) if isinstance(aba_raw, int) or (isinstance(aba_raw, str) and aba_raw.isdigit()) else str(aba_raw)
            return FonteExcel(caminho=caminho, aba=aba)

        if tipo in ("sql", "banco", "database", "sqlite", "postgres"):
            url_conexao = str(config_fonte.get("url", config_fonte.get("caminho", "")))
            consulta_sql = str(config_fonte.get("consulta", config_fonte.get("sql", "SELECT * FROM imoveis")))
            params_raw = config_fonte.get("parametros")
            parametros = params_raw if isinstance(params_raw, dict) else None
            return FonteSql(url_conexao=url_conexao, consulta_sql=consulta_sql, parametros=parametros)

        if tipo in ("api", "rest", "http", "https"):
            url_endpoint = str(config_fonte.get("url", config_fonte.get("endpoint", "")))
            metodo = str(config_fonte.get("metodo", "GET"))
            cabecalhos_raw = config_fonte.get("cabecalhos")
            cabecalhos = cabecalhos_raw if isinstance(cabecalhos_raw, dict) else None
            params_raw = config_fonte.get("parametros")
            parametros = params_raw if isinstance(params_raw, dict) else None
            chave_dados = str(config_fonte["chave_dados"]) if "chave_dados" in config_fonte and config_fonte["chave_dados"] is not None else None
            return FonteApi(
                url_endpoint=url_endpoint,
                metodo_http=metodo,
                cabecalhos=cabecalhos,
                parametros=parametros,
                chave_dados=chave_dados
            )

        raise ErroFabricaFontes(f"Tipo de fonte de dados não suportado pela fábrica: '{tipo}'")
