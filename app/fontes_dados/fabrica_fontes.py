"""Fábrica GoF (Factory Method) para instanciação desacoplada de fontes de dados."""

from __future__ import annotations

from typing import final
from app.contratos_dados.fonte_dados import FonteDados
from app.esquema_configuracao import ConfiguracaoFonte
from app.fontes_dados.fonte_excel import FonteExcel


class ErroFabricaFontes(Exception):
    """Exceção levantada quando um tipo de fonte não suportado é requisitado."""


@final
class FabricaFontes:
    """Fábrica para instanciação de estratégias de ingestão de dados."""

    @staticmethod
    def criar_fonte(configuracao: ConfiguracaoFonte) -> FonteDados:
        """Cria e retorna uma instância concreta de FonteDados com base na configuração."""
        tipo_normalizado = configuracao.tipo.lower().strip()

        if tipo_normalizado == "xlsx":
            return FonteExcel(
                caminho=configuracao.caminho,
                aba=configuracao.aba,
            )

        raise ErroFabricaFontes(
            f"Tipo de fonte de dados não suportado pela fábrica: '{configuracao.tipo}'. "
            "Suportados atualmente: 'xlsx'."
        )
