"""Fábrica GoF (Factory Method) para instanciação desacoplada de repositórios de staging."""

from __future__ import annotations

from typing import final
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.esquema_configuracao import ConfiguracaoRepositorioStaging
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite


class ErroFabricaRepositorios(Exception):
    """Exceção levantada quando um tipo de repositório não suportado é requisitado."""


@final
class FabricaRepositorios:
    """Fábrica para instanciação de repositórios de dados para persistência intermediária."""

    @staticmethod
    def criar_repositorio(configuracao: ConfiguracaoRepositorioStaging) -> RepositorioDados:
        """Cria e retorna uma instância concreta de RepositorioDados com base na configuração."""
        tipo_normalizado = configuracao.tipo.lower().strip()

        if tipo_normalizado == "sqlite":
            return RepositorioSqlite(
                caminho=configuracao.caminho,
                tabela=configuracao.tabela,
            )

        raise ErroFabricaRepositorios(
            f"Tipo de repositório não suportado pela fábrica: '{configuracao.tipo}'. "
            "Suportados atualmente: 'sqlite'."
        )
