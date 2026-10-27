"""Testes automatizados para os repositórios de dados e persistência SQLite."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from app.contratos_dados.repositorio_dados import RepositorioDados
from app.enums_dominio import TipoRepositorioStaging
from app.esquema_configuracao import ConfiguracaoRepositorioStaging
from app.repositorios_dados.fabrica_repositorios import (
    ErroFabricaRepositorios,
    FabricaRepositorios,
)
from app.repositorios_dados.repositorio_sqlite import (
    ErroRepositorioSqlite,
    RepositorioSqlite,
)


def test_repositorio_sqlite_gravar_e_ler(tmp_path: Path) -> None:
    """Valida gravação e posterior leitura fiel de registros no SQLite."""
    caminho_db = tmp_path / "banco_teste.db"
    repo = RepositorioSqlite(caminho=caminho_db, tabela="imoveis_staging")

    assert isinstance(repo, RepositorioDados)
    assert repo.verificar_existencia() is False

    dados_originais = pd.DataFrame(
        {
            "Valor_da_Venda": [450000, 680000],
            "Metragem": [75.5, 110.0],
            "Zona": ["Sul", "Leste"],
        }
    )

    total_gravado = repo.gravar_dados(dados_originais, substituir=True)
    assert total_gravado == 2
    assert repo.verificar_existencia() is True

    dados_lidos = repo.ler_dados()
    assert len(dados_lidos) == 2
    assert list(dados_lidos.columns) == ["Valor_da_Venda", "Metragem", "Zona"]
    assert dados_lidos["Valor_da_Venda"].tolist() == [450000, 680000]


def test_repositorio_sqlite_substituir_vs_append(tmp_path: Path) -> None:
    """Valida o comportamento dos modos replace e append no SQLite."""
    caminho_db = tmp_path / "append_teste.db"
    repo = RepositorioSqlite(caminho=caminho_db, tabela="teste_append")

    lote_1 = pd.DataFrame({"Valor_da_Venda": [100, 200]})
    lote_2 = pd.DataFrame({"Valor_da_Venda": [300, 400, 500]})

    repo.gravar_dados(lote_1, substituir=True)
    assert len(repo.ler_dados()) == 2

    # Append
    repo.gravar_dados(lote_2, substituir=False)
    assert len(repo.ler_dados()) == 5

    # Replace
    repo.gravar_dados(lote_1, substituir=True)
    assert len(repo.ler_dados()) == 2


def test_repositorio_sqlite_gravar_dataframe_vazio(tmp_path: Path) -> None:
    """Deve rejeitar tentativa de gravar DataFrame vazio."""
    repo = RepositorioSqlite(caminho=tmp_path / "vazio.db")
    with pytest.raises(ErroRepositorioSqlite, match="vazio"):
        repo.gravar_dados(pd.DataFrame())


def test_repositorio_sqlite_ler_tabela_inexistente(tmp_path: Path) -> None:
    """Deve levantar ErroRepositorioSqlite quando a tabela não foi criada."""
    repo = RepositorioSqlite(caminho=tmp_path / "inexistente.db", tabela="tabela_fantasma")
    with pytest.raises(ErroRepositorioSqlite, match="não existe"):
        repo.ler_dados()


def test_repositorio_sqlite_uri_prefix(tmp_path: Path) -> None:
    """Valida que o prefixo sqlite:/// é normalizado corretamente."""
    caminho_arquivo = tmp_path / "uri_teste.db"
    repo = RepositorioSqlite(f"sqlite:///{caminho_arquivo}")

    repo.gravar_dados(pd.DataFrame({"Valor_da_Venda": [1000]}), substituir=True)
    assert caminho_arquivo.is_file()
    assert len(repo.ler_dados()) == 1


def test_fabrica_repositorios_criacao_sqlite(tmp_path: Path) -> None:
    """Valida criação de RepositorioSqlite a partir da FabricaRepositorios."""
    config = ConfiguracaoRepositorioStaging(
        tipo=TipoRepositorioStaging.SQLITE, caminho=str(tmp_path / "fabrica.db"), tabela="staging"
    )
    repo = FabricaRepositorios.criar_repositorio(config)

    assert isinstance(repo, RepositorioSqlite)
    repo.gravar_dados(pd.DataFrame({"Valor_da_Venda": [999]}), substituir=True)
    assert len(repo.ler_dados()) == 1


def test_fabrica_repositorios_tipo_invalido() -> None:
    """Deve levantar ErroFabricaRepositorios para tipos não suportados."""
    config = ConfiguracaoRepositorioStaging.model_construct(
        tipo="banco_desconhecido", caminho="dsn", tabela="tab"
    )
    with pytest.raises(ErroFabricaRepositorios, match="Tipo de repositório não suportado"):
        FabricaRepositorios.criar_repositorio(config)
