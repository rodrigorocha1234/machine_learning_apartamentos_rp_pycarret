"""Configurações globais e fixtures para testes pytest."""

from __future__ import annotations

import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def configurar_ambiente_teste() -> None:
    """Garante variáveis de ambiente padrão para testes."""
    os.environ.setdefault("IMOVEIS_XLSX_PATH", "storage-data/imoveis_apartamentos_rp.xlsx")
    os.environ.setdefault("STAGING_SQLITE_PATH", "storage-data/staging_imoveis.db")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("GRAFANA_URL", "http://localhost:3000")


@pytest.fixture
def diretorio_configuracoes() -> Path:
    """Retorna o caminho do diretório contendo as etapas modulares."""
    return Path("configuracoes")


@pytest.fixture
def diretorio_copia_modular(tmp_path: Path, diretorio_configuracoes: Path) -> Path:
    """Copia os arquivos YAML modulares para uma pasta temporária para testes de mutação."""
    pasta_destino = tmp_path / "configuracoes"
    pasta_destino.mkdir()
    for arq in sorted(diretorio_configuracoes.glob("*.yaml")):
        (pasta_destino / arq.name).write_text(arq.read_text(encoding="utf-8"), encoding="utf-8")
    return pasta_destino
