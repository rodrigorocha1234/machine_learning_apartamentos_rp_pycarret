import os
import pytest
from app.configuracao_projeto.leitor_configuracao import LeitorConfiguracao

def test_carregar_projeto_yaml() -> None:
    os.environ["IMOVEIS_XLSX_PATH"] = "storage-data/imoveis_apartamentos_rp.xlsx"
    os.environ["STAGING_DATABASE_URL"] = "storage-data/staging_imoveis.db"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["GRAFANA_URL"] = "http://localhost:3000"

    leitor = LeitorConfiguracao()
    config = leitor.carregar_configuracao("configuracoes/projeto.yaml")

    assert "projeto" in config
    assert "fonte" in config
    fonte = config["fonte"]
    assert isinstance(fonte, dict)
    assert fonte["caminho"] == "storage-data/imoveis_apartamentos_rp.xlsx"
    assert "regras_negocio" in config
    regras = config["regras_negocio"]
    assert isinstance(regras, dict)
    assert regras["ativo"] is True
