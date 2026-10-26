"""Testes automatizados integrados para o orquestrador de drift e observabilidade."""

from __future__ import annotations

from pathlib import Path
from app.conf_yaml import ConfYAML
from app.servicos_observabilidade.orquestrador_drift import OrquestradorDrift


def test_orquestrador_executa_monitoramento() -> None:
    conf = ConfYAML()
    orquestrador = OrquestradorDrift.da_configuracao(conf)

    relatorio = orquestrador.executar_monitoramento(
        caminho_artefatos_campeao="production_artifacts/campeao"
    )

    assert relatorio.total_amostras_referencia > 0
    assert relatorio.total_amostras_producao > 0
    assert relatorio.classificacao_drift in {
        "sem_drift",
        "virtual_drift",
        "real_concept_drift",
        "label_drift",
    }
    assert len(relatorio.detalhes_variaveis) >= 4

    # Verifica persistência do relatório e do JSON
    assert Path("production_artifacts/monitoramento/relatorio_monitoramento.md").exists()
    assert Path("storage-data/relatorio_drift.json").exists()
    assert Path("config_ob/dashboards/dashboard_drift.json").exists()
