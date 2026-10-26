"""Testes para o publicador e gerador de dashboards de drift no Grafana."""

from __future__ import annotations

import json
from pathlib import Path

from app.contratos_drift.detector_drift import MetricasQualidadeDados, ResultadoRelatorioDrift
from app.esquema_configuracao import ConfiguracaoObservabilidade
from app.servicos_observabilidade.publicador_grafana import PublicadorGrafana


def test_geracao_dashboard_grafana_drift(tmp_path: Path) -> None:
    caminho_dash = tmp_path / "dashboard_drift.json"
    config_obs = ConfiguracaoObservabilidade(grafana_url="http://localhost:3000")
    publicador = PublicadorGrafana(config_obs, caminho_dashboard=caminho_dash)

    relatorio = ResultadoRelatorioDrift(
        timestamp_analise="2026-10-26T00:00:00Z",
        total_amostras_referencia=280,
        total_amostras_producao=70,
        qualidade_dados=MetricasQualidadeDados(
            total_amostras=70,
            taxa_nulos_geral=0.0,
            contagem_nulos_por_coluna={},
            novas_categorias_detectadas={},
            valores_fora_limites={},
        ),
        covariate_drift_detectado=False,
        prediction_drift_detectado=False,
        target_drift_detectado=False,
        concept_drift_detectado=False,
        classificacao_drift="sem_drift",
        degradacao_rmse_percentual=1.2,
        variaveis_com_drift=[],
        detalhes_variaveis=[],
        recomendacao_acao="Operação nominal",
    )

    res = publicador.publicar_relatorio_drift(relatorio)

    assert res.sucesso is True
    assert caminho_dash.exists()

    with open(caminho_dash, "r", encoding="utf-8") as f:
        dados_json = json.load(f)

    assert dados_json["uid"] == "drift-apartamentos-rp"
    assert len(dados_json["panels"]) >= 4
