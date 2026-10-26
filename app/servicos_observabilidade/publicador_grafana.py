"""Publicador de métricas e gerador de dashboard de observabilidade de drift no Grafana."""

from __future__ import annotations

import json
from pathlib import Path
from typing import final, override

from app.contratos_drift.detector_drift import ResultadoRelatorioDrift
from app.contratos_drift.publicador_metricas import PublicadorMetricas, ResultadoPublicacao
from app.esquema_configuracao import ConfiguracaoObservabilidade


class ErroPublicadorGrafana(Exception):
    """Exceção levantada em falhas durante a exportação ou provisionamento do dashboard."""


@final
class PublicadorGrafana(PublicadorMetricas):
    """Implementação de PublicadorMetricas gerando dashboards JSON para provisioning automático."""

    def __init__(
        self,
        config_observabilidade: ConfiguracaoObservabilidade,
        caminho_dashboard: Path | str = "config_ob/dashboards/dashboard_drift.json",
    ) -> None:
        self._config = config_observabilidade
        self._caminho_dashboard = Path(caminho_dashboard)

    @override
    def publicar_relatorio_drift(
        self,
        relatorio: ResultadoRelatorioDrift,
    ) -> ResultadoPublicacao:
        """Gera o arquivo JSON do dashboard do Grafana e persiste para provisioning."""
        try:
            dashboard_json = self._construir_dashboard(relatorio)
            self._caminho_dashboard.parent.mkdir(parents=True, exist_ok=True)

            with open(self._caminho_dashboard, "w", encoding="utf-8") as arq_dash:
                json.dump(dashboard_json, arq_dash, indent=2, ensure_ascii=False)

            return ResultadoPublicacao(
                destino="grafana",
                sucesso=True,
                caminho_artefato=str(self._caminho_dashboard),
                mensagem=(
                    f"Dashboard de monitoramento de drift provisionado com sucesso em '{self._caminho_dashboard}'."
                ),
            )
        except Exception as erro:
            raise ErroPublicadorGrafana(
                f"Falha ao gerar dashboard de drift para Grafana: {erro}"
            ) from erro

    def _construir_dashboard(self, relatorio: ResultadoRelatorioDrift) -> dict[str, object]:
        """Monta o schema JSON oficial do Grafana com painéis analíticos de drift."""
        # 1. Painel de Status Geral (Stat)
        cor_status = (
            "red"
            if relatorio.classificacao_drift == "real_concept_drift"
            else ("yellow" if "drift" in relatorio.classificacao_drift else "green")
        )

        painel_status: dict[str, object] = {
            "id": 1,
            "title": "Diagnóstico Geral de Drift",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 5},
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "justifyMode": "auto",
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "fixed", "fixedColor": cor_status},
                    "thresholds": {"mode": "absolute", "steps": []},
                }
            },
            "targets": [
                {
                    "refId": "A",
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "expr": "model_drift_status",
                }
            ],
            "description": relatorio.recomendacao_acao,
        }

        # 2. Painel de Indicadores Chave (Stat)
        painel_resumo: dict[str, object] = {
            "id": 2,
            "title": "Amostras e Cobertura",
            "type": "stat",
            "gridPos": {"x": 6, "y": 0, "w": 6, "h": 5},
            "options": {"colorMode": "value", "graphMode": "none"},
            "fieldConfig": {
                "defaults": {
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "blue", "value": None}],
                    },
                }
            },
            "targets": [],
            "description": (
                f"Referência: {relatorio.total_amostras_referencia} amostras | "
                f"Produção: {relatorio.total_amostras_producao} amostras"
            ),
        }

        # 3. Painel de Degradação de Erro RMSE (Gauge)
        degrau_valor = relatorio.degradacao_rmse_percentual or 0.0
        painel_degradacao: dict[str, object] = {
            "id": 3,
            "title": "Degradação de Performance (RMSE %)",
            "type": "gauge",
            "gridPos": {"x": 12, "y": 0, "w": 6, "h": 5},
            "options": {
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
            },
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": -20,
                    "max": 50,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": -20},
                            {"color": "yellow", "value": 10},
                            {"color": "red", "value": 15},
                        ],
                    },
                }
            },
            "targets": [],
            "description": f"Variação percentual: {degrau_valor:.2f}%",
        }

        # 4. Painel de Qualidade dos Dados (Stat)
        painel_qualidade: dict[str, object] = {
            "id": 4,
            "title": "Integridade dos Dados",
            "type": "stat",
            "gridPos": {"x": 18, "y": 0, "w": 6, "h": 5},
            "options": {"colorMode": "value", "graphMode": "none"},
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "red", "value": 0.05},
                        ],
                    }
                }
            },
            "targets": [],
            "description": (
                f"Taxa de Nulos: {relatorio.qualidade_dados.taxa_nulos_geral * 100:.2f}% | "
                f"Fora de Limites: {sum(relatorio.qualidade_dados.valores_fora_limites.values())}"
            ),
        }

        # 5. Painel de Detalhamento de Data Drift por Feature (Table)
        linhas_tabela: list[dict[str, object]] = []
        for det in relatorio.detalhes_variaveis:
            linhas_tabela.append(
                {
                    "Variavel": det.nome_variavel,
                    "Tipo": det.tipo_variavel,
                    "Metodo": det.metodo_estatistico,
                    "Estatistica": det.estatistica_teste,
                    "P-Valor": det.p_valor if det.p_valor is not None else "-",
                    "Limiar": det.limiar_alerta,
                    "Drift": "SIM" if det.drift_detectado else "NÃO",
                }
            )

        painel_tabela_drift: dict[str, object] = {
            "id": 5,
            "title": "Detecção Estatística de Covariate & Prediction Drift",
            "type": "table",
            "gridPos": {"x": 0, "y": 5, "w": 24, "h": 10},
            "options": {"sortBy": [{"displayName": "Drift", "desc": True}]},
            "fieldConfig": {
                "defaults": {
                    "custom": {"align": "center", "filterable": True},
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Drift"},
                        "properties": [
                            {
                                "id": "custom.cellOptions",
                                "value": {"type": "color-background", "mode": "basic"},
                            },
                            {
                                "id": "mappings",
                                "value": [
                                    {"type": "value", "options": {"SIM": {"color": "red"}}},
                                    {"type": "value", "options": {"NÃO": {"color": "green"}}},
                                ],
                            },
                        ],
                    }
                ],
            },
            "targets": [],
            "description": "Resultados dos testes de Kolmogorov-Smirnov e PSI feature a feature.",
        }

        dashboard_completo: dict[str, object] = {
            "annotations": {"list": []},
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "liveNow": False,
            "panels": [
                painel_status,
                painel_resumo,
                painel_degradacao,
                painel_qualidade,
                painel_tabela_drift,
            ],
            "refresh": "10s",
            "schemaVersion": 38,
            "style": "dark",
            "tags": ["mlops", "drift", "apartamentos-rp", "qualidade", "observabilidade"],
            "templating": {"list": []},
            "time": {"from": "now-6h", "to": "now"},
            "timepicker": {},
            "timezone": "browser",
            "title": "Monitoramento de Drift — Previsão Apartamentos RP",
            "uid": "drift-apartamentos-rp",
            "version": 1,
            "weekStart": "",
        }

        return dashboard_completo
