"""Adaptador de visualização e publicação de dashboards para Grafana."""

from __future__ import annotations

import json
from pathlib import Path
from typing import final, override

from app.contratos_visualizacao.adaptador_visualizacao import (
    AdaptadorVisualizacao,
    ResultadoAnaliseEda,
    ResultadoVisualizacao,
)
from app.esquema_configuracao import ConfiguracaoEda


class ErroAdaptadorGrafana(Exception):
    """Exceção levantada em falhas de renderização ou publicação para o Grafana."""


@final
class AdaptadorGrafana(AdaptadorVisualizacao):
    """Adaptador concreto para exportação de dashboards provisionados e métricas no Grafana."""

    def __init__(
        self,
        caminho_dashboard_provisionado: str | Path = "config_ob/dashboards/dashboard_eda.json",
    ) -> None:
        self._caminho_dashboard = Path(caminho_dashboard_provisionado)

    @override
    def obter_nome_motor(self) -> str:
        return "grafana"

    @override
    def renderizar_analises(
        self, analises: list[ResultadoAnaliseEda], configuracao: ConfiguracaoEda
    ) -> ResultadoVisualizacao:
        if not analises:
            return ResultadoVisualizacao(
                motor_utilizado=self.obter_nome_motor(),
                total_paineis=0,
                arquivos_gerados=[],
                sucesso=True,
                mensagem="Nenhuma análise ativa para renderização.",
            )

        arquivos_gerados: list[str] = []

        try:
            # 1. Geração do Dashboard JSON do Grafana para provisioning automático
            dashboard_json = self._construir_dashboard_grafana(analises)
            self._caminho_dashboard.parent.mkdir(parents=True, exist_ok=True)
            self._caminho_dashboard.write_text(
                json.dumps(dashboard_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            arquivos_gerados.append(str(self._caminho_dashboard))

            # 2. Exportação de dados consolidados em JSON
            caminho_relatorio = Path(configuracao.caminho_relatorio_json)
            caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

            dados_exportacao: dict[str, object] = {
                "titulo": "Relatório Analítico de EDA - Apartamentos Ribeirão Preto",
                "motor": "grafana",
                "quantidade_analises": len(analises),
                "dashboard_provisionado": str(self._caminho_dashboard),
                "analises": [analise.model_dump() for analise in analises],
            }

            caminho_relatorio.write_text(
                json.dumps(dados_exportacao, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            arquivos_gerados.append(str(caminho_relatorio))

            return ResultadoVisualizacao(
                motor_utilizado=self.obter_nome_motor(),
                total_paineis=len(analises),
                arquivos_gerados=arquivos_gerados,
                sucesso=True,
                mensagem=f"Dashboard Grafana gerado com sucesso em '{self._caminho_dashboard}'.",
            )

        except Exception as erro:
            raise ErroAdaptadorGrafana(
                f"Falha ao gerar dashboard ou métricas para o Grafana: {erro}"
            ) from erro

    def _construir_dashboard_grafana(
        self, analises: list[ResultadoAnaliseEda]
    ) -> dict[str, object]:
        """Constrói a estrutura JSON do dashboard compatível com o schema do Grafana."""
        paineis: list[dict[str, object]] = []
        painel_id = 1
        pos_y = 0

        # Painel 1: Cabeçalho com visão geral (Stat)
        total_analises = len(analises)
        paineis.append(
            {
                "id": painel_id,
                "title": "Visão Geral da EDA",
                "type": "stat",
                "gridPos": {"h": 4, "w": 24, "x": 0, "y": pos_y},
                "options": {
                    "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                    "orientation": "horizontal",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "none",
                },
                "description": f"Pipeline EDA executado com sucesso: {total_analises} análises ativas.",
            }
        )
        painel_id += 1
        pos_y += 4

        for analise in analises:
            largura = 12
            altura = 8
            pos_x = (painel_id % 2) * 12

            if analise.tipo_grafico == "histograma":
                tipo_painel = "barchart"
            elif analise.tipo_grafico == "boxplot":
                tipo_painel = "barchart"
            elif analise.tipo_grafico == "correlacao":
                tipo_painel = "table"
            elif analise.tipo_grafico == "tabela":
                tipo_painel = "table"
                largura = 24
                pos_x = 0
            else:
                tipo_painel = "xychart"

            painel_dict: dict[str, object] = {
                "id": painel_id,
                "title": analise.titulo,
                "type": tipo_painel,
                "gridPos": {"h": altura, "w": largura, "x": pos_x, "y": pos_y},
                "description": analise.descricao or f"Análise {analise.nome}",
                "custom": {
                    "tipo_eda": analise.tipo_grafico,
                    "metricas_resumo": analise.metricas_resumo,
                    "dados_estruturados": analise.dados_estruturados,
                },
            }

            paineis.append(painel_dict)
            painel_id += 1
            if pos_x == 0 and largura == 24:
                pos_y += altura
            elif pos_x == 12:
                pos_y += altura

        dashboard: dict[str, object] = {
            "title": "EDA - Previsão de Preços de Apartamentos RP",
            "uid": "eda-apartamentos-rp",
            "schemaVersion": 38,
            "version": 1,
            "timezone": "America/Sao_Paulo",
            "refresh": "10s",
            "tags": ["ml", "eda", "apartamentos", "ribeirao-preto"],
            "panels": paineis,
            "time": {"from": "now-6h", "to": "now"},
            "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m"]},
        }

        return dashboard
