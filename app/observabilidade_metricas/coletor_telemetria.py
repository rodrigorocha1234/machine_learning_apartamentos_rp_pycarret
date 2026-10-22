from __future__ import annotations
import os
from typing import final, override
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow, ObservadorMLflow


@final
class ColetorTelemetria(ObservadorMLflow):
    """Coleta e exporta métricas de telemetria operacional de publicação para formato compatível com Prometheus/Grafana."""

    def __init__(self, caminho_metricas_prom: str = "storage-data/metricas_mlflow.prom") -> None:
        self._caminho_metricas_prom = caminho_metricas_prom
        self.total_sucessos: int = 0
        self.total_falhas: int = 0
        diretorio = os.path.dirname(self._caminho_metricas_prom) or "."
        if not os.path.exists(diretorio):
            os.makedirs(diretorio, exist_ok=True)

    @override
    def notificar(self, evento: EventoPublicacaoMLflow) -> None:
        """Atualiza os contadores de telemetria e grava o arquivo de scraping do Prometheus."""
        if evento.status == "sucesso":
            self.total_sucessos += 1
        else:
            self.total_falhas += 1

        conteudo_prom = (
            "# HELP mlflow_publicacao_sucesso_total Total de modelos publicados com sucesso no MLflow Registry\n"
            "# TYPE mlflow_publicacao_sucesso_total counter\n"
            f"mlflow_publicacao_sucesso_total {self.total_sucessos}\n\n"
            "# HELP mlflow_publicacao_falha_total Total de falhas durante publicacao no MLflow\n"
            "# TYPE mlflow_publicacao_falha_total counter\n"
            f"mlflow_publicacao_falha_total {self.total_falhas}\n\n"
            "# HELP mlflow_ultimo_modelo_info Metadados do ultimo modelo campeao publicado\n"
            "# TYPE mlflow_ultimo_modelo_info gauge\n"
            f'mlflow_ultimo_modelo_info{{modelo="{evento.modelo_campeao}",modo="{evento.modo_execucao}",experimento="{evento.experimento}"}} 1\n'
        )

        try:
            with open(self._caminho_metricas_prom, "w", encoding="utf-8") as f:
                f.write(conteudo_prom)
        except Exception:
            # Falhas transitórias de escrita de telemetria não devem quebrar o fluxo principal
            pass
