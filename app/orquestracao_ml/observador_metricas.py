from __future__ import annotations
import os
from typing import final, override
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow, ObservadorMLflow

@final
class ObservadorMetricas(ObservadorMLflow):
    """Observador concreto que calcula e exporta contadores operacionais de publicação do MLflow para Prometheus."""

    def __init__(self, caminho_metricas_prom: str = "storage-data/metricas_mlflow.prom") -> None:
        self._caminho_metricas_prom = caminho_metricas_prom
        self.total_sucessos: int = 0
        self.total_falhas: int = 0
        os.makedirs(os.path.dirname(self._caminho_metricas_prom) or ".", exist_ok=True)

    @override
    def notificar(self, evento: EventoPublicacaoMLflow) -> None:
        """Atualiza contadores e regrava arquivo de métricas Prometheus."""
        if evento.status == "sucesso":
            self.total_sucessos += 1
        else:
            self.total_falhas += 1

        conteudo_prom = (
            f"# HELP mlflow_publicacao_sucesso_total Total de modelos publicados com sucesso no MLflow Registry\n"
            f"# TYPE mlflow_publicacao_sucesso_total counter\n"
            f"mlflow_publicacao_sucesso_total {self.total_sucessos}\n\n"
            f"# HELP mlflow_publicacao_falha_total Total de falhas durante publicacao no MLflow\n"
            f"# TYPE mlflow_publicacao_falha_total counter\n"
            f"mlflow_publicacao_falha_total {self.total_falhas}\n\n"
            f"# HELP mlflow_ultimo_modelo_info Metadados do ultimo modelo campeao publicado\n"
            f"# TYPE mlflow_ultimo_modelo_info gauge\n"
            f'mlflow_ultimo_modelo_info{{modelo="{evento.modelo_campeao}",modo="{evento.modo_execucao}",experimento="{evento.experimento}"}} 1\n'
        )

        try:
            with open(self._caminho_metricas_prom, "w", encoding="utf-8") as f:
                f.write(conteudo_prom)
        except Exception:
            # Falha de escrita de métrica não deve quebrar o pipeline
            pass
