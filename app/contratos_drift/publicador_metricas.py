"""Contrato para publicadores e exportadores de métricas de observabilidade e dashboards."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field

from app.contratos_drift.detector_drift import ResultadoRelatorioDrift


class ResultadoPublicacao(BaseModel):
    """Evidência do envio ou provisionamento de métricas no destino de observabilidade."""

    destino: str = Field(description="Identificador do destino (ex: grafana, prometheus)")
    sucesso: bool = Field(description="Status booleano do provisionamento ou envio")
    caminho_artefato: str | None = Field(default=None, description="Caminho do dashboard exportado")
    mensagem: str = Field(description="Descrição do resultado")


@runtime_checkable
class PublicadorMetricas(Protocol):
    """Protocolo GoF Strategy para publicação de telemetria e dashboards."""

    def publicar_relatorio_drift(
        self,
        relatorio: ResultadoRelatorioDrift,
    ) -> ResultadoPublicacao:
        """Exporta métricas de drift e atualiza o dashboard de observabilidade."""
        ...
