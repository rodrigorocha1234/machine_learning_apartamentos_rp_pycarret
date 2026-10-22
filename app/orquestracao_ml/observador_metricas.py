from __future__ import annotations
from typing import final
from app.observabilidade_metricas.coletor_telemetria import ColetorTelemetria


@final
class ObservadorMetricas(ColetorTelemetria):
    """Observador concreto de métricas operacionais que delega ao ColetorTelemetria."""
    pass
