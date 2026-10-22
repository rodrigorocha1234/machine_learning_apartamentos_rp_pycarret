from __future__ import annotations
from typing import final
from app.observabilidade_metricas.registrador_auditoria import RegistradorAuditoria


@final
class ObservadorAuditoria(RegistradorAuditoria):
    """Observador concreto de auditoria que delega ao RegistradorAuditoria."""
    pass
