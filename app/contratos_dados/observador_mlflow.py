from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@dataclass(frozen=True)
class EventoPublicacaoMLflow:
    """Payload de dados imutável transmitido pelo Subject (PublicadorServico) aos Observers."""
    tipo_evento: str  # ex: "modelo_publicado", "falha_publicacao"
    status: str       # "sucesso", "erro", "aviso"
    modelo_campeao: str
    modo_execucao: str
    experimento: str
    nome_modelo_registrado: str
    run_id: str = ""
    model_uri: str = ""
    mensagem: str = ""
    metricas: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    timestamp_iso: str = ""

@runtime_checkable
class ObservadorMLflow(Protocol):
    """Protocolo GoF Observer para eventos de ciclo de vida e publicação no MLflow."""
    def notificar(self, evento: EventoPublicacaoMLflow) -> None:
        """Recebe notificação síncrona sobre mudanças de estado e eventos de publicação no MLflow."""
        ...
