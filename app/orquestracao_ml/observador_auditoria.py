from __future__ import annotations
import json
import os
from typing import final, override
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow, ObservadorMLflow

@final
class ObservadorAuditoria(ObservadorMLflow):
    """Observador concreto que registra eventos do MLflow em arquivo JSONL para governança e auditoria."""

    def __init__(self, caminho_arquivo: str = "storage-data/auditoria_mlflow.jsonl") -> None:
        self._caminho_arquivo = caminho_arquivo
        self._historico_eventos: list[EventoPublicacaoMLflow] = []
        os.makedirs(os.path.dirname(self._caminho_arquivo) or ".", exist_ok=True)

    @override
    def notificar(self, evento: EventoPublicacaoMLflow) -> None:
        """Registra o evento no histórico em memória e persiste em arquivo JSONL."""
        self._historico_eventos.append(evento)

        registro = {
            "timestamp_iso": evento.timestamp_iso,
            "tipo_evento": evento.tipo_evento,
            "status": evento.status,
            "modelo_campeao": evento.modelo_campeao,
            "modo_execucao": evento.modo_execucao,
            "experimento": evento.experimento,
            "nome_modelo_registrado": evento.nome_modelo_registrado,
            "run_id": evento.run_id,
            "model_uri": evento.model_uri,
            "mensagem": evento.mensagem,
            "metricas": evento.metricas,
            "tags": evento.tags
        }

        try:
            with open(self._caminho_arquivo, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except Exception:
            # Não interrompe o fluxo principal caso haja falha temporária de I/O
            pass

    def obter_historico(self) -> list[EventoPublicacaoMLflow]:
        """Retorna cópia do histórico de eventos recebidos em memória."""
        return list(self._historico_eventos)
