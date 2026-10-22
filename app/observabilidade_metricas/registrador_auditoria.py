from __future__ import annotations
import json
import os
from typing import final, override
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow, ObservadorMLflow


@final
class RegistradorAuditoria(ObservadorMLflow):
    """Registra trilhas de auditoria e governança de eventos de modelos em formato estruturado JSONL."""

    def __init__(self, caminho_arquivo: str = "storage-data/auditoria_mlflow.jsonl") -> None:
        self._caminho_arquivo = caminho_arquivo
        self._historico_eventos: list[EventoPublicacaoMLflow] = []
        diretorio = os.path.dirname(self._caminho_arquivo) or "."
        if not os.path.exists(diretorio):
            os.makedirs(diretorio, exist_ok=True)

    @override
    def notificar(self, evento: EventoPublicacaoMLflow) -> None:
        """Adiciona o evento à memória e persiste no arquivo de auditoria append-only."""
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
            "tags": evento.tags,
        }

        try:
            with open(self._caminho_arquivo, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except Exception:
            # Falha de I/O em log não deve interromper pipeline produtivo
            pass

    def obter_historico(self) -> list[EventoPublicacaoMLflow]:
        """Retorna uma cópia do histórico dos eventos registrados."""
        return list(self._historico_eventos)
