"""Contratos e modelos para construção e avaliação de comitês de votação (Ensemble)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd
from pydantic import BaseModel, Field

from app.esquema_configuracao import ConfiguracaoVotacao


class ResultadoComite(BaseModel):
    """Métricas e metadados da combinação de regressores em comitê de votação."""

    tecnica_utilizada: str = Field(description="Estratégia de combinação (media_simples, voting_regressor, etc.)")
    ativo: bool = Field(description="Indica se a votação/ensemble foi efetivamente ativada")
    modelos_membros: list[str] = Field(description="Nomes dos modelos que compõem o comitê")
    pesos: dict[str, float] = Field(
        default_factory=dict, description="Pesos atribuídos a cada modelo membro no ensemble"
    )
    metricas_holdout: dict[str, float] = Field(
        default_factory=dict, description="Performance do ensemble avaliado no conjunto holdout"
    )


@runtime_checkable
class ConstrutorComite(Protocol):
    """Protocolo GoF Strategy para instanciação e teste de comitês de votação."""

    def construir_e_avaliar(
        self,
        modelos_treinados: dict[str, object],
        dados_treino: pd.DataFrame,
        dados_holdout: pd.DataFrame,
        coluna_alvo: str,
        config_votacao: ConfiguracaoVotacao,
    ) -> ResultadoComite:
        """Monta o ensemble configurado e avalia sua performance no holdout isolado."""
        ...
