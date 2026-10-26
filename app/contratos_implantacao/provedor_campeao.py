"""Contrato de acesso ao modelo campeão persistido para inferência em produção."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd
from pydantic import BaseModel, Field


class MetadadosModelo(BaseModel):
    """Metadados de governança e rastreabilidade do modelo campeão promovido."""

    nome_modelo: str = Field(description="Identificador do modelo (ex: lightgbm)")
    versao: str = Field(description="Versão semântica ou hash da rodada")
    algoritmo: str = Field(description="Classe ou framework do algoritmo")
    transformacao_alvo: str = Field(description="Função de transformação do alvo (ex: log1p ou nenhuma)")
    rmse_cv: float = Field(description="Métrica RMSE obtida na validação cruzada pareada")
    rmse_holdout: float = Field(description="Métrica RMSE obtida no holdout de 20% retido")
    r2_holdout: float = Field(description="Métrica R² obtida no holdout de 20% retido")
    data_promocao: str = Field(description="Timestamp ISO da aprovação e promoção")


@runtime_checkable
class ProvedorCampeao(Protocol):
    """Protocolo GoF Strategy para carregamento e inferência com o modelo campeão."""

    def prever(self, dados: pd.DataFrame) -> list[float]:
        """Executa a inferência sobre o DataFrame e retorna a lista de valores preditos em R$."""
        ...

    def obter_metadados(self) -> MetadadosModelo:
        """Retorna os metadados oficiais de governança do modelo promovido."""
        ...
