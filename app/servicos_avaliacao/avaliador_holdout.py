"""Avaliador definitivo de modelos sobre o conjunto de holdout isolado."""

from __future__ import annotations

from typing import Protocol, cast, final
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class PreditorTreinado(Protocol):
    """Protocolo para estimador com método predict."""

    def predict(self, X: pd.DataFrame) -> object: ...


class MetricasHoldout(BaseModel):
    """Métricas de performance no conjunto de teste holdout intocado."""

    nome_modelo: str = Field(description="Identificador do modelo")
    rmse: float = Field(ge=0.0, description="Root Mean Squared Error")
    mae: float = Field(ge=0.0, description="Mean Absolute Error")
    r2: float = Field(description="Coeficiente de Determinação R2")
    mape: float = Field(ge=0.0, description="Mean Absolute Percentage Error em porcentagem")
    medae: float = Field(ge=0.0, description="Mediana do Erro Absoluto")


class ErroAvaliadorHoldout(Exception):
    """Exceção levantada em falhas na avaliação do conjunto holdout."""


@final
class AvaliadorHoldout:
    """Avaliador cego de generalização no conjunto holdout."""

    def __init__(self, coluna_alvo: str = "Valor_da_Venda") -> None:
        self._coluna_alvo = coluna_alvo

    def avaliar_modelos_holdout(
        self,
        modelos_treinados: dict[str, object],
        dados_holdout: pd.DataFrame,
    ) -> dict[str, MetricasHoldout]:
        """Calcula as métricas reais de generalização em dados intocados."""
        if dados_holdout.empty:
            raise ErroAvaliadorHoldout("DataFrame de holdout está vazio.")
        if self._coluna_alvo not in dados_holdout.columns:
            raise ErroAvaliadorHoldout(f"Coluna alvo '{self._coluna_alvo}' ausente no holdout.")
        if not modelos_treinados:
            raise ErroAvaliadorHoldout("Dicionário de modelos para avaliação no holdout está vazio.")

        X_holdout = dados_holdout.drop(columns=[self._coluna_alvo for _ in [1]])
        y_real = np.asarray(dados_holdout[self._coluna_alvo].to_numpy(), dtype=np.float64)
        var_y = float(np.var(y_real))

        resultados: dict[str, MetricasHoldout] = {}

        for nome, mod in modelos_treinados.items():
            preditor = cast(PreditorTreinado, mod)
            preds = np.asarray(preditor.predict(X_holdout), dtype=np.float64)

            residuos = y_real - preds
            residuos_abs = np.abs(residuos)

            rmse = float(np.sqrt(np.mean(residuos**2)))
            mae = float(np.mean(residuos_abs))
            r2 = float(1.0 - (float(np.mean(residuos**2)) / var_y)) if var_y > 0 else 0.0

            # MAPE com proteção contra divisão por zero
            y_seguro = np.where(y_real == 0, 1.0, y_real)
            mape = float(np.mean(residuos_abs / y_seguro) * 100.0)
            medae = float(np.median(residuos_abs))

            resultados[nome] = MetricasHoldout(
                nome_modelo=nome,
                rmse=round(rmse, 2),
                mae=round(mae, 2),
                r2=round(r2, 4),
                mape=round(mape, 2),
                medae=round(medae, 2),
            )

        return resultados
