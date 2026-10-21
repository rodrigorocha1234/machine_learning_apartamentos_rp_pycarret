from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd

type ValorMetricaParametro = (
    str
    | int
    | float
    | bool
    | list[str]
    | list[int]
    | list[float]
    | dict[str, str | int | float | bool | dict[str, float]]
)

@runtime_checkable
class ModeloPreditor(Protocol):
    """Contrato estrito para estimadores treinados que realizam inferência tabular."""

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Produz predições a partir dos dados de entrada fornecidos."""
        ...

@runtime_checkable
class MotorTreino(Protocol):
    """Contrato para motores de treino, validação cruzada e tuning de modelos."""

    def ajustar_parametros(
        self,
        dados: pd.DataFrame,
        alvo: str,
        iteracoes: int,
        modelo_nome: str | None = None,
        metodo_tuning: str | None = None,
    ) -> dict[str, ValorMetricaParametro]:
        """Ajusta hiperparâmetros e retorna métricas e parâmetros selecionados."""
        ...

    def avaliar_modelos(self, dados: pd.DataFrame, alvo: str) -> pd.DataFrame:
        """Avalia modelos em validação cruzada pareada e retorna DataFrame com perdas por fold."""
        ...

    def treinar_final(
        self,
        dados: pd.DataFrame,
        alvo: str,
        modelo_nome: str | None = None,
    ) -> ModeloPreditor:
        """Treina e finaliza o modelo campeão no conjunto integral de desenvolvimento."""
        ...
