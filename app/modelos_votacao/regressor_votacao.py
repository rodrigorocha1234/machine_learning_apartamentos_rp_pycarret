from __future__ import annotations
from typing import final, override
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from app.contratos_dados.motor_treino import ModeloPreditor

class ErroRegressorVotacao(Exception):
    """Exceção levantada em operações do ensemble de votação para regressão."""
    pass

@final
class RegressorVotacao(BaseEstimator, RegressorMixin):
    """Regressor de conjunto (ensemble) com múltiplas estratégias de votação/agregação para regressão."""

    _ESTRATEGIAS_VALIDAS: set[str] = {
        "media_simples",
        "voting_regressor",
        "media_ponderada_rmse",
        "media_ponderada_postos",
        "mediana",
        "media_aparada"
    }

    def __init__(
        self,
        estimadores: list[tuple[str, ModeloPreditor]],
        estrategia: str = "media_simples",
        pesos: list[float] | None = None,
    ) -> None:
        self.estimadores = estimadores
        self.estrategia = estrategia.lower().strip()
        self.pesos = pesos

        if self.estrategia not in self._ESTRATEGIAS_VALIDAS:
            raise ErroRegressorVotacao(
                f"Estratégia de votação inválida: '{estrategia}'. Válidas: {self._ESTRATEGIAS_VALIDAS}"
            )
        if not self.estimadores:
            raise ErroRegressorVotacao("A lista de estimadores para votação não pode ser vazia.")

    @override
    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> RegressorVotacao:
        for nome, estimador in self.estimadores:
            if hasattr(estimador, "fit"):
                estimador.fit(x, y)
        return self

    @override
    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        predicoes_lista: list[np.ndarray] = []

        for nome, estimador in self.estimadores:
            if not hasattr(estimador, "predict"):
                raise ErroRegressorVotacao(f"O estimador '{nome}' não possui método predict.")
            pred = estimador.predict(x)
            predicoes_lista.append(np.asarray(pred, dtype=float).ravel())

        # Matriz com shape (n_amostras, n_modelos)
        matriz_predicoes = np.column_stack(predicoes_lista)

        if self.estrategia in ("media_simples", "voting_regressor"):
            return np.mean(matriz_predicoes, axis=1)

        if self.estrategia in ("media_ponderada_rmse", "media_ponderada_postos"):
            if self.pesos is not None and len(self.pesos) == matriz_predicoes.shape[1]:
                pesos_norm = np.asarray(self.pesos, dtype=float)
                soma = float(np.sum(pesos_norm))
                if soma > 0:
                    pesos_norm = pesos_norm / soma
                    return np.average(matriz_predicoes, axis=1, weights=pesos_norm)
            return np.mean(matriz_predicoes, axis=1)

        if self.estrategia == "mediana":
            return np.median(matriz_predicoes, axis=1)

        if self.estrategia == "media_aparada":
            n_modelos = matriz_predicoes.shape[1]
            if n_modelos >= 3:
                # Descarta a menor e a maior predição por amostra e tira a média do intervalo central
                ordenadas = np.sort(matriz_predicoes, axis=1)
                return np.mean(ordenadas[:, 1:-1], axis=1)
            return np.mean(matriz_predicoes, axis=1)

        return np.mean(matriz_predicoes, axis=1)
