from __future__ import annotations
from typing import final
import numpy as np
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from app.contratos_dados.motor_treino import ModeloPreditor
from app.modelos_votacao.regressor_votacao import RegressorVotacao

class ErroFabricaVotacao(Exception):
    """Exceção levantada na fábrica de ensembles de votação."""
    pass

@final
class FabricaVotacao:
    """Fábrica GoF para montagem desacoplada de técnicas de votação e combinação para regressão."""

    _TECNICAS_DISPONIVEIS: set[str] = {
        "media_simples",
        "voting_regressor",
        "media_ponderada_rmse",
        "media_ponderada_postos",
        "mediana",
        "media_aparada",
        "stacking"
    }

    @staticmethod
    def listar_tecnicas_disponiveis() -> list[str]:
        return sorted(list(FabricaVotacao._TECNICAS_DISPONIVEIS))

    @staticmethod
    def criar_ensemble_votacao(
        estimadores: list[tuple[str, ModeloPreditor]],
        tecnica: str = "media_simples",
        metricas_rmse: dict[str, float] | None = None,
        postos_friedman: dict[str, float] | None = None,
        meta_modelo: ModeloPreditor | None = None,
        cv_folds: int = 3,
    ) -> ModeloPreditor:
        tecnica_norm = tecnica.lower().strip()

        if tecnica_norm not in FabricaVotacao._TECNICAS_DISPONIVEIS:
            raise ErroFabricaVotacao(
                f"Técnica de votação '{tecnica}' não suportada. Disponíveis: {sorted(list(FabricaVotacao._TECNICAS_DISPONIVEIS))}"
            )

        if not estimadores:
            raise ErroFabricaVotacao("A lista de estimadores para votação não pode ser vazia.")

        # 1. Stacking Regressor com meta-modelo
        if tecnica_norm == "stacking":
            estimador_final = meta_modelo if meta_modelo is not None else RidgeCV()
            return StackingRegressor(
                estimators=estimadores,
                final_estimator=estimador_final,
                cv=cv_folds
            )

        # 2. VotingRegressor nativo do Scikit-Learn
        if tecnica_norm == "voting_regressor":
            return VotingRegressor(estimators=estimadores)

        # 2. Média Ponderada pelo Inverso do Erro (RMSE)
        if tecnica_norm == "media_ponderada_rmse":
            pesos: list[float] = []
            if metricas_rmse:
                inversos: list[float] = []
                for nome, _ in estimadores:
                    rmse = metricas_rmse.get(nome, 1.0)
                    # Evita divisão por zero adicionando pequeno epsilon
                    inversos.append(1.0 / max(float(rmse), 1e-6))
                soma_inversos = sum(inversos)
                pesos = [inv / soma_inversos for inv in inversos]
            else:
                pesos = [1.0 / len(estimadores)] * len(estimadores)

            return RegressorVotacao(
                estimadores=estimadores,
                estrategia="media_ponderada_rmse",
                pesos=pesos
            )

        # 3. Média Ponderada pelos Postos de Friedman
        if tecnica_norm == "media_ponderada_postos":
            pesos = []
            if postos_friedman:
                k = len(estimadores)
                pesos_brutos: list[float] = []
                for nome, _ in estimadores:
                    posto = postos_friedman.get(nome, float(k))
                    # Posto 1 recebe peso k, posto k recebe peso 1
                    peso_modelo = max(float(k - posto + 1.0), 0.1)
                    pesos_brutos.append(peso_modelo)
                soma_pesos = sum(pesos_brutos)
                pesos = [p / soma_pesos for p in pesos_brutos]
            else:
                pesos = [1.0 / len(estimadores)] * len(estimadores)

            return RegressorVotacao(
                estimadores=estimadores,
                estrategia="media_ponderada_postos",
                pesos=pesos
            )

        # 4. Mediana (robusta a previsões discrepantes / outliers)
        if tecnica_norm == "mediana":
            return RegressorVotacao(
                estimadores=estimadores,
                estrategia="mediana"
            )

        # 5. Média Aparada (Trimmed Mean)
        if tecnica_norm == "media_aparada":
            return RegressorVotacao(
                estimadores=estimadores,
                estrategia="media_aparada"
            )

        # 6. Média Simples (Ponderação uniforme)
        return RegressorVotacao(
            estimadores=estimadores,
            estrategia="media_simples"
        )
