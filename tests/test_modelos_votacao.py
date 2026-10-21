import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from app.modelos_votacao.fabrica_votacao import ErroFabricaVotacao, FabricaVotacao
from app.modelos_votacao.regressor_votacao import ErroRegressorVotacao, RegressorVotacao

class EstimadorFalso:
    def __init__(self, predicoes: np.ndarray) -> None:
        self._predicoes = predicoes

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._predicoes

def test_regressor_votacao_media_simples() -> None:
    p1 = np.array([100.0, 200.0])
    p2 = np.array([120.0, 180.0])
    estimadores = [("m1", EstimadorFalso(p1)), ("m2", EstimadorFalso(p2))]

    votador = RegressorVotacao(estimadores=estimadores, estrategia="media_simples")
    pred = votador.predict(np.zeros((2, 1)))

    np.testing.assert_allclose(pred, np.array([110.0, 190.0]))

def test_regressor_votacao_mediana_robusta() -> None:
    # 3 modelos, com o 3º sendo um outlier na primeira amostra (300.0)
    p1 = np.array([100.0, 200.0])
    p2 = np.array([110.0, 210.0])
    p3 = np.array([300.0, 190.0])
    estimadores = [("m1", EstimadorFalso(p1)), ("m2", EstimadorFalso(p2)), ("m3", EstimadorFalso(p3))]

    votador = RegressorVotacao(estimadores=estimadores, estrategia="mediana")
    pred = votador.predict(np.zeros((2, 1)))

    np.testing.assert_allclose(pred, np.array([110.0, 200.0]))

def test_regressor_votacao_media_aparada() -> None:
    # Ordena: 100, 110, 300 -> descarta 100 e 300 -> sobra 110
    p1 = np.array([100.0, 200.0])
    p2 = np.array([110.0, 250.0])
    p3 = np.array([300.0, 150.0])
    estimadores = [("m1", EstimadorFalso(p1)), ("m2", EstimadorFalso(p2)), ("m3", EstimadorFalso(p3))]

    votador = RegressorVotacao(estimadores=estimadores, estrategia="media_aparada")
    pred = votador.predict(np.zeros((2, 1)))

    np.testing.assert_allclose(pred, np.array([110.0, 200.0]))

def test_regressor_votacao_media_ponderada_rmse() -> None:
    p1 = np.array([100.0, 200.0])
    p2 = np.array([200.0, 300.0])
    estimadores = [("m1", EstimadorFalso(p1)), ("m2", EstimadorFalso(p2))]

    # m1 tem RMSE=10 (peso maior), m2 tem RMSE=40
    ensemble = FabricaVotacao.criar_ensemble_votacao(
        estimadores=estimadores,
        tecnica="media_ponderada_rmse",
        metricas_rmse={"m1": 10.0, "m2": 40.0}
    )
    pred = ensemble.predict(np.zeros((2, 1)))
    # Peso m1 = 0.8, Peso m2 = 0.2 -> amostra 0: 0.8*100 + 0.2*200 = 120.0
    assert abs(pred[0] - 120.0) < 1e-4

def test_regressor_votacao_media_ponderada_postos() -> None:
    p1 = np.array([100.0, 200.0])
    p2 = np.array([200.0, 300.0])
    estimadores = [("m1", EstimadorFalso(p1)), ("m2", EstimadorFalso(p2))]

    # m1 posto 1 (peso 2), m2 posto 2 (peso 1) -> 2/3 e 1/3
    ensemble = FabricaVotacao.criar_ensemble_votacao(
        estimadores=estimadores,
        tecnica="media_ponderada_postos",
        postos_friedman={"m1": 1.0, "m2": 2.0}
    )
    pred = ensemble.predict(np.zeros((2, 1)))
    esperado = (2.0 / 3.0) * 100.0 + (1.0 / 3.0) * 200.0
    assert abs(pred[0] - esperado) < 1e-4

def test_fabrica_votacao_stacking() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    m1 = LinearRegression()
    m2 = Ridge(alpha=1.0)
    estimadores = [("m1", m1), ("m2", m2)]

    stacking = FabricaVotacao.criar_ensemble_votacao(
        estimadores=estimadores,
        tecnica="stacking",
        cv_folds=2
    )
    stacking.fit(X, y)
    pred = stacking.predict(X)
    assert len(pred) == 6

def test_fabrica_votacao_voting_regressor_nativo() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    m1 = LinearRegression()
    m2 = Ridge(alpha=1.0)
    estimadores = [("m1", m1), ("m2", m2)]

    votador = FabricaVotacao.criar_ensemble_votacao(
        estimadores=estimadores,
        tecnica="voting_regressor"
    )
    votador.fit(X, y)
    pred = votador.predict(X)
    assert len(pred) == 6

def test_fabrica_votacao_erros() -> None:
    with pytest.raises(ErroFabricaVotacao):
        FabricaVotacao.criar_ensemble_votacao([], tecnica="media_simples")

    with pytest.raises(ErroFabricaVotacao):
        FabricaVotacao.criar_ensemble_votacao([("m1", EstimadorFalso(np.array([1])))], tecnica="tecnica_invalida")
