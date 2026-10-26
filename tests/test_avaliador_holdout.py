"""Testes automatizados para AvaliadorHoldout."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from app.servicos_avaliacao.avaliador_holdout import AvaliadorHoldout, ErroAvaliadorHoldout


def test_avaliador_holdout_validacoes() -> None:
    avaliador = AvaliadorHoldout(coluna_alvo="Preco")

    with pytest.raises(ErroAvaliadorHoldout, match="está vazio"):
        avaliador.avaliar_modelos_holdout({"m": Ridge()}, pd.DataFrame())

    df_sem_alvo = pd.DataFrame({"Area": [50.0, 60.0]})
    with pytest.raises(ErroAvaliadorHoldout, match="Coluna alvo 'Preco' ausente"):
        avaliador.avaliar_modelos_holdout({"m": Ridge()}, df_sem_alvo)


def test_avaliador_holdout_metricas_completas() -> None:
    avaliador = AvaliadorHoldout(coluna_alvo="Preco")
    df_hd = pd.DataFrame(
        {
            "Area": [50.0, 70.0, 100.0, 120.0],
            "Preco": [200000.0, 300000.0, 450000.0, 520000.0],
        }
    )

    modelo = Ridge().fit(df_hd[["Area"]], df_hd["Preco"])
    resultados = avaliador.avaliar_modelos_holdout({"ridge": modelo}, df_hd)

    assert "ridge" in resultados
    metr = resultados["ridge"]
    assert metr.nome_modelo == "ridge"
    assert metr.rmse >= 0.0
    assert metr.mae >= 0.0
    assert metr.r2 <= 1.0
    assert metr.mape >= 0.0
    assert metr.medae >= 0.0
