"""Testes automatizados para ComiteSklearn (técnicas de ensemble)."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from app.comites_votacao.comite_sklearn import ComiteSklearn
from app.contratos_avaliacao.construtor_comite import ResultadoComite
from app.esquema_configuracao import ConfiguracaoVotacao


@pytest.fixture
def dados_ensemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tr = pd.DataFrame(
        {
            "Area": [40.0, 60.0, 80.0, 100.0],
            "Alvo": [100.0, 150.0, 200.0, 250.0],
        }
    )
    df_hd = pd.DataFrame(
        {
            "Area": [50.0, 70.0],
            "Alvo": [125.0, 175.0],
        }
    )
    return df_tr, df_hd


def test_comite_desativado(dados_ensemble: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_tr, df_hd = dados_ensemble
    comite = ComiteSklearn()
    cfg_votacao = ConfiguracaoVotacao(ativo=False)

    res = comite.construir_e_avaliar({}, df_tr, df_hd, "Alvo", cfg_votacao)
    assert isinstance(res, ResultadoComite)
    assert res.ativo is False
    assert res.tecnica_utilizada == "desativada"


def test_comite_media_simples(dados_ensemble: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_tr, df_hd = dados_ensemble
    comite = ComiteSklearn()
    cfg_votacao = ConfiguracaoVotacao(ativo=True, tecnica="media_simples", melhores_n=2)

    mod_1 = Ridge().fit(df_tr[["Area"]], df_tr["Alvo"])
    mod_2 = Ridge(alpha=10.0).fit(df_tr[["Area"]], df_tr["Alvo"])

    res = comite.construir_e_avaliar(
        {"m1": mod_1, "m2": mod_2}, df_tr, df_hd, "Alvo", cfg_votacao
    )
    assert res.ativo is True
    assert res.tecnica_utilizada == "media_simples"
    assert len(res.modelos_membros) == 2
    assert "RMSE" in res.metricas_holdout
    assert res.metricas_holdout["RMSE"] >= 0.0


def test_comite_media_ponderada(dados_ensemble: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_tr, df_hd = dados_ensemble
    comite = ComiteSklearn()
    cfg_votacao = ConfiguracaoVotacao(ativo=True, tecnica="media_ponderada_rmse", melhores_n=2)

    mod_1 = Ridge().fit(df_tr[["Area"]], df_tr["Alvo"])
    mod_2 = Ridge(alpha=5.0).fit(df_tr[["Area"]], df_tr["Alvo"])

    res = comite.construir_e_avaliar(
        {"m1": mod_1, "m2": mod_2}, df_tr, df_hd, "Alvo", cfg_votacao
    )
    assert res.ativo is True
    assert res.tecnica_utilizada == "media_ponderada_rmse"
    assert sum(res.pesos.values()) == pytest.approx(1.0, abs=0.01)


def test_comite_voting_regressor(dados_ensemble: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_tr, df_hd = dados_ensemble
    comite = ComiteSklearn()
    cfg_votacao = ConfiguracaoVotacao(ativo=True, tecnica="voting_regressor", melhores_n=2)

    mod_1 = Ridge().fit(df_tr[["Area"]], df_tr["Alvo"])
    mod_2 = Ridge(alpha=2.0).fit(df_tr[["Area"]], df_tr["Alvo"])

    res = comite.construir_e_avaliar(
        {"m1": mod_1, "m2": mod_2}, df_tr, df_hd, "Alvo", cfg_votacao
    )
    assert res.ativo is True
    assert res.tecnica_utilizada == "voting_regressor"
    assert len(res.modelos_membros) == 2


def test_comite_stacking(dados_ensemble: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_tr, df_hd = dados_ensemble
    comite = ComiteSklearn()
    cfg_votacao = ConfiguracaoVotacao(
        ativo=True, tecnica="stacking", melhores_n=2, meta_modelo="ridge"
    )

    mod_1 = Ridge().fit(df_tr[["Area"]], df_tr["Alvo"])
    mod_2 = Ridge(alpha=2.0).fit(df_tr[["Area"]], df_tr["Alvo"])

    res = comite.construir_e_avaliar(
        {"m1": mod_1, "m2": mod_2}, df_tr, df_hd, "Alvo", cfg_votacao
    )
    assert res.ativo is True
    assert res.tecnica_utilizada == "stacking"
    assert len(res.modelos_membros) == 2
