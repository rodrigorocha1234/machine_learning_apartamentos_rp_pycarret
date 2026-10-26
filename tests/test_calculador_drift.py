"""Testes automatizados para detecção estatística de drift (KS, PSI, Taxonomia)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.detectores_drift.calculador_drift import CalculadorDrift
from app.esquema_configuracao import ConfiguracaoDrift


def test_ausencia_de_drift_em_distribuicoes_similares() -> None:
    rng = np.random.default_rng(42)
    # Distribuição normal similar
    ref_vals = rng.normal(loc=100.0, scale=10.0, size=200)
    prod_vals = rng.normal(loc=100.0, scale=10.0, size=100)

    df_ref = pd.DataFrame({"Metragem": ref_vals})
    df_prod = pd.DataFrame({"Metragem": prod_vals})

    config = ConfiguracaoDrift(ativo=True)
    detector = CalculadorDrift(config)

    relatorio = detector.analisar_drift(df_ref, df_prod)

    assert not relatorio.covariate_drift_detectado
    assert relatorio.classificacao_drift == "sem_drift"
    assert len(relatorio.variaveis_com_drift) == 0


def test_deteccao_covariate_drift_com_desvio_numerico() -> None:
    rng = np.random.default_rng(42)
    # Média deslocada de 100 para 160 (drift brusco)
    ref_vals = rng.normal(loc=100.0, scale=10.0, size=200)
    prod_vals = rng.normal(loc=160.0, scale=10.0, size=100)

    df_ref = pd.DataFrame({"Metragem": ref_vals})
    df_prod = pd.DataFrame({"Metragem": prod_vals})

    config = ConfiguracaoDrift(ativo=True)
    detector = CalculadorDrift(config)

    relatorio = detector.analisar_drift(df_ref, df_prod)

    assert relatorio.covariate_drift_detectado
    assert "Metragem" in relatorio.variaveis_com_drift
    assert relatorio.classificacao_drift == "virtual_drift"


def test_deteccao_real_concept_drift_com_degradacao_rmse() -> None:
    rng = np.random.default_rng(42)
    df_ref = pd.DataFrame({"Metragem": rng.normal(100.0, 10.0, 100)})
    df_prod = pd.DataFrame({"Metragem": rng.normal(100.0, 10.0, 100)})

    # Simulação de alvos e predições com degradação expressiva de erro (RMSE 20.000 vs 5.000 baseline = +300%)
    y_real = pd.Series([500_000.0, 600_000.0, 700_000.0])
    y_pred = pd.Series([450_000.0, 500_000.0, 550_000.0])

    config = ConfiguracaoDrift(ativo=True)
    detector = CalculadorDrift(config)

    relatorio = detector.analisar_drift(
        dados_referencia=df_ref,
        dados_producao=df_prod,
        y_producao_real=y_real,
        predicoes_producao=y_pred,
        rmse_baseline=10_000.0,
    )

    assert relatorio.concept_drift_detectado is True
    assert relatorio.classificacao_drift == "real_concept_drift"
    assert relatorio.degradacao_rmse_percentual is not None
    assert relatorio.degradacao_rmse_percentual > 15.0
