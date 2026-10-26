"""Testes automatizados para o avaliador de qualidade de dados."""

from __future__ import annotations

import pandas as pd

from app.detectores_drift.avaliador_qualidade import AvaliadorQualidade


def test_qualidade_sem_nulos_e_sem_anomalias() -> None:
    df_ref = pd.DataFrame(
        {
            "Metragem": [50.0, 80.0, 100.0],
            "Bairro": ["Centro", "Sul", "Centro"],
        }
    )
    df_prod = pd.DataFrame(
        {
            "Metragem": [60.0, 75.0],
            "Bairro": ["Sul", "Centro"],
        }
    )

    avaliador = AvaliadorQualidade()
    res = avaliador.avaliar_qualidade(df_ref, df_prod)

    assert res.total_amostras == 2
    assert res.taxa_nulos_geral == 0.0
    assert len(res.contagem_nulos_por_coluna) == 0
    assert len(res.novas_categorias_detectadas) == 0
    assert len(res.valores_fora_limites) == 0


def test_qualidade_com_nulos_e_novas_categorias() -> None:
    df_ref = pd.DataFrame(
        {
            "Metragem": [50.0, 100.0],
            "Bairro": ["Centro", "Sul"],
        }
    )
    df_prod = pd.DataFrame(
        {
            "Metragem": [40.0, None, 150.0],  # 40 e 150 fora dos limites [50, 100], 1 nulo
            "Bairro": ["Centro", "Novo Bairro", "Sul"],  # "Novo Bairro" é inédito
        }
    )

    avaliador = AvaliadorQualidade()
    res = avaliador.avaliar_qualidade(df_ref, df_prod)

    assert res.total_amostras == 3
    assert res.taxa_nulos_geral > 0.0
    assert res.contagem_nulos_por_coluna.get("Metragem") == 1
    assert "Bairro" in res.novas_categorias_detectadas
    assert "Novo Bairro" in res.novas_categorias_detectadas["Bairro"]
    assert res.valores_fora_limites.get("Metragem") == 2
