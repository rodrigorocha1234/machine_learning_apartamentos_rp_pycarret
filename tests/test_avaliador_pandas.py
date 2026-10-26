"""Testes automatizados para o motor de cálculo de regras baseado em Pandas."""

from __future__ import annotations

from typing import cast
import pandas as pd
import pytest

from app.esquema_configuracao import ConfiguracaoImplantacao, RegraNegocioItem
from app.motores_regras.avaliador_pandas import AvaliadorPandas, ErroAvaliadorRegras


def test_calculo_regras_registro_individual() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="min_negociacao",
                ativo=True,
                destino="preco_min",
                expressao="preco_predito * 0.95",
            ),
            RegraNegocioItem(
                nome="max_negociacao",
                ativo=True,
                destino="preco_max",
                expressao="preco_predito * 1.05",
            ),
            RegraNegocioItem(
                nome="m2_predito",
                ativo=True,
                destino="preco_m2",
                expressao="preco_predito / Metragem",
            ),
        ],
    )
    avaliador = AvaliadorPandas(config)
    contexto: dict[str, float | int | str] = {
        "preco_predito": 500_000.0,
        "Metragem": 100.0,
    }

    resultado = avaliador.avaliar_regras_registro(contexto)

    assert resultado["preco_min"] == 475_000.0
    assert resultado["preco_max"] == 525_000.0
    assert resultado["preco_m2"] == 5_000.0


def test_regras_desativadas_sao_ignoradas() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="regra_ativa",
                ativo=True,
                destino="val_ativo",
                expressao="preco_predito * 0.10",
            ),
            RegraNegocioItem(
                nome="regra_inativa",
                ativo=False,
                destino="val_inativo",
                expressao="preco_predito * 0.50",
            ),
        ],
    )
    avaliador = AvaliadorPandas(config)
    contexto: dict[str, float | int | str] = {"preco_predito": 100_000.0}

    resultado = avaliador.avaliar_regras_registro(contexto)

    assert "val_ativo" in resultado
    assert resultado["val_ativo"] == 10_000.0
    assert "val_inativo" not in resultado


def test_divisao_por_zero_nao_quebra_execucao() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="divisao_por_zero",
                ativo=True,
                destino="resultado_div",
                expressao="preco_predito / Metragem",
            ),
        ],
    )
    avaliador = AvaliadorPandas(config)
    contexto: dict[str, float | int | str] = {
        "preco_predito": 300_000.0,
        "Metragem": 0.0,
    }

    resultado = avaliador.avaliar_regras_registro(contexto)
    assert resultado["resultado_div"] == 0.0


def test_rejeicao_expressao_insegura_no_init() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        regras=[
            RegraNegocioItem(
                nome="maliciosa",
                ativo=True,
                destino="saida",
                expressao="__import__('os').system('ls')",
            ),
        ],
    )
    with pytest.raises(ErroAvaliadorRegras, match="Expressão insegura"):
        AvaliadorPandas(config)


def test_calculo_regras_em_tabela_pandas() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="margem_95",
                ativo=True,
                destino="preco_min",
                expressao="preco_predito * 0.95",
            ),
        ],
    )
    avaliador = AvaliadorPandas(config)
    df = pd.DataFrame(
        {
            "preco_predito": [200_000.0, 400_000.0],
            "Metragem": [50.0, 80.0],
        }
    )

    df_resultado = avaliador.avaliar_regras_tabela(df, coluna_predita="preco_predito")

    assert "preco_min" in df_resultado.columns
    serie_min = cast(pd.Series, df_resultado["preco_min"])
    val0 = cast(object, serie_min.iloc[0])
    val1 = cast(object, serie_min.iloc[1])
    assert float(str(val0)) == 190_000.0
    assert float(str(val1)) == 380_000.0
