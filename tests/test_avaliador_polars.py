"""Testes automatizados para o motor de cálculo de regras baseado em Polars."""

from __future__ import annotations

from typing import cast
import pandas as pd
import pytest

from app.esquema_configuracao import ConfiguracaoImplantacao, RegraNegocioItem
from app.motores_regras.avaliador_polars import AvaliadorPolars, ErroAvaliadorPolars


def test_calculo_regras_polars_registro() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="comissao",
                ativo=True,
                destino="comissao_imob",
                expressao="preco_predito * 0.06",
            ),
        ],
    )
    avaliador = AvaliadorPolars(config)
    contexto: dict[str, float | int | str] = {"preco_predito": 600_000.0}

    resultado = avaliador.avaliar_regras_registro(contexto)
    assert resultado["comissao_imob"] == 36_000.0


def test_calculo_regras_polars_tabela() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="metro_quadrado",
                ativo=True,
                destino="preco_m2",
                expressao="preco_predito / Metragem",
            ),
        ],
    )
    avaliador = AvaliadorPolars(config)
    df = pd.DataFrame(
        {
            "preco_predito": [300_000.0, 500_000.0],
            "Metragem": [60.0, 100.0],
        }
    )

    df_res = avaliador.avaliar_regras_tabela(df, coluna_predita="preco_predito")
    assert "preco_m2" in df_res.columns
    serie_m2 = cast(pd.Series, df_res["preco_m2"])
    val0 = cast(object, serie_m2.iloc[0])
    val1 = cast(object, serie_m2.iloc[1])
    assert float(str(val0)) == 5_000.0
    assert float(str(val1)) == 5_000.0


def test_erro_coluna_alvo_inexistente_polars() -> None:
    config = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="teste",
                ativo=True,
                destino="teste_res",
                expressao="preco_predito * 1.1",
            ),
        ],
    )
    avaliador = AvaliadorPolars(config)
    df = pd.DataFrame({"coluna_errada": [10.0]})

    with pytest.raises(ErroAvaliadorPolars, match="não encontrada na tabela"):
        avaliador.avaliar_regras_tabela(df, coluna_predita="preco_predito")
