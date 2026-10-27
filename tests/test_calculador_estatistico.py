"""Testes automatizados para o CalculadorEstatistico da Etapa 3 (EDA)."""

from __future__ import annotations

from typing import cast
import pandas as pd
import pytest

from app.enums_dominio import TipoGrafico
from app.esquema_configuracao import ItemAnaliseEda
from app.servicos_eda.calculador_estatistico import (
    CalculadorEstatistico,
    ErroCalculoEstatistico,
)


@pytest.fixture
def dados_teste() -> pd.DataFrame:
    """Gera DataFrame simulado com dados representativos de apartamentos."""
    return pd.DataFrame(
        {
            "Valor_da_Venda": [500000.0, 750000.0, 1000000.0, 1250000.0, 1500000.0],
            "Metragem": [50.0, 75.0, 100.0, 125.0, 150.0],
            "Quartos": [1, 2, 3, 3, 4],
            "Bairro": ["Centro", "Centro", "Jardim Botânico", "Jardim Botânico", "Zona Sul"],
            "Zona": ["Centro", "Centro", "Sul", "Sul", "Sul"],
        }
    )


def test_calculador_histograma(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="hist_preco",
        tipo_grafico=TipoGrafico.HISTOGRAMA,
        coluna="Valor_da_Venda",
        nbins=5,
    )
    resultado = calculador.calcular_analise(dados_teste, item)

    assert resultado.nome == "hist_preco"
    assert resultado.tipo_grafico == "histograma"
    assert resultado.coluna_principal == "Valor_da_Venda"
    assert "media" in resultado.metricas_resumo
    assert resultado.metricas_resumo["media"] == 1000000.0
    assert "frequencias" in resultado.dados_estruturados
    freqs = cast(list[int], resultado.dados_estruturados["frequencias"])
    assert sum(freqs) == 5


def test_calculador_histograma_com_expressao(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="hist_preco_m2",
        tipo_grafico=TipoGrafico.HISTOGRAMA,
        coluna="Preco_M2",
        expressao="df['Valor_da_Venda'] / df['Metragem']",
        nbins=5,
    )
    resultado = calculador.calcular_analise(dados_teste, item)

    assert resultado.nome == "hist_preco_m2"
    # Preco_M2 é exatamente 10.000 para todos os registros
    assert resultado.metricas_resumo["media"] == 10000.0
    assert resultado.metricas_resumo["desvio_padrao"] == 0.0


def test_calculador_boxplot(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="box_preco_bairro",
        tipo_grafico=TipoGrafico.BOXPLOT,
        coluna_x="Bairro",
        coluna_y="Valor_da_Venda",
    )
    resultado = calculador.calcular_analise(dados_teste, item)

    assert resultado.tipo_grafico == "boxplot"
    assert "Centro" in resultado.dados_estruturados["categorias"]  # pyright: ignore[reportOperatorIssue]
    assert resultado.metricas_resumo["total_categorias"] == 3.0


def test_calculador_dispersao(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="disp_metragem_preco",
        tipo_grafico=TipoGrafico.DISPERSAO,
        coluna_x="Metragem",
        coluna_y="Valor_da_Venda",
        cor="Quartos",
    )
    resultado = calculador.calcular_analise(dados_teste, item)

    assert resultado.tipo_grafico == "dispersao"
    assert resultado.metricas_resumo["correlacao_pearson"] == 1.0
    pontos = cast(list[object], resultado.dados_estruturados["pontos_x"])
    assert len(pontos) == 5


def test_calculador_correlacao(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="corr_matriz",
        tipo_grafico=TipoGrafico.CORRELACAO,
    )
    resultado = calculador.calcular_analise(dados_teste, item)

    assert resultado.tipo_grafico == "correlacao"
    assert "colunas" in resultado.dados_estruturados
    assert resultado.metricas_resumo["total_variaveis_analisadas"] >= 3.0


def test_calculador_tabela_resumo(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="tab_bairro",
        tipo_grafico=TipoGrafico.TABELA,
        agrupar_por="Bairro",
        coluna="Valor_da_Venda",
        ordenar_por="contagem",
        direcao_ordenacao="desc",
    )
    resultado = calculador.calcular_analise(dados_teste, item)

    assert resultado.tipo_grafico == "tabela"
    linhas = cast(list[object], resultado.dados_estruturados["linhas"])
    assert len(linhas) == 3


def test_calculador_coluna_inexistente(dados_teste: pd.DataFrame) -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="coluna_fantasma",
        tipo_grafico=TipoGrafico.HISTOGRAMA,
        coluna="Coluna_Inexistente",
    )
    with pytest.raises(ErroCalculoEstatistico, match="Coluna 'Coluna_Inexistente' não encontrada"):
        calculador.calcular_analise(dados_teste, item)


def test_calculador_dataframe_vazio() -> None:
    calculador = CalculadorEstatistico()
    item = ItemAnaliseEda(
        nome="hist_vazio",
        tipo_grafico=TipoGrafico.HISTOGRAMA,
        coluna="Valor_da_Venda",
    )
    with pytest.raises(ErroCalculoEstatistico, match="DataFrame vazio"):
        calculador.calcular_analise(pd.DataFrame(), item)
