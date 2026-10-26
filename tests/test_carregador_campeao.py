"""Testes para serialização, carregamento e execução do estimador campeão."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from app.contratos_implantacao.provedor_campeao import MetadadosModelo
from app.servicos_implantacao.carregador_campeao import CarregadorCampeao, ErroCarregadorCampeao


def test_salvar_e_carregar_modelo_campeao(tmp_path: Path) -> None:
    # 1. Cria e treina um estimador simples
    X = pd.DataFrame({"Metragem": [50.0, 70.0, 100.0]})
    y = np.array([250_000.0, 350_000.0, 500_000.0])
    reg = LinearRegression()
    reg.fit(X, y)

    metadados = MetadadosModelo(
        nome_modelo="regressao_linear",
        versao="1.0.0-teste",
        algoritmo="LinearRegression",
        transformacao_alvo="nenhuma",
        rmse_cv=1000.0,
        rmse_holdout=1200.0,
        r2_holdout=0.98,
        data_promocao="2026-10-26T00:00:00Z",
    )

    carregador = CarregadorCampeao.salvar_modelo_e_metadados(
        estimador=reg,
        metadados=metadados,
        diretorio_destino=tmp_path,
    )

    # 2. Testa predição
    df_novo = pd.DataFrame({"Metragem": [80.0]})
    predicoes = carregador.prever(df_novo)

    assert len(predicoes) == 1
    assert round(predicoes[0]) == 400_000.0

    # 3. Metadados
    meta = carregador.obter_metadados()
    assert meta.nome_modelo == "regressao_linear"
    assert meta.r2_holdout == 0.98


def test_erro_carregar_arquivo_inexistente(tmp_path: Path) -> None:
    with pytest.raises(ErroCarregadorCampeao, match="não encontrado"):
        CarregadorCampeao(
            caminho_modelo=tmp_path / "inexistente.joblib",
            caminho_metadados=tmp_path / "inexistente.json",
        )
