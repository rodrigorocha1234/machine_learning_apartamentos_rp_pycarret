"""Testes de integração de endpoints HTTP para a API FastAPI."""

from __future__ import annotations

from typing import final, override
from fastapi.testclient import TestClient
import pandas as pd

from app.contratos_implantacao.provedor_campeao import MetadadosModelo, ProvedorCampeao
from app.esquema_configuracao import (
    ConfiguracaoImplantacao,
    ConfiguracaoServing,
    RegraNegocioItem,
)
from app.motores_regras.avaliador_pandas import AvaliadorPandas
from app.servicos_implantacao.aplicacao_fastapi import criar_aplicacao_fastapi
from app.servicos_implantacao.servico_inferencia import ServicoInferencia


@final
class ProvedorMockFastapi(ProvedorCampeao):
    """Mock determinístico para teste das rotas FastAPI."""

    @override
    def prever(self, dados: pd.DataFrame) -> list[float]:
        return [400_000.0 for _ in range(len(dados))]

    @override
    def obter_metadados(self) -> MetadadosModelo:
        return MetadadosModelo(
            nome_modelo="lightgbm",
            versao="1.0.0-champion",
            algoritmo="LGBMRegressor",
            transformacao_alvo="log1p_embutida",
            rmse_cv=84599.21,
            rmse_holdout=84599.21,
            r2_holdout=0.9592,
            data_promocao="2026-10-26T00:00:00Z",
        )


def _obter_cliente_teste() -> TestClient:
    config_imp = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="margem_min",
                ativo=True,
                descricao="Mínimo",
                destino="preco_minimo",
                expressao="preco_predito * 0.95",
            ),
            RegraNegocioItem(
                nome="comissao",
                ativo=True,
                descricao="Comissão 6%",
                destino="comissao_imob",
                expressao="preco_predito * 0.06",
            ),
        ],
        serving=ConfiguracaoServing(host="0.0.0.0", porta=8080, endpoint="/invocations"),
    )
    provedor = ProvedorMockFastapi()
    avaliador = AvaliadorPandas(config_imp)
    servico = ServicoInferencia(provedor, avaliador, config_imp)
    app = criar_aplicacao_fastapi(servico, config_imp)
    return TestClient(app)


def test_rota_health() -> None:
    cliente = _obter_cliente_teste()
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status"] == "operacional"
    assert dados["modelo_campeao"] == "lightgbm"
    assert dados["versao_modelo"] == "1.0.0-champion"
    assert dados["total_regras_ativas"] == 2


def test_rota_predict_valida() -> None:
    cliente = _obter_cliente_teste()
    payload = {
        "Metragem": 90.0,
        "Quartos": 3,
        "Banheiros": 2,
        "Vagas": 2,
        "Bairro": "Jardim Botanico",
        "Zona": "Sul",
        "Valor_Condominio": 500.0,
        "Valor_Iptu": 800.0,
        "Area_Privativa": 85.0,
    }
    resposta = cliente.post("/predict", json=payload)
    assert resposta.status_code == 200
    dados = resposta.json()

    assert dados["preco_predito"] == 400_000.0
    assert dados["modelo_campeao"] == "lightgbm"
    assert dados["regras_calculadas"]["preco_minimo"] == 380_000.0
    assert dados["regras_calculadas"]["comissao_imob"] == 24_000.0
    assert len(dados["detalhes_regras"]) == 2


def test_rota_predict_dado_invalido() -> None:
    cliente = _obter_cliente_teste()
    # Metragem inválida (<= 0)
    payload = {
        "Metragem": 0.0,
        "Quartos": 2,
        "Banheiros": 1,
        "Vagas": 1,
        "Bairro": "Centro",
    }
    resposta = cliente.post("/predict", json=payload)
    assert resposta.status_code == 422


def test_rota_invocations_mlflow() -> None:
    cliente = _obter_cliente_teste()
    payload = {
        "instances": [
            {"Metragem": 75.0, "Quartos": 2, "Banheiros": 1, "Vagas": 1, "Bairro": "Centro"}
        ]
    }
    resposta = cliente.post("/invocations", json=payload)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert "predictions" in dados
    assert len(dados["predictions"]) == 1
    assert "preco_predito" in dados["predictions"][0]
    assert "preco_minimo" in dados["predictions"][0]
