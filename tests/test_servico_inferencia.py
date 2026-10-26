"""Testes para o serviço de inferência acoplado ao cálculo de regras dinâmicas."""

from __future__ import annotations

from typing import cast, final, override
import pandas as pd

from app.contratos_implantacao.avaliador_regras import RequisicaoImovel
from app.contratos_implantacao.provedor_campeao import MetadadosModelo, ProvedorCampeao
from app.esquema_configuracao import ConfiguracaoImplantacao, RegraNegocioItem
from app.motores_regras.avaliador_pandas import AvaliadorPandas
from app.servicos_implantacao.servico_inferencia import ServicoInferencia


@final
class ProvedorMock(ProvedorCampeao):
    """Mock determinístico para teste de inferência sem carregar modelo real."""

    @override
    def prever(self, dados: pd.DataFrame) -> list[float]:
        # Previsão sintética fixa para cada linha
        return [500_000.0 for _ in range(len(dados))]

    @override
    def obter_metadados(self) -> MetadadosModelo:
        return MetadadosModelo(
            nome_modelo="lightgbm_mock",
            versao="1.0.0-mock",
            algoritmo="MockLGBM",
            transformacao_alvo="nenhuma",
            rmse_cv=5000.0,
            rmse_holdout=6000.0,
            r2_holdout=0.96,
            data_promocao="2026-10-26T00:00:00Z",
        )


def test_fluxo_inferencia_alimenta_regras() -> None:
    config_imp = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="comissao",
                ativo=True,
                destino="comissao_imobiliaria",
                expressao="preco_predito * 0.06",
            ),
            RegraNegocioItem(
                nome="metro_quadrado",
                ativo=True,
                destino="preco_m2",
                expressao="preco_predito / Metragem",
            ),
        ],
    )
    provedor = ProvedorMock()
    avaliador = AvaliadorPandas(config_imp)
    servico = ServicoInferencia(provedor, avaliador, config_imp)

    requisicao = RequisicaoImovel(
        Metragem=100.0,
        Quartos=3,
        Banheiros=2,
        Vagas=2,
        Bairro="Irajá",
        Zona="Sul",
    )

    resposta = servico.predizer_imovel(requisicao)

    # 1. Preço previsto pelo modelo
    assert resposta.preco_predito == 500_000.0

    # 2. As regras de negócio foram alimentadas pelo valor predito
    assert resposta.regras_calculadas["comissao_imobiliaria"] == 30_000.0
    assert resposta.regras_calculadas["preco_m2"] == 5_000.0
    assert resposta.modelo_campeao == "lightgbm_mock"
    assert resposta.versao_modelo == "1.0.0-mock"
    assert len(resposta.detalhes_regras) == 2


def test_inferencia_em_lote() -> None:
    config_imp = ConfiguracaoImplantacao(
        ativo=True,
        coluna_alvo_predito="preco_predito",
        regras=[
            RegraNegocioItem(
                nome="faixa_min",
                ativo=True,
                destino="min_sugerido",
                expressao="preco_predito * 0.90",
            ),
        ],
    )
    provedor = ProvedorMock()
    avaliador = AvaliadorPandas(config_imp)
    servico = ServicoInferencia(provedor, avaliador, config_imp)

    df_lote = pd.DataFrame(
        [
            {"Metragem": 70.0, "Bairro": "Centro"},
            {"Metragem": 90.0, "Bairro": "Sul"},
        ]
    )

    df_res = servico.predizer_lote(df_lote)

    assert "preco_predito" in df_res.columns
    assert "min_sugerido" in df_res.columns
    serie_sugerido = cast(pd.Series, df_res["min_sugerido"])
    val0 = cast(object, serie_sugerido.iloc[0])
    val1 = cast(object, serie_sugerido.iloc[1])
    assert float(str(val0)) == 450_000.0
    assert float(str(val1)) == 450_000.0
