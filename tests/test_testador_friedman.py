"""Testes automatizados para TestadorFriedman (Friedman e pós-teste de Nemenyi)."""

from __future__ import annotations

import pytest

from app.contratos_avaliacao.avaliador_estatistico import ResultadoComparacaoEstatistica
from app.servicos_avaliacao.testador_friedman import ErroTestadorFriedman, TestadorFriedman


def test_friedman_validacao_minimo_modelos() -> None:
    testador = TestadorFriedman()
    with pytest.raises(ErroTestadorFriedman, match="pelo menos 2 modelos"):
        testador.comparar_modelos({"modelo_a": [1.0, 2.0, 3.0]})


def test_friedman_validacao_tamanhos_desiguais() -> None:
    testador = TestadorFriedman()
    with pytest.raises(ErroTestadorFriedman, match="mesmo número de dobras pareadas"):
        testador.comparar_modelos(
            {
                "modelo_a": [1.0, 2.0, 3.0],
                "modelo_b": [1.0, 2.0],
            }
        )


def test_friedman_e_nemenyi_com_diferenca_significativa() -> None:
    testador = TestadorFriedman()
    # Modelo A sistematicamente muito melhor que Modelo B e C em 10 dobras
    matriz_cv: dict[str, list[float]] = {
        "modelo_campeao": [10.0, 11.0, 10.5, 9.8, 10.2, 11.1, 10.0, 10.4, 10.1, 9.9],
        "modelo_medio": [20.0, 21.0, 20.5, 19.8, 20.2, 21.1, 20.0, 20.4, 20.1, 19.9],
        "modelo_pior": [50.0, 52.0, 51.0, 49.0, 50.5, 51.5, 50.0, 52.0, 51.0, 49.5],
    }

    resultado = testador.comparar_modelos(matriz_cv, metrica="RMSE", alpha=0.05)

    assert isinstance(resultado, ResultadoComparacaoEstatistica)
    assert resultado.friedman.hipotese_nula_rejeitada is True
    assert resultado.friedman.diferenca_significativa is True
    assert resultado.friedman.p_valor < 0.05
    assert resultado.modelo_campeao_estatistico == "modelo_campeao"
    assert resultado.friedman.ranks_medios["modelo_campeao"] == 1.0

    # Nemenyi deve ser acionado
    assert resultado.nemenyi is not None
    assert resultado.nemenyi.diferenca_critica > 0.0
    assert "modelo_campeao" in resultado.candidatos_selecionados
    assert len(resultado.nemenyi.comparacoes) == 3


def test_friedman_sem_diferenca_significativa() -> None:
    testador = TestadorFriedman()
    # Modelos com desempenho alternado e idêntico em média
    matriz_cv: dict[str, list[float]] = {
        "modelo_1": [10.0, 12.0, 11.0, 10.0],
        "modelo_2": [12.0, 10.0, 10.0, 11.0],
        "modelo_3": [11.0, 11.0, 12.0, 10.0],
    }

    resultado = testador.comparar_modelos(matriz_cv, metrica="RMSE", alpha=0.01)

    assert isinstance(resultado, ResultadoComparacaoEstatistica)
    assert resultado.total_amostras_folds == 4
    # Quando não há diferença suficiente, todos os modelos são candidatos
    assert len(resultado.candidatos_selecionados) == 3
    assert "não fornecem evidência estatística suficiente" in resultado.justificativa_decisao
