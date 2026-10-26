"""Testes unitários para a Cadeia de Responsabilidade (Chain of Responsibility) em executar_pipeline."""

from __future__ import annotations

from typing import override
from unittest.mock import MagicMock, patch

import pytest

from app.conf_yaml import ConfYAML
from app.executar_pipeline import (
    ConstrutorCadeiaPipeline,
    ContextoPipeline,
    EloAvaliacaoSelecao,
    EloCadeiaPipeline,
    EloColetaDados,
    EloImplantacaoRegras,
    EloModelagemRegressao,
    EloObservabilidadeDrift,
    EloPreparacaoDados,
    EloAnaliseEda,
)


class EloMockado(EloCadeiaPipeline):
    """Elo de teste para validação isolada do comportamento da cadeia."""

    def __init__(self, numero: str, titulo: str) -> None:
        super().__init__()
        self._numero = numero
        self._titulo = titulo
        self.foi_executado: bool = False

    @property
    @override
    def numero_etapa(self) -> str:
        return self._numero

    @property
    @override
    def titulo_etapa(self) -> str:
        return self._titulo

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        self.foi_executado = True


@pytest.fixture
def conf_mockada() -> ConfYAML:
    """Fixture com mock de configuração YAML para testes rápidos de cadeia."""
    return MagicMock(spec=ConfYAML)


def test_cadeia_execucao_completa_todas_etapas(conf_mockada: ConfYAML) -> None:
    """Verifica se todos os elos conectados são executados quando etapa_alvo='todas'."""
    elo1 = EloMockado("1", "Etapa 1")
    elo2 = EloMockado("2", "Etapa 2")
    elo3 = EloMockado("3", "Etapa 3")

    elo1.definir_proximo(elo2).definir_proximo(elo3)

    contexto = ContextoPipeline(conf=conf_mockada, etapa_alvo="todas")
    resultado = elo1.processar(contexto)

    assert elo1.foi_executado is True
    assert elo2.foi_executado is True
    assert elo3.foi_executado is True
    assert resultado.etapas_executadas == ["1", "2", "3"]
    assert "1" in resultado.tempos_etapas
    assert "2" in resultado.tempos_etapas
    assert "3" in resultado.tempos_etapas


def test_cadeia_execucao_etapa_individual_intermediaria(conf_mockada: ConfYAML) -> None:
    """Verifica se a cadeia salta etapas anteriores e interrompe após executar a etapa alvo."""
    elo1 = EloMockado("1", "Etapa 1")
    elo2 = EloMockado("2", "Etapa 2")
    elo3 = EloMockado("3", "Etapa 3")

    elo1.definir_proximo(elo2).definir_proximo(elo3)

    contexto = ContextoPipeline(conf=conf_mockada, etapa_alvo="2")
    resultado = elo1.processar(contexto)

    assert elo1.foi_executado is False
    assert elo2.foi_executado is True
    assert elo3.foi_executado is False
    assert resultado.etapas_executadas == ["2"]
    assert "2" in resultado.tempos_etapas
    assert "1" not in resultado.tempos_etapas
    assert "3" not in resultado.tempos_etapas


def test_construtor_cadeia_completa_ordena_elos_1_a_7() -> None:
    """Verifica se ConstrutorCadeiaPipeline encadeia corretamente todas as 7 etapas na ordem exata."""
    cabeca = ConstrutorCadeiaPipeline.construir_cadeia_completa()

    elos: list[EloCadeiaPipeline] = []
    atual: EloCadeiaPipeline | None = cabeca
    while atual is not None:
        elos.append(atual)
        atual = atual._proximo  # pyright: ignore[reportPrivateUsage]

    assert len(elos) == 7
    assert isinstance(elos[0], EloColetaDados)
    assert elos[0].numero_etapa == "1"

    assert isinstance(elos[1], EloPreparacaoDados)
    assert elos[1].numero_etapa == "2"

    assert isinstance(elos[2], EloAnaliseEda)
    assert elos[2].numero_etapa == "3"

    assert isinstance(elos[3], EloModelagemRegressao)
    assert elos[3].numero_etapa == "4"

    assert isinstance(elos[4], EloAvaliacaoSelecao)
    assert elos[4].numero_etapa == "5"

    assert isinstance(elos[5], EloImplantacaoRegras)
    assert elos[5].numero_etapa == "6"

    assert isinstance(elos[6], EloObservabilidadeDrift)
    assert elos[6].numero_etapa == "7"


def test_elo_coleta_dados_executa_coletor(conf_mockada: ConfYAML) -> None:
    """Verifica se EloColetaDados invoca o serviço ColetorDados e armazena resultado no contexto."""
    elo = EloColetaDados()
    contexto = ContextoPipeline(conf=conf_mockada, etapa_alvo="1")

    with patch("app.executar_pipeline.ColetorDados.da_configuracao") as mock_fabrica:
        mock_coletor = MagicMock()
        mock_resultado = MagicMock()
        mock_resultado.total_registros = 100
        mock_resultado.total_colunas = 10
        mock_resultado.tabela_destino = "dados_staging"
        mock_resultado.tempo_execucao_segundos = 0.5
        mock_coletor.executar_coleta.return_value = mock_resultado
        mock_fabrica.return_value = mock_coletor

        elo.processar(contexto)

        mock_fabrica.assert_called_once_with(conf_mockada)
        mock_coletor.executar_coleta.assert_called_once()
        assert contexto.resultado_coleta == mock_resultado
        assert contexto.etapas_executadas == ["1"]
