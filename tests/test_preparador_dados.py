"""Testes automatizados de integração para o serviço PreparadorDados."""

from __future__ import annotations

import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.motores_tabela.motor_pandas import MotorPandas
from app.motores_tabela.motor_polars import MotorPolars
from app.repositorios_dados.fabrica_repositorios import FabricaRepositorios
from app.servicos_preparacao.preparador_dados import (
    ErroPreparacaoDados,
    PreparadorDados,
)


@pytest.fixture
def dados_brutos() -> pd.DataFrame:
    """Retorna dados simulados brutos para pipeline de preparação."""
    return pd.DataFrame(
        {
            "Codigo_Imovel": ["APT-01", "APT-02", "APT-03", "APT-04"],
            "Bairro": ["Centro", "Jardim Botânico", "Vila Ana Maria", "Centro"],
            "Zona": ["Centro", "Sul", "Sul", "Centro"],
            "Metragem": [55.0, 85.0, 140.0, -10.0],  # Registro 4 com metragem negativa
            "Quartos": [2, 3, 4, 1],
            "Banheiros": [1, 2, 3, 1],
            "Vagas": [1, 2, 2, 1],
            "Valor_Condominio": [300.0, 650.0, 1100.0, 200.0],
            "Valor_Iptu": [150.0, 320.0, 800.0, 100.0],
            "Area_Privativa": [50.0, 80.0, 135.0, 45.0],
            "Valor_da_Venda": [280000, 520000, 950000, 0],  # Registro 4 com valor 0
        }
    )


def test_preparador_dados_fluxo_completo_pandas(dados_brutos: pd.DataFrame) -> None:
    """Valida o pipeline de preparação completo executado pelo motor Pandas."""
    conf = ConfYAML()
    preparador = PreparadorDados.da_configuracao(conf, motor_sobrescrito=MotorPandas())

    df_preparado, resultado = preparador.preparar(dados_brutos)

    # 4 linhas originais, 2 removidas pelos filtros (metragem > 0 e preco > 0)
    assert resultado.total_linhas_originais == 4
    assert resultado.total_linhas_filtradas == 3
    assert resultado.linhas_removidas_filtros == 1
    assert resultado.motor_utilizado == "pandas"
    assert "Valor_da_Venda" in df_preparado.columns
    assert (df_preparado["Metragem"] > 0).all()
    assert (df_preparado["Valor_da_Venda"] > 0).all()


def test_preparador_dados_fluxo_completo_polars(dados_brutos: pd.DataFrame) -> None:
    """Valida o pipeline de preparação completo executado pelo motor Polars."""
    conf = ConfYAML()
    preparador = PreparadorDados.da_configuracao(conf, motor_sobrescrito=MotorPolars())

    df_preparado, resultado = preparador.preparar(dados_brutos)

    assert resultado.total_linhas_originais == 4
    assert resultado.total_linhas_filtradas == 3
    assert resultado.motor_utilizado == "polars"
    assert "Valor_da_Venda" in df_preparado.columns
    assert (df_preparado["Metragem"] > 0).all()


def test_equivalencia_pandas_e_polars(dados_brutos: pd.DataFrame) -> None:
    """Testa que os resultados produzidos por Pandas e Polars são rigorosamente equivalentes."""
    conf = ConfYAML()
    preparador_pd = PreparadorDados.da_configuracao(conf, motor_sobrescrito=MotorPandas())
    preparador_pl = PreparadorDados.da_configuracao(conf, motor_sobrescrito=MotorPolars())

    df_pd, res_pd = preparador_pd.preparar(dados_brutos)
    df_pl, res_pl = preparador_pl.preparar(dados_brutos)

    assert res_pd.total_linhas_filtradas == res_pl.total_linhas_filtradas
    assert res_pd.total_colunas_finais == res_pl.total_colunas_finais
    assert res_pd.colunas_finais == res_pl.colunas_finais

    # Equivalência numérica e categórica dos valores resultantes
    pd.testing.assert_frame_equal(df_pd, df_pl, check_dtype=False)


def test_preparador_dados_filtro_elimina_tudo(dados_brutos: pd.DataFrame) -> None:
    """Deve levantar ErroPreparacaoDados se os filtros eliminarem todos os registros."""
    conf = ConfYAML()
    preparador = PreparadorDados.da_configuracao(conf)

    # DataFrame onde todas as metragens são negativas
    dados_invalidos = dados_brutos.copy()
    dados_invalidos["Metragem"] = -10.0

    with pytest.raises(ErroPreparacaoDados, match="Nenhum registro restante após a aplicação dos filtros"):
        preparador.preparar(dados_invalidos)


def test_preparador_dados_com_staging_real() -> None:
    """Valida a integração da Etapa 2 lendo dados reais a partir do Staging SQLite da Etapa 1."""
    conf = ConfYAML()
    repo_staging = FabricaRepositorios.criar_repositorio(conf.obter_staging())

    if not repo_staging.verificar_existencia():
        pytest.skip("Staging SQLite não materializado.")

    dados_staging = repo_staging.ler_dados()
    assert len(dados_staging) > 0

    preparador = PreparadorDados.da_configuracao(conf)
    df_preparado, resultado = preparador.preparar(dados_staging)

    assert resultado.total_linhas_originais == len(dados_staging)
    assert resultado.total_linhas_filtradas > 0
    assert "Valor_da_Venda" in df_preparado.columns
    assert resultado.tempo_execucao_segundos >= 0.0
