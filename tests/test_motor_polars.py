"""Testes automatizados para o motor tabular MotorPolars e FabricaMotores."""

from __future__ import annotations

import pandas as pd
import pytest

from app.contratos_dados.motor_tabela import MotorTabela
from app.esquema_configuracao import ConfiguracaoTabular
from app.motores_tabela.fabrica_motores import ErroFabricaMotores, FabricaMotores
from app.motores_tabela.motor_pandas import MotorPandas
from app.motores_tabela.motor_polars import ErroMotorPolars, MotorPolars


@pytest.fixture
def dados_exemplo() -> pd.DataFrame:
    """Retorna um DataFrame de exemplo para testes de manipulação com Polars."""
    return pd.DataFrame(
        {
            "Valor_da_Venda": [200000, 350000, 500000, 0],
            "Metragem": [45.0, 70.0, 120.0, -5.0],
            "Quartos": [1, 2, 3, 1],
            "Banheiros": [1, 2, 2, 1],
            "Bairro": ["Centro", "Jardim Botânico", "Vila Ana Maria", "Centro"],
        }
    )


def test_motor_polars_conformidade_protocolo() -> None:
    """Garante que MotorPolars implementa o contrato MotorTabela."""
    motor = MotorPolars()
    assert isinstance(motor, MotorTabela)
    assert motor.obter_nome_motor() == "polars"


def test_motor_polars_filtrar_dados(dados_exemplo: pd.DataFrame) -> None:
    """Valida a filtragem de linhas com expressões no Polars."""
    motor = MotorPolars()

    # Sintaxe indexada df['coluna']
    df_filtrado_1 = motor.filtrar_dados(dados_exemplo, "df['Metragem'] > 50")
    assert len(df_filtrado_1) == 2
    assert df_filtrado_1["Metragem"].tolist() == [70.0, 120.0]

    # Sintaxe nativa pl.col('coluna')
    df_filtrado_2 = motor.filtrar_dados(dados_exemplo, "pl.col('Metragem') > 50")
    assert len(df_filtrado_2) == 2
    assert df_filtrado_2["Metragem"].tolist() == [70.0, 120.0]

    # Filtro vazio
    df_copia = motor.filtrar_dados(dados_exemplo, "")
    assert len(df_copia) == len(dados_exemplo)


def test_motor_polars_transformar_dados(dados_exemplo: pd.DataFrame) -> None:
    """Valida transformações e criação de colunas no Polars."""
    motor = MotorPolars()

    # Operação aritmética
    df_novo = motor.transformar_dados(
        dados_exemplo, "df['Banheiros'] / df['Quartos']", destino="Razao_Banheiro_Quarto"
    )
    assert "Razao_Banheiro_Quarto" in df_novo.columns
    assert df_novo["Razao_Banheiro_Quarto"].tolist() == [1.0, 1.0, 2 / 3, 1.0]

    # Método permitido: clip
    df_clip = motor.transformar_dados(
        dados_exemplo, "df['Metragem'].clip(lower=0)", destino="Metragem_Clip"
    )
    assert "Metragem_Clip" in df_clip.columns
    assert (df_clip["Metragem_Clip"] >= 0).all()


def test_motor_polars_selecionar_colunas(dados_exemplo: pd.DataFrame) -> None:
    """Valida projeção de colunas no Polars."""
    motor = MotorPolars()
    colunas_desejadas = ["Valor_da_Venda", "Bairro"]
    df_selecionado = motor.selecionar_colunas(dados_exemplo, colunas_desejadas)

    assert list(df_selecionado.columns) == colunas_desejadas
    assert len(df_selecionado) == len(dados_exemplo)


def test_motor_polars_erro_coluna_inexistente(dados_exemplo: pd.DataFrame) -> None:
    """Deve levantar ErroMotorPolars para colunas inexistentes."""
    motor = MotorPolars()

    with pytest.raises(ErroMotorPolars, match="inexistente"):
        motor.filtrar_dados(dados_exemplo, "df['Coluna_Fantasma'] > 0")

    with pytest.raises(ErroMotorPolars, match="não encontradas no DataFrame Polars"):
        motor.selecionar_colunas(dados_exemplo, ["Coluna_Fantasma"])


def test_motor_polars_rejeicao_expressao_maliciosa(dados_exemplo: pd.DataFrame) -> None:
    """Deve rejeitar código malicioso no Polars."""
    motor = MotorPolars()

    with pytest.raises(ErroMotorPolars):
        motor.filtrar_dados(dados_exemplo, "__import__('os').system('ls')")

    with pytest.raises(ErroMotorPolars):
        motor.transformar_dados(dados_exemplo, "eval('df.shape')", destino="teste")


def test_fabrica_motores_sucesso() -> None:
    """Valida a criação desacoplada de motores via FabricaMotores."""
    motor_pd = FabricaMotores.criar_motor(ConfiguracaoTabular(motor="pandas"))
    assert isinstance(motor_pd, MotorPandas)

    motor_pl = FabricaMotores.criar_motor(ConfiguracaoTabular(motor="polars"))
    assert isinstance(motor_pl, MotorPolars)


def test_fabrica_motores_invalido() -> None:
    """Deve levantar ErroFabricaMotores para motores não suportados."""
    config = ConfiguracaoTabular.model_construct(motor="spark")
    with pytest.raises(ErroFabricaMotores, match="Motor tabular não suportado"):
        FabricaMotores.criar_motor(config)
