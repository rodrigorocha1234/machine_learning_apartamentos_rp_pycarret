"""Testes automatizados para o motor tabular MotorPandas."""

from __future__ import annotations

import pandas as pd
import pytest

from app.contratos_dados.motor_tabela import MotorTabela
from app.motores_tabela.motor_pandas import ErroMotorPandas, MotorPandas


@pytest.fixture
def dados_exemplo() -> pd.DataFrame:
    """Retorna um DataFrame de exemplo para testes de manipulação tabular."""
    return pd.DataFrame(
        {
            "Valor_da_Venda": [200000, 350000, 500000, 0],
            "Metragem": [45.0, 70.0, 120.0, -5.0],
            "Quartos": [1, 2, 3, 1],
            "Banheiros": [1, 2, 2, 1],
            "Bairro": ["Centro", "Jardim Botânico", "Vila Ana Maria", "Centro"],
        }
    )


def test_motor_pandas_conformidade_protocolo() -> None:
    """Garante que MotorPandas implementa o contrato MotorTabela."""
    motor = MotorPandas()
    assert isinstance(motor, MotorTabela)
    assert motor.obter_nome_motor() == "pandas"


def test_motor_pandas_filtrar_dados(dados_exemplo: pd.DataFrame) -> None:
    """Valida a filtragem de linhas com expressões válidas."""
    motor = MotorPandas()

    df_filtrado = motor.filtrar_dados(dados_exemplo, "df['Metragem'] > 50")
    assert len(df_filtrado) == 2
    assert df_filtrado["Metragem"].tolist() == [70.0, 120.0]

    # Teste de filtro vazio
    df_copia = motor.filtrar_dados(dados_exemplo, "")
    assert len(df_copia) == len(dados_exemplo)


def test_motor_pandas_transformar_dados(dados_exemplo: pd.DataFrame) -> None:
    """Valida a aplicação de transformações com expressões e chamadas de método."""
    motor = MotorPandas()

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


def test_motor_pandas_selecionar_colunas(dados_exemplo: pd.DataFrame) -> None:
    """Valida a projeção e reordenação de colunas."""
    motor = MotorPandas()
    colunas_desejadas = ["Valor_da_Venda", "Bairro"]
    df_selecionado = motor.selecionar_colunas(dados_exemplo, colunas_desejadas)

    assert list(df_selecionado.columns) == colunas_desejadas
    assert len(df_selecionado) == len(dados_exemplo)


def test_motor_pandas_erro_coluna_inexistente(dados_exemplo: pd.DataFrame) -> None:
    """Deve levantar ErroMotorPandas quando uma coluna referenciada não existir."""
    motor = MotorPandas()

    with pytest.raises(ErroMotorPandas, match="não encontrada"):
        motor.filtrar_dados(dados_exemplo, "df['Coluna_Fantasma'] > 0")

    with pytest.raises(ErroMotorPandas, match="não encontradas no DataFrame"):
        motor.selecionar_colunas(dados_exemplo, ["Coluna_Fantasma"])


def test_motor_pandas_rejeicao_expressao_maliciosa(dados_exemplo: pd.DataFrame) -> None:
    """Deve rejeitar tentativas de execução de código inseguro."""
    motor = MotorPandas()

    with pytest.raises(ErroMotorPandas):
        motor.filtrar_dados(dados_exemplo, "__import__('os').system('ls')")

    with pytest.raises(ErroMotorPandas):
        motor.transformar_dados(dados_exemplo, "eval('df.shape')", destino="teste")
