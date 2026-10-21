import pytest
import pandas as pd
from app.contratos_dados.motor_tabela import MotorTabela
from app.motores_tabela.motor_pandas import MotorPandas, ErroMotorTabela

def test_motor_pandas_satisfaz_protocolo() -> None:
    motor: MotorTabela = MotorPandas()
    assert isinstance(motor, MotorTabela)

def test_motor_pandas_filtrar_dados() -> None:
    dados = pd.DataFrame({"Metragem": [50.0, -10.0, 120.0, 0.0]})
    motor = MotorPandas()
    resultado = motor.filtrar_dados(dados, "Metragem > 0")
    assert len(resultado) == 2
    assert list(resultado["Metragem"]) == [50.0, 120.0]

def test_motor_pandas_transformar_dados() -> None:
    dados = pd.DataFrame({"Metragem": [10.0, 20.0]})
    motor = MotorPandas()
    resultado = motor.transformar_dados(dados, "Metragem * Metragem", "Metragem_Quadrada")
    assert "Metragem_Quadrada" in resultado.columns
    assert list(resultado["Metragem_Quadrada"]) == [100.0, 400.0]

def test_motor_pandas_selecionar_colunas() -> None:
    dados = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    motor = MotorPandas()
    resultado = motor.selecionar_colunas(dados, ["A", "C"])
    assert list(resultado.columns) == ["A", "C"]


def test_motor_pandas_rejeita_codigo_perigoso() -> None:
    dados = pd.DataFrame({"Metragem": [50.0]})
    motor = MotorPandas()
    with pytest.raises(ErroMotorTabela):
        motor.filtrar_dados(dados, "__import__('os').system('ls')")

def test_motor_pandas_metodo_clip_e_operadores_logicos() -> None:
    dados = pd.DataFrame({"Metragem": [0.5, 2.0, 150.0, -5.0]})
    motor = MotorPandas()
    
    # Testa transformação com método clip
    resultado_clip = motor.transformar_dados(dados, "Metragem.clip(lower=1)", "Metragem_Ajustada")
    assert list(resultado_clip["Metragem_Ajustada"]) == [1.0, 2.0, 150.0, 1.0]

    # Testa filtro composto com operador lógico
    resultado_filtro = motor.filtrar_dados(dados, "(Metragem > 0) & (Metragem < 100)")
    assert len(resultado_filtro) == 2
    assert list(resultado_filtro["Metragem"]) == [0.5, 2.0]

def test_motor_pandas_sintaxe_df_coluna() -> None:
    dados = pd.DataFrame({"Metragem": [50.0, -10.0, 120.0, 0.0]})
    motor = MotorPandas()
    
    # Testa filtro com sintaxe df['coluna'] > 0
    resultado_filtro = motor.filtrar_dados(dados, "df['Metragem'] > 0")
    assert len(resultado_filtro) == 2
    assert list(resultado_filtro["Metragem"]) == [50.0, 120.0]

    # Testa transformação com df['coluna'].clip(...)
    resultado_clip = motor.transformar_dados(dados, "df['Metragem'].clip(lower=1)", "Metragem_Ajustada")
    assert list(resultado_clip["Metragem_Ajustada"]) == [50.0, 1.0, 120.0, 1.0]

    # Testa criação de novo atributo com df['coluna'] * df['coluna']
    resultado_mult = motor.transformar_dados(dados, "df['Metragem'] * df['Metragem']", "Metragem_Quadrada")
    assert list(resultado_mult["Metragem_Quadrada"]) == [2500.0, 100.0, 14400.0, 0.0]


