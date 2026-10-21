import pytest
import pandas as pd
from app.contratos_dados.motor_tabela import MotorTabela
from app.motores_tabela.fabrica_motores import FabricaMotores, ErroFabricaMotores
from app.motores_tabela.motor_pandas import MotorPandas
from app.motores_tabela.motor_polars import MotorPolars, ErroMotorPolars

def test_motor_polars_satisfaz_protocolo() -> None:
    motor: MotorTabela = MotorPolars()
    assert isinstance(motor, MotorTabela)

def test_motor_polars_filtrar_dados() -> None:
    dados = pd.DataFrame({"Metragem": [45.0, -5.0, 110.0, 0.0]})
    motor = MotorPolars()
    resultado = motor.filtrar_dados(dados, "Metragem > 0")
    assert len(resultado) == 2
    assert list(resultado["Metragem"]) == [45.0, 110.0]

def test_motor_polars_transformar_dados() -> None:
    dados = pd.DataFrame({"Metragem": [10.0, 15.0]})
    motor = MotorPolars()
    resultado = motor.transformar_dados(dados, "Metragem * Metragem", "Metragem_Quadrada")
    assert "Metragem_Quadrada" in resultado.columns
    assert list(resultado["Metragem_Quadrada"]) == [100.0, 225.0]

def test_motor_polars_selecionar_colunas() -> None:
    dados = pd.DataFrame({"X": [1, 2], "Y": [3, 4], "Z": [5, 6]})
    motor = MotorPolars()
    resultado = motor.selecionar_colunas(dados, ["X", "Z"])
    assert list(resultado.columns) == ["X", "Z"]


def test_motor_polars_rejeita_codigo_perigoso() -> None:
    dados = pd.DataFrame({"Metragem": [50.0]})
    motor = MotorPolars()
    with pytest.raises(ErroMotorPolars):
        motor.filtrar_dados(dados, "__import__('os').system('echo 1')")

def test_fabrica_motores_dispatch() -> None:
    motor_pandas = FabricaMotores.criar_motor("pandas")
    assert isinstance(motor_pandas, MotorPandas)

    motor_polars = FabricaMotores.criar_motor("polars")
    assert isinstance(motor_polars, MotorPolars)

    with pytest.raises(ErroFabricaMotores):
        FabricaMotores.criar_motor("motor_invalido")

def test_motor_polars_metodo_clip_e_operadores_logicos() -> None:
    dados = pd.DataFrame({"Metragem": [0.5, 2.0, 150.0, -5.0]})
    motor = MotorPolars()
    
    # Testa transformação com método clip nativo Polars
    resultado_clip = motor.transformar_dados(dados, "Metragem.clip(lower_bound=1)", "Metragem_Ajustada")
    assert list(resultado_clip["Metragem_Ajustada"]) == [1.0, 2.0, 150.0, 1.0]

    # Testa transformação com lower transparente
    resultado_clip_lower = motor.transformar_dados(dados, "Metragem.clip(lower=1)", "Metragem_Ajustada2")
    assert list(resultado_clip_lower["Metragem_Ajustada2"]) == [1.0, 2.0, 150.0, 1.0]

    # Testa filtro composto com operador lógico
    resultado_filtro = motor.filtrar_dados(dados, "(Metragem > 0) & (Metragem < 100)")
    assert len(resultado_filtro) == 2
    assert list(resultado_filtro["Metragem"]) == [0.5, 2.0]

def test_motor_polars_sintaxe_df_coluna_e_pl_col() -> None:
    dados = pd.DataFrame({"Metragem": [50.0, -10.0, 120.0, 0.0]})
    motor = MotorPolars()
    
    # Testa filtro com sintaxe df['coluna'] > 0
    resultado_filtro = motor.filtrar_dados(dados, "df['Metragem'] > 0")
    assert len(resultado_filtro) == 2
    assert list(resultado_filtro["Metragem"]) == [50.0, 120.0]

    # Testa filtro com sintaxe pl.col('coluna') > 0
    resultado_filtro_pl = motor.filtrar_dados(dados, "pl.col('Metragem') > 0")
    assert len(resultado_filtro_pl) == 2
    assert list(resultado_filtro_pl["Metragem"]) == [50.0, 120.0]

    # Testa transformação com df['coluna'].clip(...)
    resultado_clip = motor.transformar_dados(dados, "df['Metragem'].clip(lower_bound=1)", "Metragem_Ajustada")
    assert list(resultado_clip["Metragem_Ajustada"]) == [50.0, 1.0, 120.0, 1.0]

    # Testa criação de novo atributo com df['coluna'] * df['coluna']
    resultado_mult = motor.transformar_dados(dados, "df['Metragem'] * df['Metragem']", "Metragem_Quadrada")
    assert list(resultado_mult["Metragem_Quadrada"]) == [2500.0, 100.0, 14400.0, 0.0]


