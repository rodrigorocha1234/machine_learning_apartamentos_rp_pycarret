import os
import pandas as pd
from app.contratos_dados.motor_treino import MotorTreino
from app.motores_treino.motor_pycaret import MotorPycaret


def test_motor_pycaret_satisfaz_protocolo() -> None:
    motor: MotorTreino = MotorPycaret()
    assert isinstance(motor, MotorTreino)

def test_motor_pycaret_inputs_e_preprocessamento() -> None:
    dados = pd.DataFrame({
        "Valor_da_Venda": [300000.0, 400000.0, 500000.0, 600000.0, 700000.0, 800000.0],
        "Metragem": [50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
        "Zona": ["Sul", "Norte", "Sul", "Norte", "Sul", "Norte"]
    })

    preprocessamento = {
        "ridge": {"escalonador": "standard", "codificador": "one_hot"},
        "random_forest": {"escalonador": "nenhum", "codificador": "one_hot"}
    }

    motor = MotorPycaret(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem"],
        colunas_categoricas=["Zona"],
        preprocessamento_por_modelo=preprocessamento
    )

    # Executa preparação com configuração explícita e One-Hot Encoding
    motor.preparar_experimento(dados, alvo="Valor_da_Venda", modelo_nome="ridge")
    
    # Valida ajuste e tuning de modelo
    resultado_tuning = motor.ajustar_parametros(dados, alvo="Valor_da_Venda", iteracoes=1)
    assert "modelo" in resultado_tuning
    assert resultado_tuning["modelo"] == "ridge"

    # Valida finalização
    modelo_final = motor.treinar_final(dados, alvo="Valor_da_Venda")
    assert modelo_final is not None

def test_motor_pycaret_escolha_metodo_folds() -> None:
    dados = pd.DataFrame({
        "Valor_da_Venda": [250000.0, 350000.0, 450000.0, 550000.0, 650000.0, 750000.0],
        "Metragem": [45.0, 60.0, 80.0, 100.0, 120.0, 140.0],
        "Zona": ["Leste", "Oeste", "Leste", "Oeste", "Leste", "Oeste"]
    })

    # Instancia com método de folds explícito
    motor = MotorPycaret(
        semente=123,
        folds=2,
        metodo_folds="kfold",
        embaralhar_folds=True,
        colunas_numericas=["Metragem"],
        colunas_categoricas=["Zona"]
    )

    motor.preparar_experimento(
        dados,
        alvo="Valor_da_Venda",
        modelo_nome="ridge",
        metodo_folds="kfold",
        embaralhar_folds=True
    )

    tabela_perdas = motor.avaliar_modelos(dados, alvo="Valor_da_Venda")
    assert "ridge" in tabela_perdas.columns
    assert len(tabela_perdas["ridge"]) == 2


def test_motor_pycaret_transformacao_alvo() -> None:
    caminho_xlsx = "storage-data/imoveis_apartamentos_rp.xlsx"
    if os.path.exists(caminho_xlsx):
        dados = (
            pd.read_excel(caminho_xlsx)
            .dropna(subset=["Valor_da_Venda", "Metragem"])[["Valor_da_Venda", "Metragem"]]
            .head(30)
        )
    else:
        dados = pd.DataFrame({
            "Valor_da_Venda": [200000.0, 300000.0, 400000.0, 500000.0, 600000.0, 700000.0],
            "Metragem": [40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
        })

    motor = MotorPycaret(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem"],
        transformar_alvo=True,
        metodo_transformacao_alvo="yeo-johnson"
    )

    modelo = motor.treinar_final(dados, alvo="Valor_da_Venda", modelo_nome="ridge")
    x = dados.drop(columns=["Valor_da_Venda"])
    predicoes = modelo.predict(x)
    assert len(predicoes) == len(dados)
    assert all(float(p) > 100000.0 for p in predicoes)


def test_motor_pycaret_transformacao_atributos() -> None:
    if os.path.exists("storage-data/imoveis_apartamentos_rp.xlsx"):
        dados = pd.read_excel("storage-data/imoveis_apartamentos_rp.xlsx").head(30)
    else:
        dados = pd.DataFrame({
            "Valor_da_Venda": [200000.0, 300000.0, 400000.0, 500000.0, 600000.0, 700000.0],
            "Metragem": [40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
        })

    motor = MotorPycaret(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem"],
        transformar_alvo=False,
        transformar_atributos=True,
        metodo_transformacao_atributos="yeo-johnson"
    )

    modelo = motor.treinar_final(dados, alvo="Valor_da_Venda", modelo_nome="ridge")
    x = dados.drop(columns=["Valor_da_Venda"])
    predicoes = modelo.predict(x)
    assert len(predicoes) == len(dados)
    assert all(float(p) > 50000.0 for p in predicoes)




