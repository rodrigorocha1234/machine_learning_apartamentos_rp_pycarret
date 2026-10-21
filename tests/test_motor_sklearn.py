import pandas as pd
import pytest
from app.contratos_dados.motor_treino import MotorTreino
from app.motores_treino.motor_sklearn import ErroMotorTreino, MotorSklearn

def test_motor_sklearn_satisfaz_protocolo() -> None:
    motor: MotorTreino = MotorSklearn()
    assert isinstance(motor, MotorTreino)

def test_motor_sklearn_ajuste_e_avaliacao_10_modelos() -> None:
    dados = pd.DataFrame({
        "Valor_da_Venda": [250000.0, 350000.0, 450000.0, 550000.0, 650000.0, 750000.0],
        "Metragem": [45.0, 65.0, 85.0, 105.0, 125.0, 145.0],
        "Zona": ["Sul", "Norte", "Leste", "Oeste", "Sul", "Norte"]
    })

    modelos_10 = [
        {"nome": "regressao_linear", "ativo": True},
        {"nome": "lasso", "ativo": True},
        {"nome": "ridge", "ativo": True},
        {"nome": "elasticnet", "ativo": True},
        {"nome": "arvore_decisao", "ativo": True},
        {"nome": "random_forest", "ativo": True},
        {"nome": "svr", "ativo": True},
        {"nome": "rede_neural", "ativo": True},
        {"nome": "xgboost", "ativo": True},
        {"nome": "lightgbm", "ativo": True},
    ]

    preprocessamento = {
        "regressao_linear": {"escalonador": "standard", "codificador": "one_hot"},
        "lasso": {"escalonador": "standard", "codificador": "one_hot"},
        "ridge": {"escalonador": "standard", "codificador": "one_hot"},
        "elasticnet": {"escalonador": "standard", "codificador": "one_hot"},
        "arvore_decisao": {"escalonador": "nenhum", "codificador": "one_hot"},
        "random_forest": {"escalonador": "nenhum", "codificador": "one_hot"},
        "svr": {"escalonador": "standard", "codificador": "one_hot"},
        "rede_neural": {"escalonador": "standard", "codificador": "one_hot"},
        "xgboost": {"escalonador": "nenhum", "codificador": "one_hot"},
        "lightgbm": {"escalonador": "nenhum", "codificador": "one_hot"},
    }

    tuning_params = {
        "ridge": {"alpha": [0.1, 1.0]},
        "xgboost": {"n_estimators": [10, 20]}
    }

    motor = MotorSklearn(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem"],
        colunas_categoricas=["Zona"],
        preprocessamento_por_modelo=preprocessamento,
        modelos_configurados=modelos_10,
        parametros_tuning=tuning_params
    )

    # 1. Validação cruzada pareada de todos os 10 modelos
    avaliacao = motor.avaliar_modelos(dados, alvo="Valor_da_Venda")
    for mod in modelos_10:
        nome = str(mod["nome"])
        assert nome in avaliacao.columns
        assert len(avaliacao[nome]) == 2

    # 2. Tuning de hiperparâmetros para um modelo específico
    resultado_tuning = motor.ajustar_parametros(
        dados,
        alvo="Valor_da_Venda",
        iteracoes=2,
        modelo_nome="xgboost"
    )
    assert resultado_tuning["modelo"] == "xgboost"
    assert "melhores_parametros" in resultado_tuning

    # 3. Finalização
    modelo_final = motor.treinar_final(dados, alvo="Valor_da_Venda", modelo_nome="xgboost")
    assert modelo_final is not None

def test_motor_sklearn_escolha_metodo_tuning() -> None:
    dados = pd.DataFrame({
        "Valor_da_Venda": [250000.0, 350000.0, 450000.0, 550000.0, 650000.0, 750000.0],
        "Metragem": [45.0, 65.0, 85.0, 105.0, 125.0, 145.0],
    })
    tuning_params = {"ridge": {"alpha": [0.1, 1.0]}}

    # Testa GridSearchCV
    motor_grid = MotorSklearn(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem"],
        parametros_tuning=tuning_params,
        metodo_tuning="grid"
    )
    res_grid = motor_grid.ajustar_parametros(dados, alvo="Valor_da_Venda", iteracoes=2, modelo_nome="ridge")
    assert res_grid["modelo"] == "ridge"
    assert "melhores_parametros" in res_grid

    # Testa RandomizedSearchCV
    motor_random = MotorSklearn(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem"],
        parametros_tuning=tuning_params,
        metodo_tuning="random"
    )
    res_random = motor_random.ajustar_parametros(dados, alvo="Valor_da_Venda", iteracoes=2, modelo_nome="ridge")
    assert res_random["modelo"] == "ridge"
    assert "melhores_parametros" in res_random

    # Testa sobrescrita dinâmica de método via parâmetro
    res_sobrescrito = motor_random.ajustar_parametros(
        dados,
        alvo="Valor_da_Venda",
        iteracoes=2,
        modelo_nome="ridge",
        metodo_tuning="grid"
    )
    assert res_sobrescrito["modelo"] == "ridge"
    assert "melhores_parametros" in res_sobrescrito

    # Testa rejeição de método inválido
    with pytest.raises(ErroMotorTreino, match="não suportado no motor Sklearn"):
        motor_random.ajustar_parametros(
            dados,
            alvo="Valor_da_Venda",
            iteracoes=2,
            modelo_nome="ridge",
            metodo_tuning="metodo_inexistente"
        )
