import pandas as pd
from sklearn.linear_model import LinearRegression
from app.contratos_dados.regra_negocio import DefinicaoRegra
from app.orquestracao_ml.modelo_servico import ModeloServico

def test_modelo_servico_predicao_com_regras_dinamicas() -> None:
    # Cria modelo simples treinado
    x = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0]})
    y = [250000.0, 500000.0, 750000.0]
    modelo = LinearRegression().fit(x, y)

    regras = [
        DefinicaoRegra(
            nome="margem_inferior_negociacao",
            ativo=True,
            descricao="Faixa mínima de 5%",
            destino="preco_minimo_sugerido",
            expressao="preco_predito * 0.95"
        ),
        DefinicaoRegra(
            nome="margem_superior_negociacao",
            ativo=True,
            descricao="Faixa máxima de 5%",
            destino="preco_maximo_sugerido",
            expressao="preco_predito * 1.05"
        ),
        DefinicaoRegra(
            nome="valor_metro_quadrado_predito",
            ativo=True,
            descricao="Preço por m2",
            destino="preco_m2_predito",
            expressao="preco_predito / Metragem"
        ),
        DefinicaoRegra(
            nome="regra_inativa",
            ativo=False,
            descricao="Regra desativada",
            destino="regra_inativa",
            expressao="preco_predito * 2.0"
        ),
    ]

    servico = ModeloServico(modelo_campeao=modelo, regras=regras, coluna_alvo_predito="preco_predito")
    
    dados_entrada = pd.DataFrame({"Metragem": [100.0]})
    resposta = servico.predict(context=None, model_input=dados_entrada)

    assert "preco_predito" in resposta.columns
    assert "preco_minimo_sugerido" in resposta.columns
    assert "preco_maximo_sugerido" in resposta.columns
    assert "preco_m2_predito" in resposta.columns
    assert "regra_inativa" not in resposta.columns

    assert round(float(resposta.loc[0, "preco_predito"]), 2) == 500000.0
    assert round(float(resposta.loc[0, "preco_minimo_sugerido"]), 2) == 475000.0
    assert round(float(resposta.loc[0, "preco_maximo_sugerido"]), 2) == 525000.0
    assert round(float(resposta.loc[0, "preco_m2_predito"]), 2) == 5000.0
