import pytest
import pandas as pd
from app.contratos_dados.regra_negocio import DefinicaoRegra, RegraNegocio
from app.regras_negocio.avaliador_regras import AvaliadorRegras, ErroRegraNegocio

def test_avaliador_satisfaz_protocolo() -> None:
    avaliador: RegraNegocio = AvaliadorRegras()
    assert isinstance(avaliador, RegraNegocio)

def test_avaliar_regras_dinamicas_sucesso() -> None:
    dados = pd.DataFrame({
        "preco_predito": [500000.0, 1000000.0],
        "Metragem": [100.0, 200.0]
    })
    regras = [
        DefinicaoRegra(
            nome="margem_inferior",
            ativo=True,
            descricao="Margem inferior de 5%",
            destino="preco_minimo",
            expressao="preco_predito * 0.95"
        ),
        DefinicaoRegra(
            nome="margem_superior",
            ativo=True,
            descricao="Margem superior de 5%",
            destino="preco_maximo",
            expressao="preco_predito * 1.05"
        ),
        DefinicaoRegra(
            nome="valor_m2",
            ativo=True,
            descricao="Preço por m2",
            destino="preco_m2",
            expressao="preco_predito / Metragem"
        ),
        DefinicaoRegra(
            nome="regra_inativa",
            ativo=False,
            descricao="Não deve ser calculada",
            destino="coluna_inativa",
            expressao="preco_predito * 2"
        )
    ]
    avaliador = AvaliadorRegras()
    resultado = avaliador.aplicar_regra(dados, regras)

    assert "preco_minimo" in resultado.columns
    assert "preco_maximo" in resultado.columns
    assert "preco_m2" in resultado.columns
    assert "coluna_inativa" not in resultado.columns

    assert resultado.loc[0, "preco_minimo"] == 475000.0
    assert resultado.loc[0, "preco_maximo"] == 525000.0
    assert resultado.loc[0, "preco_m2"] == 5000.0

    assert resultado.loc[1, "preco_minimo"] == 950000.0
    assert resultado.loc[1, "preco_maximo"] == 1050000.0
    assert resultado.loc[1, "preco_m2"] == 5000.0

def test_bloqueio_injecao_codigo() -> None:
    dados = pd.DataFrame({"preco_predito": [500000.0]})
    avaliador = AvaliadorRegras()

    expressoes_perigosas = [
        "__import__('os').system('echo 1')",
        "open('/etc/passwd').read()",
        "preco_predito.__class__",
        "lambda x: x",
    ]

    for expressao in expressoes_perigosas:
        regra = [DefinicaoRegra(
            nome="ataque",
            ativo=True,
            descricao="Tentativa de injeção",
            destino="alvo",
            expressao=expressao
        )]
        with pytest.raises(ErroRegraNegocio):
            avaliador.aplicar_regra(dados, regra)

def test_coluna_inexistente_lanca_erro() -> None:
    dados = pd.DataFrame({"preco_predito": [500000.0]})
    avaliador = AvaliadorRegras()
    regra = [DefinicaoRegra(
        nome="invalida",
        ativo=True,
        descricao="Coluna inexistente",
        destino="alvo",
        expressao="preco_predito / Coluna_Inexistente"
    )]
    with pytest.raises(ErroRegraNegocio):
        avaliador.aplicar_regra(dados, regra)
