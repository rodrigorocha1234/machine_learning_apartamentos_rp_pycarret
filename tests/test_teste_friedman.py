import pandas as pd
from app.contratos_dados.avaliador_estatistico import AvaliadorEstatistico
from app.analises_estatisticas.teste_friedman import TesteFriedman

def test_teste_friedman_satisfaz_protocolo() -> None:
    avaliador: AvaliadorEstatistico = TesteFriedman()
    assert isinstance(avaliador, AvaliadorEstatistico)

def test_teste_friedman_identifica_campeao() -> None:
    # 5 folds pareados, Modelo A tem perdas menores que Modelo B e Modelo C
    perdas = pd.DataFrame({
        "modelo_a": [10.0, 12.0, 11.0, 13.0, 10.5],
        "modelo_b": [20.0, 22.0, 21.0, 23.0, 20.5],
        "modelo_c": [30.0, 32.0, 31.0, 33.0, 30.5],
    })
    avaliador = TesteFriedman()
    resultado = avaliador.testar_diferencas(perdas, alpha=0.05)

    assert resultado.modelo_campeao == "modelo_a"
    assert resultado.hipotese_nula_rejeitada is True
    assert resultado.postos_medios["modelo_a"] == 1.0
    assert resultado.postos_medios["modelo_c"] == 3.0
    assert not resultado.comparacao_pos_hoc.empty
