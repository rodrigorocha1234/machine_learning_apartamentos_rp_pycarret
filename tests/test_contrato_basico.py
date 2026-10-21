from app.analises_eda.analise_basica import AnaliseBasica
import pandas as pd

def test_eda_basica() -> None:
    resultado = AnaliseBasica().resumir_dados(pd.DataFrame({"preco": [100, 200]}))
    assert resultado.loc["preco", "mean"] == 150
