import os
import tempfile
import pandas as pd
from app.contratos_dados.fonte_dados import FonteDados
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.fontes_dados.fonte_excel import FonteExcel
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite

def test_fonte_excel_satisfaz_protocolo() -> None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        caminho_tmp = tmp.name
    try:
        df_original = pd.DataFrame({"Valor_da_Venda": [500000.0], "Metragem": [80.0]})
        df_original.to_excel(caminho_tmp, index=False)

        fonte: FonteDados = FonteExcel(caminho=caminho_tmp)
        assert isinstance(fonte, FonteDados)
        df_carregado = fonte.carregar_dados()
        assert df_carregado.shape == (1, 2)
        assert df_carregado.loc[0, "Valor_da_Venda"] == 500000.0
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)

def test_repositorio_sqlite_gravar_e_ler() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        caminho_db = tmp.name
    try:
        repositorio: RepositorioDados = RepositorioSqlite(caminho=caminho_db, tabela="imoveis_teste")
        assert isinstance(repositorio, RepositorioDados)

        dados = pd.DataFrame({"Valor_da_Venda": [300000.0, 450000.0], "Zona": ["Sul", "Norte"]})
        repositorio.gravar_dados(dados)

        dados_lidos = repositorio.ler_dados()
        assert dados_lidos.shape == (2, 2)
        assert list(dados_lidos["Zona"]) == ["Sul", "Norte"]
    finally:
        if os.path.exists(caminho_db):
            os.remove(caminho_db)
