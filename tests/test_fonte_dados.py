import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from app.contratos_dados.fonte_dados import FonteDados
from app.fontes_dados.fabrica_fontes import ErroFabricaFontes, FabricaFontes
from app.fontes_dados.fonte_api import ErroFonteApi, FonteApi
from app.fontes_dados.fonte_excel import ErroFonteExcel, FonteExcel
from app.fontes_dados.fonte_sql import ErroFonteSql, FonteSql

def test_fonte_excel_protocolo_e_operacao() -> None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        caminho_tmp = tmp.name
    try:
        df = pd.DataFrame({"Valor_da_Venda": [500000.0, 600000.0], "Metragem": [80.0, 95.0]})
        df.to_excel(caminho_tmp, index=False)

        fonte = FonteExcel(caminho=caminho_tmp)
        assert isinstance(fonte, FonteDados)
        assert fonte.validar_conexao() is True

        dados = fonte.carregar_dados()
        assert len(dados) == 2
        assert "Valor_da_Venda" in dados.columns

        # Testa arquivo inexistente
        fonte_invalida = FonteExcel(caminho="caminho_inexistente.xlsx")
        assert fonte_invalida.validar_conexao() is False
        with pytest.raises(ErroFonteExcel):
            fonte_invalida.carregar_dados()
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)

def test_fonte_sql_protocolo_e_operacao() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        caminho_db = tmp.name
    try:
        url_conexao = f"sqlite:///{caminho_db}"
        import sqlite3
        with sqlite3.connect(caminho_db) as conn:
            conn.execute("CREATE TABLE imoveis (id INTEGER PRIMARY KEY, valor REAL, bairro TEXT)")
            conn.execute("INSERT INTO imoveis (valor, bairro) VALUES (400000.0, 'Centro')")
            conn.execute("INSERT INTO imoveis (valor, bairro) VALUES (600000.0, 'Jardins')")

        fonte = FonteSql(
            url_conexao=url_conexao,
            consulta_sql="SELECT * FROM imoveis WHERE valor >= :min_valor",
            parametros={"min_valor": 500000.0}
        )
        assert isinstance(fonte, FonteDados)
        assert fonte.validar_conexao() is True

        dados = fonte.carregar_dados()
        assert len(dados) == 1
        assert dados.loc[0, "bairro"] == "Jardins"

        # Testa consulta inválida
        fonte_erro = FonteSql(url_conexao=url_conexao, consulta_sql="SELECT * FROM tabela_que_nao_existe")
        with pytest.raises(ErroFonteSql):
            fonte_erro.carregar_dados()
    finally:
        if os.path.exists(caminho_db):
            os.remove(caminho_db)

def test_fonte_api_protocolo_e_operacao() -> None:
    fonte = FonteApi(
        url_endpoint="https://api.exemplo.com/imoveis",
        chave_dados="itens"
    )
    assert isinstance(fonte, FonteDados)

    with patch("requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "itens": [
                {"Valor_da_Venda": 350000.0, "Metragem": 60.0},
                {"Valor_da_Venda": 750000.0, "Metragem": 120.0}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        dados = fonte.carregar_dados()
        assert len(dados) == 2
        assert list(dados["Valor_da_Venda"]) == [350000.0, 750000.0]

    with patch("requests.request", side_effect=Exception("Erro de conexão")):
        with pytest.raises(ErroFonteApi):
            fonte.carregar_dados()

def test_fabrica_fontes_criacao_polimorfica() -> None:
    # 1. Excel
    f_excel = FabricaFontes.criar_fonte({"tipo": "xlsx", "caminho": "dados.xlsx", "aba": 0})
    assert isinstance(f_excel, FonteExcel)

    # 2. SQL
    f_sql = FabricaFontes.criar_fonte({"tipo": "sql", "url": "sqlite:///teste.db", "consulta": "SELECT 1"})
    assert isinstance(f_sql, FonteSql)

    # 3. API
    f_api = FabricaFontes.criar_fonte({"tipo": "api", "url": "https://api.com/v1", "metodo": "GET"})
    assert isinstance(f_api, FonteApi)

    # 4. Tipo desconhecido
    with pytest.raises(ErroFabricaFontes):
        FabricaFontes.criar_fonte({"tipo": "tipo_desconhecido"})
