import os
import tempfile
import pandas as pd
from app.analises_eda.analise_exploratoria import AnaliseExploratoria
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite

def test_executar_eda_com_dados_staging() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        caminho_db = tmp.name
    try:
        repositorio = RepositorioSqlite(caminho=caminho_db, tabela="imoveis_eda")
        dados = pd.DataFrame({
            "Valor_da_Venda": [100000.0, 200000.0, 300000.0],
            "Metragem": [50.0, 75.0, 100.0],
            "Zona": ["Sul", "Sul", "Norte"]
        })
        repositorio.gravar_dados(dados)

        eda = AnaliseExploratoria(repositorio)
        relatorio = eda.executar_eda(coluna_alvo="Valor_da_Venda")

        assert relatorio["total_linhas"] == 3
        assert relatorio["total_colunas"] == 3
        assert "distribuicao_alvo" in relatorio
        dist = relatorio["distribuicao_alvo"]
        assert isinstance(dist, dict)
        assert dist["mediana"] == 200000.0
        assert dist["minimo"] == 100000.0

        medianas_zona = eda.calcular_mediana_agrupada("Valor_da_Venda", "Zona")
        assert medianas_zona["Sul"] == 150000.0
        assert medianas_zona["Norte"] == 300000.0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_tmp:
            json_caminho = json_tmp.name
        eda.salvar_relatorio_json(relatorio, json_caminho)
        assert os.path.exists(json_caminho)
        assert os.path.getsize(json_caminho) > 0
        os.remove(json_caminho)
    finally:
        if os.path.exists(caminho_db):
            os.remove(caminho_db)
