from __future__ import annotations
import os
import tempfile
import pandas as pd
import pytest
import yaml

from app.contratos_dados.gerador_eda import ErroGeradorEda, GeradorEda
from app.analises_eda.fabrica_eda import FabricaEda
from app.analises_eda.gerador_sweetviz import GeradorSweetviz
from app.analises_eda.gerador_plotly import GeradorPlotly
from app.analises_eda.gerador_nativo import GeradorNativo
from app.analises_eda.analise_exploratoria import AnaliseExploratoria
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite


@pytest.fixture
def dataframe_imoveis_teste() -> pd.DataFrame:
    return pd.DataFrame({
        "Valor_da_Venda": [180000.0, 250000.0, 380000.0, 490000.0, 720000.0, 950000.0],
        "Metragem": [48.0, 62.0, 85.0, 110.0, 145.0, 210.0],
        "Quartos": [1, 2, 3, 3, 4, 4],
        "Vagas": [1, 1, 2, 2, 3, 3],
        "Banheiros": [1, 2, 2, 3, 4, 5],
        "Valor_do_Condominio": [300.0, 450.0, 600.0, 750.0, 1100.0, 1600.0],
        "Zona": ["Sul", "Leste", "Sul", "Oeste", "Sul", "Norte"],
    })


def test_fabrica_eda_dispatch_motores() -> None:
    gerador_sv = FabricaEda.criar_gerador("sweetviz")
    assert isinstance(gerador_sv, GeradorSweetviz)
    assert isinstance(gerador_sv, GeradorEda)

    gerador_pl = FabricaEda.criar_gerador("plotly")
    assert isinstance(gerador_pl, GeradorPlotly)
    assert isinstance(gerador_pl, GeradorEda)

    gerador_nt = FabricaEda.criar_gerador("nativo")
    assert isinstance(gerador_nt, GeradorNativo)
    assert isinstance(gerador_nt, GeradorEda)

    gerador_js = FabricaEda.criar_gerador("json")
    assert isinstance(gerador_js, GeradorNativo)

    with pytest.raises(ErroGeradorEda, match="não suportado"):
        FabricaEda.criar_gerador("motor_inexistente")


def test_gerador_sweetviz_execucao_real(dataframe_imoveis_teste: pd.DataFrame) -> None:
    gerador = GeradorSweetviz()

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste,
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        assert os.path.getsize(caminho_saida) > 1000

        # Valida rejeição de dataframe vazio
        with pytest.raises(ErroGeradorEda, match="vazio"):
            gerador.gerar_relatorio(
                dados=pd.DataFrame(),
                coluna_alvo="Valor_da_Venda",
                caminho_saida=caminho_saida,
            )
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_gerador_plotly_execucao_real(dataframe_imoveis_teste: pd.DataFrame) -> None:
    gerador = GeradorPlotly()

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste,
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        with open(caminho_saida, "r", encoding="utf-8") as f:
            conteudo = f.read()
        assert "Relatório de Análise Exploratória (Plotly)" in conteudo
        assert "Valor_da_Venda" in conteudo
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_gerador_nativo_execucao_real(dataframe_imoveis_teste: pd.DataFrame) -> None:
    gerador = GeradorNativo()

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_html:
        caminho_html = tmp_html.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
        caminho_json = tmp_json.name

    try:
        gerador.gerar_relatorio(dataframe_imoveis_teste, "Valor_da_Venda", caminho_html)
        assert os.path.exists(caminho_html)
        assert os.path.getsize(caminho_html) > 100

        gerador.gerar_relatorio(dataframe_imoveis_teste, "Valor_da_Venda", caminho_json)
        assert os.path.exists(caminho_json)
        assert os.path.getsize(caminho_json) > 100
    finally:
        if os.path.exists(caminho_html):
            os.remove(caminho_html)
        if os.path.exists(caminho_json):
            os.remove(caminho_json)


def test_analise_exploratoria_integracao_gerar_relatorio_visual(dataframe_imoveis_teste: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        caminho_db = tmp_db.name
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_html:
        caminho_html = tmp_html.name

    try:
        repositorio = RepositorioSqlite(caminho=caminho_db, tabela="imoveis_teste_eda")
        repositorio.gravar_dados(dataframe_imoveis_teste)

        eda = AnaliseExploratoria(repositorio)
        saida = eda.gerar_relatorio_visual(
            coluna_alvo="Valor_da_Venda",
            motor="sweetviz",
            caminho_saida=caminho_html,
        )
        assert saida == caminho_html
        assert os.path.exists(caminho_html)
        assert os.path.getsize(caminho_html) > 1000
    finally:
        if os.path.exists(caminho_db):
            os.remove(caminho_db)
        if os.path.exists(caminho_html):
            os.remove(caminho_html)


def test_configuracao_analise_grafica_dataclass() -> None:
    from app.contratos_dados.gerador_eda import ConfiguracaoAnaliseGrafica
    
    cfg = ConfiguracaoAnaliseGrafica.de_dicionario({
        "nome": "teste_hist",
        "tipo_grafico": "histograma",
        "coluna": "Valor_da_Venda",
        "escala": "log",
        "nbins": 25,
    })
    assert cfg.nome == "teste_hist"
    assert cfg.tipo_grafico == "histograma"
    assert cfg.coluna == "Valor_da_Venda"
    assert cfg.escala == "log"
    assert cfg.nbins == 25


def test_gerador_plotly_com_multiplas_analises_cadastradas(dataframe_imoveis_teste: pd.DataFrame) -> None:
    gerador = GeradorPlotly()

    analises_yaml: list[dict[str, str | int | float | bool | None]] = [
        {"nome": "hist_preco", "tipo_grafico": "histograma", "coluna": "Valor_da_Venda", "escala": "log", "ativo": True},
        {"nome": "box_bairro", "tipo_grafico": "boxplot", "coluna_x": "Zona", "coluna_y": "Valor_da_Venda", "ativo": True},
        {"nome": "disp_area", "tipo_grafico": "dispersao", "coluna_x": "Metragem", "coluna_y": "Valor_da_Venda", "ativo": True},
        {"nome": "corr_matriz", "tipo_grafico": "correlacao", "ativo": True},
        {"nome": "bar_zona", "tipo_grafico": "barras", "coluna_x": "Zona", "coluna_y": "Valor_da_Venda", "agregacao": "mediana", "ativo": True},
    ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste,
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
            analises_configuradas=analises_yaml,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        with open(caminho_saida, "r", encoding="utf-8") as f:
            conteudo = f.read()
        assert "Gráficos Cadastrados no YAML: 5" in conteudo
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_configuracao_eda_no_projeto_yaml() -> None:
    caminho_yaml = "configuracoes/projeto.yaml"
    assert os.path.exists(caminho_yaml)

    with open(caminho_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "eda" in config
    bloco_eda = config["eda"]
    assert bloco_eda.get("ativo") is True
    assert bloco_eda.get("motor") in ("sweetviz", "plotly", "altair", "bokeh", "ydata", "nativo")
    assert "caminho_relatorio_html" in bloco_eda
    assert "caminho_relatorio_json" in bloco_eda
    assert "analises" in bloco_eda
    analises = bloco_eda["analises"]
    assert isinstance(analises, list)
    assert len(analises) >= 5

    # Valida tipos de gráficos cadastrados
    tipos_cadastrados = [item.get("tipo_grafico") for item in analises]
    assert "histograma" in tipos_cadastrados
    assert "boxplot" in tipos_cadastrados
    assert "dispersao" in tipos_cadastrados
    assert "correlacao" in tipos_cadastrados
    assert "barras" in tipos_cadastrados
    assert "tabela" in tipos_cadastrados

    # Testa se a Fabrica consegue instanciar o motor do YAML
    motor_configurado = str(bloco_eda["motor"])
    gerador = FabricaEda.criar_gerador(motor_configurado)
    assert isinstance(gerador, GeradorEda)


def test_gerador_plotly_com_tabelas_e_expressoes_python(dataframe_imoveis_teste: pd.DataFrame) -> None:
    gerador = GeradorPlotly()

    analises_teste = [
        {
            "nome": "tabela_bairros_resumo",
            "tipo_grafico": "tabela",
            "titulo": "Resumo por Bairro",
            "agrupar_por": "Bairro",
            "coluna": "Valor_da_Venda",
            "metricas": ["contagem", "media", "mediana", "min", "max"],
            "ordenar_por": "contagem",
            "direcao_ordenacao": "desc",
            "limite": 5,
        },
        {
            "nome": "tabela_ranking_m2",
            "tipo_grafico": "tabela",
            "titulo": "Top Preço por m²",
            "expressao_calculo": "Valor_da_Venda / Metragem",
            "coluna_destino": "Preco_M2",
            "colunas": ["Bairro", "Zona", "Metragem", "Valor_da_Venda", "Preco_M2"],
            "ordenar_por": "Preco_M2",
            "direcao_ordenacao": "desc",
            "limite": 5,
        },
        {
            "nome": "scatter_filtrado",
            "tipo_grafico": "dispersao",
            "titulo": "Metragem vs Preço Sul",
            "expressao_filtro": "Zona == 'Sul'",
            "coluna_x": "Metragem",
            "coluna_y": "Valor_da_Venda",
        },
    ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste,
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
            analises_configuradas=analises_teste,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        with open(caminho_saida, "r", encoding="utf-8") as f:
            html = f.read()

        assert "Gráficos Cadastrados no YAML: 3" in html
        assert "Resumo por Bairro" in html
        assert "Top Preço por m²" in html
        assert "Metragem vs Preço Sul" in html
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_gerador_plotly_seguranca_bloqueio_expressao_maliciosa(dataframe_imoveis_teste: pd.DataFrame) -> None:
    gerador = GeradorPlotly()

    analise_perigosa = [
        {
            "nome": "tentativa_ataque",
            "tipo_grafico": "histograma",
            "expressao_filtro": "__import__('os').system('echo hack')",
            "coluna": "Valor_da_Venda",
        }
    ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        import pytest
        with pytest.raises(Exception) as excinfo:
            gerador.gerar_relatorio(
                dados=dataframe_imoveis_teste,
                coluna_alvo="Valor_da_Venda",
                caminho_saida=caminho_saida,
                analises_configuradas=analise_perigosa,
            )
        assert "Falha ao avaliar expressão de filtro" in str(excinfo.value)
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_fabrica_eda_instanciacao_novos_motores() -> None:
    from app.analises_eda.gerador_ydata import GeradorYData
    from app.analises_eda.gerador_altair import GeradorAltair
    from app.analises_eda.gerador_bokeh import GeradorBokeh

    g_ydata = FabricaEda.criar_gerador("ydata")
    assert isinstance(g_ydata, GeradorYData)

    g_altair = FabricaEda.criar_gerador("altair")
    assert isinstance(g_altair, GeradorAltair)

    g_bokeh = FabricaEda.criar_gerador("bokeh")
    assert isinstance(g_bokeh, GeradorBokeh)


def test_gerador_altair_execucao_real(dataframe_imoveis_teste: pd.DataFrame) -> None:
    from app.analises_eda.gerador_altair import GeradorAltair
    gerador = GeradorAltair()

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste,
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        with open(caminho_saida, "r", encoding="utf-8") as f:
            conteudo = f.read()
        assert "vega" in conteudo.lower() or "altair" in conteudo.lower()
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_gerador_bokeh_execucao_real(dataframe_imoveis_teste: pd.DataFrame) -> None:
    from app.analises_eda.gerador_bokeh import GeradorBokeh
    gerador = GeradorBokeh()

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste,
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        with open(caminho_saida, "r", encoding="utf-8") as f:
            conteudo = f.read()
        assert "bokeh" in conteudo.lower()
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_gerador_ydata_execucao_real(dataframe_imoveis_teste: pd.DataFrame) -> None:
    from app.analises_eda.gerador_ydata import GeradorYData
    gerador = GeradorYData()

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        caminho_saida = tmp.name

    try:
        resultado = gerador.gerar_relatorio(
            dados=dataframe_imoveis_teste.head(10),
            coluna_alvo="Valor_da_Venda",
            caminho_saida=caminho_saida,
        )
        assert resultado == caminho_saida
        assert os.path.exists(caminho_saida)
        assert os.path.getsize(caminho_saida) > 500

        with open(caminho_saida, encoding="utf-8") as f:
            conteudo = f.read()
        assert "Correção do layout e alinhamento dos cabeçalhos das tabelas do YData Profiling" in conteudo
        assert "table.dataframe th" in conteudo
    finally:
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)


def test_fabrica_eda_com_injecao_motor_polars() -> None:
    from app.motores_tabela.motor_polars import MotorPolars
    from app.analises_eda.gerador_plotly import GeradorPlotly
    from app.analises_eda.gerador_altair import GeradorAltair
    from app.analises_eda.gerador_bokeh import GeradorBokeh
    from app.analises_eda.gerador_nativo import GeradorNativo

    gerador_plotly = FabricaEda.criar_gerador("plotly", motor_tabular="polars")
    assert isinstance(gerador_plotly, GeradorPlotly)
    assert isinstance(gerador_plotly._motor_tabela, MotorPolars)

    gerador_altair = FabricaEda.criar_gerador("altair", motor_tabular="polars")
    assert isinstance(gerador_altair, GeradorAltair)
    assert isinstance(gerador_altair._motor_tabela, MotorPolars)

    gerador_bokeh = FabricaEda.criar_gerador("bokeh", motor_tabular="polars")
    assert isinstance(gerador_bokeh, GeradorBokeh)
    assert isinstance(gerador_bokeh._motor_tabela, MotorPolars)

    gerador_nativo = FabricaEda.criar_gerador("nativo", motor_tabular="polars")
    assert isinstance(gerador_nativo, GeradorNativo)
    assert isinstance(gerador_nativo._motor_tabela, MotorPolars)


def test_geradores_eda_execucao_com_motor_polars(dataframe_imoveis_teste: pd.DataFrame) -> None:
    analises = [
        {
            "nome": "teste_calculo_polars",
            "tipo_grafico": "histograma",
            "expressao_calculo": "Valor_da_Venda / Metragem",
            "coluna_destino": "Preco_M2",
            "coluna": "Preco_M2",
            "titulo": "Distribuição Preço por M² via Polars",
        },
        {
            "nome": "teste_filtro_polars",
            "tipo_grafico": "dispersao",
            "expressao_filtro": "Valor_da_Venda <= 800000",
            "coluna_x": "Metragem",
            "coluna_y": "Valor_da_Venda",
            "titulo": "Metragem vs Preço Filtrado via Polars",
        },
    ]

    for motor_nome in ("plotly", "altair", "bokeh", "nativo"):
        gerador = FabricaEda.criar_gerador(motor_nome, motor_tabular="polars")
        extensao = ".json" if motor_nome == "nativo" else ".html"
        with tempfile.NamedTemporaryFile(suffix=extensao, delete=False) as tmp:
            caminho_saida = tmp.name

        try:
            resultado = gerador.gerar_relatorio(
                dados=dataframe_imoveis_teste,
                coluna_alvo="Valor_da_Venda",
                caminho_saida=caminho_saida,
                analises_configuradas=analises,
            )
            assert resultado == caminho_saida
            assert os.path.exists(caminho_saida)
            assert os.path.getsize(caminho_saida) > 100
        finally:
            if os.path.exists(caminho_saida):
                os.remove(caminho_saida)


def test_configuracao_analise_com_motor_tabular_individual() -> None:
    from app.contratos_dados.gerador_eda import ConfiguracaoAnaliseGrafica
    dados_cfg = {
        "nome": "analise_com_polars",
        "tipo_grafico": "histograma",
        "motor_tabular": "polars",
        "coluna": "Valor_da_Venda",
    }
    cfg = ConfiguracaoAnaliseGrafica.de_dicionario(dados_cfg)
    assert cfg.motor_tabular == "polars"



