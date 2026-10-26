"""Testes automatizados de integração para o serviço AnalisadorEda."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from app.adaptadores_visualizacao.adaptador_grafana import AdaptadorGrafana
from app.adaptadores_visualizacao.adaptador_relatorios import AdaptadorRelatorios
from app.conf_yaml import ConfYAML
from app.esquema_configuracao import ConfiguracaoEda, ItemAnaliseEda
from app.servicos_eda.analisador_eda import AnalisadorEda
from app.servicos_eda.calculador_estatistico import CalculadorEstatistico


@pytest.fixture
def dados_apartamentos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Valor_da_Venda": [450000.0, 700000.0, 950000.0, 1300000.0, 1800000.0],
            "Metragem": [45.0, 68.0, 95.0, 130.0, 180.0],
            "Quartos": [1, 2, 3, 3, 4],
            "Banheiros": [1, 2, 2, 3, 4],
            "Vagas": [1, 1, 2, 2, 3],
            "Bairro": ["Centro", "Centro", "Jardim Botânico", "Jardim Botânico", "Vila Ana Maria"],
            "Zona": ["Centro", "Centro", "Sul", "Sul", "Sul"],
            "Valor_Condominio": [400.0, 600.0, 900.0, 1200.0, 1600.0],
            "Valor_Iptu": [500.0, 800.0, 1200.0, 1800.0, 2400.0],
            "Area_Privativa": [45.0, 68.0, 95.0, 130.0, 180.0],
        }
    )


def test_analisador_eda_fluxo_completo_grafana(
    tmp_path: Path, dados_apartamentos: pd.DataFrame
) -> None:
    caminho_dash = tmp_path / "dashboard_eda.json"
    caminho_json = tmp_path / "relatorio_eda.json"

    config_eda = ConfiguracaoEda(
        motor_grafico="grafana",
        caminho_relatorio_json=str(caminho_json),
        analises=[
            ItemAnaliseEda(
                nome="dist_preco",
                tipo_grafico="histograma",
                coluna="Valor_da_Venda",
                nbins=5,
            ),
            ItemAnaliseEda(
                nome="disp_bairro",
                tipo_grafico="boxplot",
                coluna_x="Bairro",
                coluna_y="Valor_da_Venda",
            ),
        ],
    )

    adaptador = AdaptadorGrafana(caminho_dashboard_provisionado=caminho_dash)
    analisador = AnalisadorEda(
        adaptador=adaptador,
        calculador=CalculadorEstatistico(),
        configuracao_eda=config_eda,
    )

    analises, res_vis, res_eda = analisador.executar_eda(dados_apartamentos)

    assert len(analises) == 2
    assert res_vis.sucesso is True
    assert res_eda.total_analises_processadas == 2
    assert res_eda.motor_visualizacao == "grafana"
    assert caminho_dash.exists()
    assert caminho_json.exists()


def test_analisador_eda_fluxo_completo_relatorios_html(
    tmp_path: Path, dados_apartamentos: pd.DataFrame
) -> None:
    caminho_html = tmp_path / "relatorio_eda.html"

    config_eda = ConfiguracaoEda(
        motor_grafico="seaborn",
        caminho_relatorio_html=str(caminho_html),
        analises=[
            ItemAnaliseEda(
                nome="preco_m2_expr",
                tipo_grafico="histograma",
                coluna="Preco_M2",
                expressao="df['Valor_da_Venda'] / df['Metragem']",
                nbins=5,
            ),
        ],
    )

    adaptador = AdaptadorRelatorios()
    analisador = AnalisadorEda(
        adaptador=adaptador,
        calculador=CalculadorEstatistico(),
        configuracao_eda=config_eda,
    )

    analises, res_vis, res_eda = analisador.executar_eda(dados_apartamentos)

    assert len(analises) == 1
    assert res_vis.sucesso is True
    assert res_eda.total_analises_processadas == 1
    assert caminho_html.exists()
    assert "Preco_M2" in caminho_html.read_text(encoding="utf-8")


def test_analisador_eda_instanciacao_conf_yaml() -> None:
    conf = ConfYAML()
    analisador = AnalisadorEda.da_configuracao(conf)
    assert isinstance(analisador, AnalisadorEda)


def test_analisador_eda_execucao_com_staging_real() -> None:
    conf = ConfYAML()
    analisador = AnalisadorEda.da_configuracao(conf)

    analises, res_vis, res_eda = analisador.executar_eda()

    assert len(analises) >= 5
    assert res_vis.sucesso is True
    assert res_eda.total_analises_processadas >= 5
    assert res_eda.tempo_execucao_segundos >= 0.0

    # Verifica se os artefatos foram gerados
    caminho_dash = Path("config_ob/dashboards/dashboard_eda.json")
    caminho_json = Path("storage-data/relatorio_eda.json")
    assert caminho_dash.exists()
    assert caminho_json.exists()
