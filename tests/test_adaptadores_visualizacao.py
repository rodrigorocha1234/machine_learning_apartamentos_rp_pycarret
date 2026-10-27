"""Testes automatizados para adaptadores e fábrica de visualizadores (EDA)."""

from __future__ import annotations

from pathlib import Path
import pytest

from app.adaptadores_visualizacao.adaptador_grafana import AdaptadorGrafana
from app.adaptadores_visualizacao.adaptador_relatorios import AdaptadorRelatorios
from app.adaptadores_visualizacao.fabrica_visualizadores import (
    ErroFabricaVisualizadores,
    FabricaVisualizadores,
)
from app.contratos_visualizacao.adaptador_visualizacao import (
    AdaptadorVisualizacao,
    ResultadoAnaliseEda,
)
from app.enums_dominio import MotorGraficoTipo
from app.esquema_configuracao import ConfiguracaoEda


@pytest.fixture
def analises_simuladas() -> list[ResultadoAnaliseEda]:
    """Retorna lista de resultados estatísticos para validação dos adaptadores."""
    return [
        ResultadoAnaliseEda(
            nome="hist_preco",
            titulo="Distribuição do Preço",
            tipo_grafico="histograma",
            descricao="Histograma do valor de venda",
            coluna_principal="Valor_da_Venda",
            dados_estruturados={
                "frequencias": [10, 25, 40, 15],
                "bordas_bins": [100000.0, 300000.0, 500000.0, 700000.0, 1000000.0],
            },
            metricas_resumo={"media": 520000.0, "mediana": 480000.0},
        ),
        ResultadoAnaliseEda(
            nome="tabela_bairro",
            titulo="Resumo por Bairro",
            tipo_grafico="tabela",
            dados_estruturados={
                "linhas": [
                    {
                        "categoria": "Centro",
                        "contagem": 30,
                        "media": 450000.0,
                        "mediana": 420000.0,
                        "min": 250000.0,
                        "max": 800000.0,
                    }
                ]
            },
            metricas_resumo={"total_categorias_listadas": 1.0},
        ),
    ]


def test_adaptador_grafana_renderizacao(
    tmp_path: Path, analises_simuladas: list[ResultadoAnaliseEda]
) -> None:
    caminho_dash = tmp_path / "dashboards" / "dash.json"
    caminho_json = tmp_path / "relatorio.json"

    adaptador = AdaptadorGrafana(caminho_dashboard_provisionado=caminho_dash)
    assert isinstance(adaptador, AdaptadorVisualizacao)
    assert adaptador.obter_nome_motor() == "grafana"

    config = ConfiguracaoEda(
        motor_grafico=MotorGraficoTipo.GRAFANA,
        caminho_relatorio_json=str(caminho_json),
    )

    resultado = adaptador.renderizar_analises(analises_simuladas, config)

    assert resultado.sucesso is True
    assert resultado.total_paineis == 2
    assert caminho_dash.exists()
    assert caminho_json.exists()
    assert "eda-apartamentos-rp" in caminho_dash.read_text(encoding="utf-8")


def test_adaptador_relatorios_renderizacao(
    tmp_path: Path, analises_simuladas: list[ResultadoAnaliseEda]
) -> None:
    caminho_html = tmp_path / "relatorio.html"

    adaptador = AdaptadorRelatorios()
    assert isinstance(adaptador, AdaptadorVisualizacao)
    assert adaptador.obter_nome_motor() == "relatorios"

    config = ConfiguracaoEda(
        motor_grafico=MotorGraficoTipo.SEABORN,
        caminho_relatorio_html=str(caminho_html),
    )

    resultado = adaptador.renderizar_analises(analises_simuladas, config)

    assert resultado.sucesso is True
    assert resultado.total_paineis == 2
    assert caminho_html.exists()
    conteudo = caminho_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in conteudo
    assert "Distribuição do Preço" in conteudo
    assert "Centro" in conteudo


def test_fabrica_visualizadores() -> None:
    config_grafana = ConfiguracaoEda(motor_grafico=MotorGraficoTipo.GRAFANA)
    vis_grafana = FabricaVisualizadores.criar_visualizador(config_grafana)
    assert isinstance(vis_grafana, AdaptadorGrafana)

    config_seaborn = ConfiguracaoEda(motor_grafico=MotorGraficoTipo.SEABORN)
    vis_seaborn = FabricaVisualizadores.criar_visualizador(config_seaborn)
    assert isinstance(vis_seaborn, AdaptadorRelatorios)

    config_invalido = ConfiguracaoEda(motor_grafico=MotorGraficoTipo.GRAFANA)
    config_invalido.motor_grafico = "motor_inexistente"  # type: ignore[assignment]
    with pytest.raises(ErroFabricaVisualizadores, match="Motor gráfico não suportado"):
        FabricaVisualizadores.criar_visualizador(config_invalido)
