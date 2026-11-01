"""Testes para o publicador e gerador de dashboard de modelos e tuning no Grafana."""

from __future__ import annotations

import json
from pathlib import Path

from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo, ResultadoTuning
from app.servicos_observabilidade.publicador_modelos_grafana import (
    PublicadorModelosGrafana,
    formatar_nome_modelo,
)


def test_formatar_nome_modelo() -> None:
    """Verifica a conversão para nomes canônicos em português."""
    assert formatar_nome_modelo("regressao_linear") == "regressão linear"
    assert formatar_nome_modelo("arvore_decisao") == "árvore de decisão"
    assert formatar_nome_modelo("random_forest") == "random forest"
    assert formatar_nome_modelo("rede_neural") == "rede neural"
    assert formatar_nome_modelo("lasso") == "lasso"


def test_registro_logs_tuning() -> None:
    """Valida o buffer de logs e mensagens de tuning."""
    PublicadorModelosGrafana.limpar_logs()

    PublicadorModelosGrafana.registrar_log(
        "regressao_linear", "realizando tuning de parâmetros regressão linear"
    )
    PublicadorModelosGrafana.registrar_log(
        "lasso", "realizando tuning de parâmetros lasso"
    )

    logs = PublicadorModelosGrafana.obter_logs()
    assert len(logs) == 2
    assert logs[0]["modelo"] == "regressao_linear"
    assert "realizando tuning de parâmetros regressão linear" in logs[0]["mensagem"]
    assert logs[1]["modelo"] == "lasso"
    assert "realizando tuning de parâmetros lasso" in logs[1]["mensagem"]


def test_geracao_dashboard_grafana_modelos(tmp_path: Path) -> None:
    """Garante que o dashboard JSON do Grafana é gerado com os logs e painéis de tuning."""
    caminho_dash = tmp_path / "dashboard_modelos.json"
    publicador = PublicadorModelosGrafana(caminho_dashboard=caminho_dash)

    PublicadorModelosGrafana.limpar_logs()
    PublicadorModelosGrafana.registrar_log(
        "regressao_linear", "realizando tuning de parâmetros regressão linear"
    )
    PublicadorModelosGrafana.registrar_log(
        "lasso", "realizando tuning de parâmetros lasso"
    )
    PublicadorModelosGrafana.registrar_log(
        "random_forest", "realizando tuning de parâmetros random forest"
    )

    res_tuning = ResultadoTuning(
        metodo_tuning="random",
        iteracoes=10,
        modelos_treinados=[
            ResultadoTreinoModelo(
                nome_modelo="regressao_linear",
                framework="sklearn",
                melhores_parametros={"fit_intercept": True},
                metricas={"RMSE": 42500.0, "MAE": 31000.0, "R2": 0.81},
                duracao_segundos=0.45,
                tuning_ativo=True,
                mensagem_tuning="realizando tuning de parâmetros regressão linear",
            ),
            ResultadoTreinoModelo(
                nome_modelo="lasso",
                framework="sklearn",
                melhores_parametros={"alpha": 0.1},
                metricas={"RMSE": 41200.0, "MAE": 29800.0, "R2": 0.83},
                duracao_segundos=0.52,
                tuning_ativo=True,
                mensagem_tuning="realizando tuning de parâmetros lasso",
            ),
        ],
        modelo_campeao="lasso",
        melhor_rmse=41200.0,
    )

    caminho_gerado = publicador.publicar_dashboard_modelos(res_tuning, motor="sklearn")

    assert caminho_gerado == caminho_dash
    assert caminho_dash.exists()

    conteudo_json = json.loads(caminho_dash.read_text(encoding="utf-8"))

    assert conteudo_json["uid"] == "modelos-apartamentos-rp"
    assert conteudo_json["timezone"] == "America/Sao_Paulo"
    assert len(conteudo_json["panels"]) >= 3

    # Painel de logs/eventos
    painel_logs = next(p for p in conteudo_json["panels"] if p["id"] == 2)
    markdown_content = painel_logs["options"]["content"]
    assert "realizando tuning de parâmetros regressão linear" in markdown_content
    assert "realizando tuning de parâmetros lasso" in markdown_content
    assert "UTC" not in markdown_content

    # Painel 4: Diagnóstico de Overfitting e Underfitting
    painel_fit = next(p for p in conteudo_json["panels"] if p["id"] == 4)
    assert "Overfitting" in painel_fit["title"]
    assert "Underfitting" in painel_fit["title"]
    assert "Gap de Generalização" in painel_fit["options"]["content"]
    assert "Ajuste Ótimo" in painel_fit["options"]["content"]


