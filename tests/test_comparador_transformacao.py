from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from app.analises_estatisticas.comparador_transformacao import (
    ComparadorTransformacao,
    ErroComparadorTransformacao,
    RelatorioTransformacao,
)


def test_comparador_transformacao_simulado() -> None:
    comparador = ComparadorTransformacao(semente=42)
    relatorio = comparador.gerar_relatorio_simulado(motor="sklearn", metodo_transformacao="log1p")

    assert isinstance(relatorio, RelatorioTransformacao)
    assert relatorio.motor == "sklearn"
    assert relatorio.metodo_transformacao == "log1p"
    assert len(relatorio.modelos) >= 10
    assert relatorio.reducao_media_rmse_percentual > 10.0
    assert "transformação logarítmica" in relatorio.recomendacao_geral

    # Verifica os modelos configurados
    nomes_modelos = [m.modelo for m in relatorio.modelos]
    assert "random_forest" in nomes_modelos
    assert "ridge" in nomes_modelos
    assert "lasso" in nomes_modelos
    assert "modelo_campeao" in nomes_modelos

    for m in relatorio.modelos:
        assert m.rmse_com_log < m.rmse_sem_log, f"Modelo {m.modelo} deveria ter menor RMSE com log"
        assert m.reducao_rmse_percentual > 0, f"Redução de RMSE deveria ser positiva para {m.modelo}"
        assert np.isclose(m.delta_rmse, m.rmse_sem_log - m.rmse_com_log, atol=0.01)
        assert m.r2_com_log > m.r2_sem_log, f"R² com log deveria ser maior para {m.modelo}"
        assert m.ganho_r2 > 0
        assert "Recomendado" in m.veredito

    # Validação da conversão para dicionário e DataFrame
    dicionario = relatorio.para_dicionario()
    assert dicionario["motor"] == "sklearn"
    assert isinstance(dicionario["modelos"], list)
    assert len(dicionario["modelos"]) == len(relatorio.modelos)

    df_relatorio = relatorio.para_dataframe()
    assert isinstance(df_relatorio, pd.DataFrame)
    assert "modelo" in df_relatorio.columns
    assert "reducao_rmse_percentual" in df_relatorio.columns
    assert "veredito" in df_relatorio.columns


def test_comparador_transformacao_dados_sinteticos() -> None:
    np.random.seed(42)
    n_amostras = 60
    # Distribuição com assimetria positiva típica de preços imobiliários (log-normal)
    preco_log = np.random.normal(12.5, 0.45, n_amostras)
    precos = np.exp(preco_log)

    dados = pd.DataFrame({
        "metragem": np.random.uniform(40, 250, n_amostras),
        "quartos": np.random.choice([1, 2, 3, 4], n_amostras),
        "banheiros": np.random.choice([1, 2, 3], n_amostras),
        "bairro": np.random.choice(["Centro", "Zona Sul", "Vila Olimpia"], n_amostras),
        "preco": precos,
    })

    comparador = ComparadorTransformacao(semente=42, cv_splits=3, metodo_transformacao="log1p")
    relatorio = comparador.avaliar_comparacao_dados(
        dados=dados,
        alvo="preco",
        modelos=["ridge", "random_forest"],
        motor="sklearn"
    )

    assert len(relatorio.modelos) == 2
    for m in relatorio.modelos:
        assert m.rmse_sem_log > 0
        assert m.rmse_com_log > 0
        assert m.mae_sem_log > 0
        assert m.mae_com_log > 0
        # Confirma que ambas as métricas estão na escala de preços em reais (ordem de centenas de milhares)
        assert m.rmse_sem_log > 1000.0, "RMSE sem log deve estar na escala de reais"
        assert m.rmse_com_log > 1000.0, "RMSE com log revertido deve estar na escala de reais"


def test_comparador_transformacao_validacao_alvo_ausente() -> None:
    dados = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    comparador = ComparadorTransformacao()
    import pytest
    with pytest.raises(ErroComparadorTransformacao):
        comparador.avaliar_modelo_dados(dados, alvo="coluna_inexistente", modelo_nome="ridge")


def test_salvar_metricas_prometheus(tmp_path: Path) -> None:
    comparador = ComparadorTransformacao(semente=42)
    relatorio = comparador.gerar_relatorio_simulado()

    caminho_prom = str(tmp_path / "metricas_test.prom")
    comparador.salvar_metricas_prometheus(relatorio, caminho_arquivo=caminho_prom)

    assert os.path.exists(caminho_prom)
    with open(caminho_prom, "r", encoding="utf-8") as f:
        conteudo = f.read()

    assert "ml_treino_rmse_sem_log" in conteudo
    assert "ml_treino_rmse_com_log" in conteudo
    assert "ml_treino_reducao_rmse_percentual" in conteudo
    assert "ml_treino_mae_sem_log" in conteudo
    assert "ml_treino_mae_com_log" in conteudo
    assert "ml_treino_r2_sem_log" in conteudo
    assert "ml_treino_r2_com_log" in conteudo


def test_dashboard_grafana_recursos_nativos_treino_log() -> None:
    caminho_dashboard = "config_ob/dashboards/dashboard_monitoramento_ml.json"
    assert os.path.exists(caminho_dashboard), f"Dashboard não encontrado em {caminho_dashboard}"

    with open(caminho_dashboard, "r", encoding="utf-8") as f:
        dados_dash = json.load(f)

    # 1. Valida existência da Row 4.4 de Comparativo de Treino Sem Log vs Com Log
    titulos_rows = [p["title"] for p in dados_dash["panels"] if p.get("type") == "row"]
    assert any("4.4 – Análise Comparativa de Treino: Sem Log vs Com Log" in t for t in titulos_rows), \
        "Row 4.4 de Análise Comparativa Sem Log vs Com Log ausente"

    # 2. Valida Painel 21 (Barchart nativo de comparação)
    painel_barchart = next((p for p in dados_dash["panels"] if p.get("id") == 21), None)
    assert painel_barchart is not None, "Painel 21 (barchart comparativo) não encontrado"
    assert painel_barchart["type"] == "barchart"
    assert painel_barchart["options"]["orientation"] == "vertical"
    assert painel_barchart["fieldConfig"]["defaults"]["unit"] == "currencyBRL"
    overrides_21 = painel_barchart["fieldConfig"]["overrides"]
    series_21 = [o["matcher"]["options"] for o in overrides_21]
    assert "RMSE Sem Log (R$)" in series_21
    assert "RMSE Com Log (R$)" in series_21

    # 3. Valida Painéis 22 e 23 (Stat com percentual de redução)
    painel_stat_rmse = next((p for p in dados_dash["panels"] if p.get("id") == 22), None)
    assert painel_stat_rmse is not None, "Painel 22 (stat redução RMSE) não encontrado"
    assert painel_stat_rmse["type"] == "stat"
    assert painel_stat_rmse["fieldConfig"]["defaults"]["unit"] == "percent"

    painel_stat_mae = next((p for p in dados_dash["panels"] if p.get("id") == 23), None)
    assert painel_stat_mae is not None, "Painel 23 (stat redução MAE) não encontrado"
    assert painel_stat_mae["type"] == "stat"

    # 4. Valida Painel 24 (Gauge de R²)
    painel_gauge_r2 = next((p for p in dados_dash["panels"] if p.get("id") == 24), None)
    assert painel_gauge_r2 is not None, "Painel 24 (gauge R²) não encontrado"
    assert painel_gauge_r2["type"] == "gauge"

    # 5. Valida Painel 25 (Table com cellOptions gauge nativo)
    painel_tabela = next((p for p in dados_dash["panels"] if p.get("id") == 25), None)
    assert painel_tabela is not None, "Painel 25 (tabela comparativa) não encontrado"
    assert painel_tabela["type"] == "table"
    overrides_25 = painel_tabela["fieldConfig"]["overrides"]
    tem_gauge_celula = any(
        prop.get("id") == "custom.cellOptions" and prop.get("value", {}).get("type") == "gauge"
        for o in overrides_25
        for prop in o.get("properties", [])
    )
    assert tem_gauge_celula, "Célula gráfica com mini-gauge nativo ausente na tabela comparativa"

    # 6. Valida tags do dashboard
    assert "transformacao-alvo" in dados_dash["tags"]
    assert "treino-log-comparativo" in dados_dash["tags"]
