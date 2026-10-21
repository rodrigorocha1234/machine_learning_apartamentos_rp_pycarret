from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from app.analises_estatisticas.importancia_atributos import (
    AtributoImportancia,
    ImportanciaAtributos,
    RelatorioImportancia,
)


def _criar_dados_amostra() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    n = 100
    metragem = np.random.uniform(40, 200, n)
    bairro_cod = np.random.uniform(1, 5, n)
    zona_cod = np.random.uniform(1, 4, n)
    quartos = np.random.choice([1, 2, 3, 4], n)

    # Preço com forte dependência de Metragem e Bairro
    preco = 5000.0 * metragem + 20000.0 * bairro_cod + 8000.0 * zona_cod + 3000.0 * quartos + np.random.normal(0, 10000, n)

    df_x = pd.DataFrame({
        "Metragem": metragem,
        "Bairro": bairro_cod,
        "Zona": zona_cod,
        "Quartos": quartos,
    })
    s_y = pd.Series(preco, name="Valor_da_Venda")
    return df_x, s_y


def test_importancia_atributos_estimador_arvore() -> None:
    x, y = _criar_dados_amostra()
    arvore = DecisionTreeRegressor(max_depth=4, random_state=42)
    arvore.fit(x, y)

    analisador = ImportanciaAtributos(semente=42)
    relatorio: RelatorioImportancia = analisador.calcular_importancia_estimador(
        estimador=arvore,
        x=x,
        y=y,
        modelo_nome="arvore_decisao",
        motor="sklearn",
    )

    assert relatorio.modelo == "arvore_decisao"
    assert relatorio.motor == "sklearn"
    assert relatorio.metodo == "tree_feature_importances"
    assert len(relatorio.atributos) == 4

    # Valida soma normalizada
    soma_importancia = sum(a.importancia for a in relatorio.atributos)
    assert 0.99 <= soma_importancia <= 1.01

    # Valida ordenação estritamente decrescente
    importancias = [a.importancia for a in relatorio.atributos]
    assert importancias == sorted(importancias, reverse=True)
    assert relatorio.atributo_mais_importante in ["Metragem", "Bairro"]


def test_importancia_atributos_estimador_linear() -> None:
    x, y = _criar_dados_amostra()
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y)

    analisador = ImportanciaAtributos(semente=42)
    relatorio: RelatorioImportancia = analisador.calcular_importancia_estimador(
        estimador=ridge,
        x=x,
        y=y,
        modelo_nome="ridge",
        motor="sklearn",
    )

    assert relatorio.modelo == "ridge"
    assert relatorio.metodo == "linear_coefficients"
    assert len(relatorio.atributos) == 4

    soma_percentuais = sum(a.percentual for a in relatorio.atributos)
    assert 99.0 <= soma_percentuais <= 101.0


def test_importancia_atributos_simulada_motores_e_campeao() -> None:
    analisador = ImportanciaAtributos(semente=42)

    # 1. Trilha Sklearn com modelo individual
    rel_rf = analisador.gerar_importancia_simulada(modelo_nome="random_forest", motor="sklearn")
    assert rel_rf.modelo == "random_forest"
    assert rel_rf.motor == "sklearn"
    assert rel_rf.atributo_mais_importante == "Metragem"
    assert rel_rf.atributos[0].percentual > rel_rf.atributos[-1].percentual

    # 2. Trilha PyCaret com modelo campeão
    rel_campeao = analisador.gerar_importancia_simulada(modelo_nome="modelo_campeao", motor="pycaret")
    assert rel_campeao.modelo == "modelo_campeao"
    assert rel_campeao.motor == "pycaret"
    assert rel_campeao.metodo == "campeao_consensus_importance"
    assert len(rel_campeao.atributos) == 9

    # 3. Modelos lineares
    rel_lasso = analisador.gerar_importancia_simulada(modelo_nome="lasso", motor="sklearn")
    assert rel_lasso.metodo == "linear_coefficients"

    # 4. Serialização para formato de API/Dicionário
    dados_dict = rel_campeao.para_dicionario()
    assert dados_dict["modelo"] == "modelo_campeao"
    assert isinstance(dados_dict["atributos"], list)


def test_dashboard_grafana_recursos_nativos_feature_importance() -> None:
    caminho_dashboard = "config_ob/dashboards/dashboard_monitoramento_ml.json"
    assert os.path.exists(caminho_dashboard), f"Dashboard não encontrado em {caminho_dashboard}"

    with open(caminho_dashboard, "r", encoding="utf-8") as f:
        dados_dash = json.load(f)

    # 1. Valida existência da Row 4.3 de Feature Importance
    titulos_rows = [p["title"] for p in dados_dash["panels"] if p.get("type") == "row"]
    assert any("4.3 – Explicabilidade e Importância de Atributos" in t for t in titulos_rows)

    # 2. Valida painel Barchart nativo horizontal sem Matplotlib
    painel_barchart = next((p for p in dados_dash["panels"] if p.get("id") == 17), None)
    assert painel_barchart is not None, "Painel 17 (barchart) não encontrado"
    assert painel_barchart["type"] == "barchart"
    assert painel_barchart["options"]["orientation"] == "horizontal"
    assert "$modelo" in painel_barchart["title"]
    assert "$motor" in painel_barchart["title"]

    # 3. Valida painel Bar Gauge nativo
    painel_gauge = next((p for p in dados_dash["panels"] if p.get("id") == 18), None)
    assert painel_gauge is not None, "Painel 18 (bargauge) não encontrado"
    assert painel_gauge["type"] == "bargauge"
    assert painel_gauge["options"]["displayMode"] == "gradient"

    # 4. Valida painel Stat para atributo dominante
    painel_stat = next((p for p in dados_dash["panels"] if p.get("id") == 19), None)
    assert painel_stat is not None, "Painel 19 (stat) não encontrado"
    assert painel_stat["type"] == "stat"

    # 5. Valida tabela nativa com mini-gauge na célula (sem matplotlib)
    painel_tabela = next((p for p in dados_dash["panels"] if p.get("id") == 20), None)
    assert painel_tabela is not None, "Painel 20 (table) não encontrado"
    assert painel_tabela["type"] == "table"
    overrides = painel_tabela["fieldConfig"]["overrides"]
    tem_gauge_na_celula = any(
        prop.get("id") == "custom.cellOptions" and prop.get("value", {}).get("type") == "gauge"
        for o in overrides
        for prop in o.get("properties", [])
    )
    assert tem_gauge_na_celula, "Célula gráfica de gauge nativo ausente na tabela"

    # 6. Valida que as tags incluem feature-importance
    assert "feature-importance" in dados_dash["tags"]
