from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from app.analises_estatisticas.diagnostico_curva import DiagnosticoCurva

def test_diagnostico_curva_simulada() -> None:
    diagnostico = DiagnosticoCurva(semente=42)
    relatorio = diagnostico.gerar_curva_simulada(
        modelo_nome="random_forest",
        motor="sklearn",
        valores_complexidade=[2.0, 4.0, 8.0, 16.0, 32.0, 48.0],
        rmse_base=28000.0
    )

    assert relatorio.modelo == "random_forest"
    assert relatorio.motor == "sklearn"
    assert len(relatorio.pontos) == 6
    assert relatorio.melhor_complexidade in [2.0, 4.0, 8.0, 16.0, 32.0, 48.0]
    assert relatorio.menor_rmse_validacao > 0

    # Valida que o gap de variância é a diferença entre validação e treino
    for p in relatorio.pontos:
        assert p.gap_variancia >= 0
        assert np.isclose(p.gap_variancia, round(max(0.0, p.rmse_validacao - p.rmse_treino), 2), atol=0.05)

def test_diagnostico_curva_classificacao_regime() -> None:
    diagnostico = DiagnosticoCurva()

    # Gap pequeno: boa generalização
    regime_otimo = diagnostico.classificar_regime(rmse_treino=20000.0, rmse_validacao=22000.0)
    assert regime_otimo == "GENERALIZACAO_OTIMA"

    # Gap grande: sobreajuste / overfitting (alta variância)
    regime_overfit = diagnostico.classificar_regime(rmse_treino=15000.0, rmse_validacao=30000.0)
    assert regime_overfit == "OVERFITTING"

def test_diagnostico_estimador_real() -> None:
    np.random.seed(42)
    X = pd.DataFrame({
        "x1": np.linspace(1, 100, 40),
        "x2": np.random.randn(40) * 10
    })
    y = pd.Series(X["x1"] * 3.5 + np.random.randn(40) * 5)

    estimador = DecisionTreeRegressor(random_state=42)
    diagnostico = DiagnosticoCurva(semente=42, cv_splits=2)

    relatorio = diagnostico.diagnosticar_estimador(
        modelo_nome="arvore_decisao",
        estimador=estimador,
        x=X,
        y=y,
        passos_complexidade=[2, 4, 6],
        motor="sklearn"
    )

    assert relatorio.modelo == "arvore_decisao"
    assert len(relatorio.pontos) == 3
    assert relatorio.melhor_complexidade in [2.0, 4.0, 6.0]
    assert relatorio.menor_rmse_validacao > 0
    assert relatorio.diagnostico_regime in ["GENERALIZACAO_OTIMA", "OVERFITTING", "UNDERFITTING"]

    # Converte para dicionário JSON-serializável
    dados_dict = relatorio.para_dicionario()
    assert "pontos" in dados_dict
    assert isinstance(dados_dict["pontos"], list)

def test_diagnostico_curva_motores_e_campeao() -> None:
    diagnostico = DiagnosticoCurva()

    # Valida trilha Sklearn
    rel_sklearn = diagnostico.gerar_curva_simulada(modelo_nome="modelo_campeao", motor="sklearn")
    assert rel_sklearn.motor == "sklearn"
    assert rel_sklearn.modelo == "modelo_campeao"
    assert len(rel_sklearn.pontos) > 0

    # Valida trilha PyCaret
    rel_pycaret = diagnostico.gerar_curva_simulada(modelo_nome="modelo_campeao", motor="pycaret")
    assert rel_pycaret.motor == "pycaret"
    assert rel_pycaret.modelo == "modelo_campeao"
    assert len(rel_pycaret.pontos) > 0

    # Valida modelos individuais
    for mod in ["random_forest", "xgboost", "ridge", "lasso"]:
        rel = diagnostico.gerar_curva_simulada(modelo_nome=mod, motor="sklearn")
        assert rel.modelo == mod
        assert rel.menor_rmse_validacao > 0

def test_diagnostico_curva_parametro_complexidade_por_modelo() -> None:
    diagnostico = DiagnosticoCurva()

    # 1. Modelos baseados em árvore devem adotar 'max_depth'
    rel_rf = diagnostico.gerar_curva_simulada(modelo_nome="random_forest")
    assert rel_rf.parametro_complexidade == "max_depth"

    rel_dt = diagnostico.gerar_curva_simulada(modelo_nome="arvore_decisao")
    assert rel_dt.parametro_complexidade == "max_depth"

    rel_xgb = diagnostico.gerar_curva_simulada(modelo_nome="xgboost")
    assert rel_xgb.parametro_complexidade == "max_depth"

    # 2. Modelos lineares penalizados devem adotar 'alpha'
    rel_ridge = diagnostico.gerar_curva_simulada(modelo_nome="ridge")
    assert rel_ridge.parametro_complexidade == "alpha"

    rel_lasso = diagnostico.gerar_curva_simulada(modelo_nome="lasso")
    assert rel_lasso.parametro_complexidade == "alpha"

    rel_elastic = diagnostico.gerar_curva_simulada(modelo_nome="elasticnet")
    assert rel_elastic.parametro_complexidade == "alpha"

    # 3. Modelos SVR e Redes Neurais
    rel_svr = diagnostico.gerar_curva_simulada(modelo_nome="svr")
    assert rel_svr.parametro_complexidade == "C"

    rel_mlp = diagnostico.gerar_curva_simulada(modelo_nome="rede_neural")
    assert rel_mlp.parametro_complexidade == "hidden_layer_sizes"

    # 4. Parâmetro customizado explicitamente informado deve ter precedência
    rel_custom = diagnostico.gerar_curva_simulada(
        modelo_nome="random_forest",
        parametro_complexidade="n_estimators"
    )
    assert rel_custom.parametro_complexidade == "n_estimators"

    # 5. Avaliação com estimador real Ridge variando alpha
    from sklearn.linear_model import Ridge
    X = pd.DataFrame({"x": np.linspace(1, 10, 20)})
    y = pd.Series(X["x"] * 2.0)
    estimador_ridge = Ridge(alpha=1.0)
    rel_estimador_ridge = diagnostico.diagnosticar_estimador(
        modelo_nome="ridge",
        estimador=estimador_ridge,
        x=X,
        y=y,
        passos_complexidade=[0.1, 1.0, 10.0]
    )
    assert rel_estimador_ridge.parametro_complexidade == "alpha"
    assert len(rel_estimador_ridge.pontos) == 3

def test_diagnostico_curva_leitura_arquivo_yaml(tmp_path: Path) -> None:
    # 1. Valida leitura direta do arquivo padrão configuracoes/projeto.yaml
    diagnostico = DiagnosticoCurva(caminho_yaml="configuracoes/projeto.yaml")
    mapa = diagnostico.mapa_parametros
    assert isinstance(mapa, dict)
    assert mapa["random_forest"] == "max_depth"
    assert mapa["ridge"] == "alpha"
    assert mapa["svr"] == "C"
    assert "modelo_campeao" in mapa

    # 2. Valida método utilitário de classe carregar_mapa_yaml
    mapa_carregado = DiagnosticoCurva.carregar_mapa_yaml("configuracoes/projeto.yaml")
    assert mapa_carregado["xgboost"] == "max_depth"
    assert mapa_carregado["lasso"] == "alpha"

    # 3. Valida customização via arquivo YAML dinâmico
    caminho_temp = str(tmp_path / "custom_projeto.yaml")
    with open(caminho_temp, "w", encoding="utf-8") as f:
        f.write("""
diagnostico_curva:
  ativo: true
  parametros_complexidade:
    random_forest: n_estimators
    ridge: l2_penalty
    modelo_campeao: hiperparametro_vencedor
""")

    diag_custom = DiagnosticoCurva(caminho_yaml=caminho_temp)
    assert diag_custom.obter_parametro_complexidade("random_forest") == "n_estimators"
    assert diag_custom.obter_parametro_complexidade("ridge") == "l2_penalty"
    assert diag_custom.obter_parametro_complexidade("modelo_campeao") == "hiperparametro_vencedor"

    rel = diag_custom.gerar_curva_simulada(modelo_nome="random_forest")
    assert rel.parametro_complexidade == "n_estimators"

def test_dashboard_grafana_recursos_nativos_overfitting_underfitting() -> None:
    import json
    import os

    caminho_dashboard = "config_ob/dashboards/dashboard_monitoramento_ml.json"
    assert os.path.exists(caminho_dashboard), f"Dashboard não encontrado em {caminho_dashboard}"

    with open(caminho_dashboard, "r", encoding="utf-8") as f:
        dados_dash = json.load(f)

    # 1. Valida variáveis de template do Grafana ($motor e $modelo)
    assert "templating" in dados_dash
    variaveis = {v["name"]: v for v in dados_dash["templating"]["list"]}
    assert "motor" in variaveis, "Variável de template $motor ausente no Grafana"
    assert "modelo" in variaveis, "Variável de template $modelo ausente no Grafana"
    assert "sklearn" in variaveis["motor"]["query"]
    assert "pycaret" in variaveis["motor"]["query"]
    assert "modelo_campeao" in variaveis["modelo"]["query"]

    # 2. Valida existência da Seção 4.2
    titulos_painel = [p.get("title", "") for p in dados_dash.get("panels", [])]
    assert any("4.2 – Análise Overfitting vs Underfitting" in t for t in titulos_painel), \
        "Seção 4.2 não encontrada no dashboard"

    # 3. Valida painel nativo de curva de viés x variância
    painel_curva = None
    for p in dados_dash.get("panels", []):
        if "Diagnóstico de Overfitting vs Underfitting" in p.get("title", ""):
            painel_curva = p
            break

    assert painel_curva is not None, "Painel de curva de diagnóstico ausente"
    assert painel_curva.get("type") in ["timeseries", "xychart"], "O painel deve ser um gráfico nativo do Grafana"

    # 4. Valida preenchimento de área entre as curvas (Gap de Variância nativo)
    overrides = painel_curva.get("fieldConfig", {}).get("overrides", [])
    nomes_series = [o.get("matcher", {}).get("options") for o in overrides]
    assert "RMSE Treino (Viés)" in nomes_series
    assert "RMSE Validação (Generalização)" in nomes_series

    # Confirma que há configuração fillBelowTo nativa do Grafana
    tem_fill_below = any(
        any(prop.get("id") == "custom.fillBelowTo" for prop in o.get("properties", []))
        for o in overrides
    )
    assert tem_fill_below, "Painel deve usar recurso nativo custom.fillBelowTo para a área de gap de variância"

def test_diagnostico_curva_parametro_dinamico_modelo_campeao(tmp_path: Path) -> None:
    """Valida que o hiperparâmetro do modelo campeão varia dinamicamente conforme o campeão eleito."""
    diagnostico = DiagnosticoCurva()

    # 1. Quando o modelo campeão for Rede Neural / MLP
    param_nn = diagnostico.obter_parametro_complexidade(
        modelo_nome="modelo_campeao",
        modelo_referencia="rede_neural"
    )
    assert param_nn == "hidden_layer_sizes (campeão)"

    # 2. Quando o modelo campeão for Ridge ou Lasso (modelos lineares penalizados)
    param_ridge = diagnostico.obter_parametro_complexidade(
        modelo_nome="modelo_campeao",
        modelo_referencia="ridge"
    )
    assert param_ridge == "alpha (campeão)"

    param_lasso = diagnostico.obter_parametro_complexidade(
        modelo_nome="modelo_campeao",
        modelo_referencia="lasso"
    )
    assert param_lasso == "alpha (campeão)"

    # 3. Quando o modelo campeão for SVR / SVM
    param_svr = diagnostico.obter_parametro_complexidade(
        modelo_nome="modelo_campeao",
        modelo_referencia="svr"
    )
    assert param_svr == "C (campeão)"

    # 4. Quando o modelo campeão for Random Forest / XGBoost / Árvore
    param_rf = diagnostico.obter_parametro_complexidade(
        modelo_nome="modelo_campeao",
        modelo_referencia="random_forest"
    )
    assert param_rf == "max_depth (campeão)"

    # 5. Validação via nome composto com dois pontos
    param_composto = diagnostico.obter_parametro_complexidade(modelo_nome="modelo_campeao:rede_neural")
    assert param_composto == "hidden_layer_sizes (campeão)"

    # 6. Mapeamento para Scikit-Learn
    assert diagnostico.mapear_parametro_sklearn("modelo_campeao", modelo_referencia="rede_neural") == "max_iter"
    assert diagnostico.mapear_parametro_sklearn("modelo_campeao", modelo_referencia="ridge") == "alpha"
    assert diagnostico.mapear_parametro_sklearn("modelo_campeao", modelo_referencia="svr") == "C"
    assert diagnostico.mapear_parametro_sklearn("modelo_campeao", modelo_referencia="random_forest") == "max_depth"

    # 7. Geração de curva simulada para modelo campeão dinâmico
    rel_nn = diagnostico.gerar_curva_simulada(
        modelo_nome="modelo_campeao",
        motor="sklearn",
        modelo_referencia="rede_neural"
    )
    assert rel_nn.modelo == "modelo_campeao"
    assert rel_nn.parametro_complexidade == "hidden_layer_sizes (campeão)"

    rel_ridge = diagnostico.gerar_curva_simulada(
        modelo_nome="modelo_campeao",
        motor="sklearn",
        modelo_referencia="ridge"
    )
    assert rel_ridge.parametro_complexidade == "alpha (campeão)"

    # 8. Validação de leitura dinâmica de arquivo YAML onde o campeão padrão é rede neural
    caminho_yaml_nn = str(tmp_path / "yaml_campeao_nn.yaml")
    with open(caminho_yaml_nn, "w", encoding="utf-8") as f:
        f.write("""
pipeline:
  modelo_campeao_padrao: rede_neural
diagnostico_curva:
  ativo: true
  parametros_complexidade:
    rede_neural: hidden_layer_sizes
    random_forest: max_depth
    modelo_campeao: "auto"
""")

    diag_yaml_nn = DiagnosticoCurva(caminho_yaml=caminho_yaml_nn)
    # Sem informar modelo_referencia, deve detectar automaticamente do YAML que o campeão é rede_neural
    assert diag_yaml_nn.obter_parametro_complexidade("modelo_campeao") == "hidden_layer_sizes (campeão)"
