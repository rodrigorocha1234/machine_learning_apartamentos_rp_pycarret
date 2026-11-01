"""Testes automatizados para TreinadorSklearn com pipelines nativos."""

from __future__ import annotations

import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.contratos_modelagem.motor_treinador import ResultadoTuning
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.treinador_sklearn import ErroTreinadorSklearn, TreinadorSklearn


@pytest.fixture
def dados_sinteticos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area_Privativa": [50.0, 75.0, 100.0, 120.0, 45.0, 80.0, 110.0, 95.0, None, 60.0],
            "Quartos": [1, 2, 3, 3, 1, 2, 3, 2, 2, 1],
            "Bairro": [
                "Centro",
                "Sul",
                "Sul",
                "Norte",
                "Centro",
                "Leste",
                "Sul",
                "Norte",
                "Centro",
                None,
            ],
            "Valor_da_Venda": [
                200000.0,
                350000.0,
                500000.0,
                550000.0,
                180000.0,
                380000.0,
                520000.0,
                460000.0,
                290000.0,
                230000.0,
            ],
        }
    )


def test_sklearn_validacao_dataframe_vazio(dados_sinteticos: pd.DataFrame) -> None:
    _ = dados_sinteticos
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()

    with pytest.raises(ErroTreinadorSklearn, match="está vazio"):
        treinador.treinar_e_ajustar(pd.DataFrame(), "Valor_da_Venda", config_modelagem)


def test_sklearn_validacao_coluna_alvo_ausente(dados_sinteticos: pd.DataFrame) -> None:
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_modelagem = conf.obter_modelagem()

    with pytest.raises(ErroTreinadorSklearn, match="Coluna alvo 'Alvo_Inexistente' ausente"):
        treinador.treinar_e_ajustar(dados_sinteticos, "Alvo_Inexistente", config_modelagem)


def test_sklearn_treinamento_e_tuning_rapido(dados_sinteticos: pd.DataFrame) -> None:
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    # Ativa apenas ridge e random_forest para teste rápido
    for modelo in config_dict["modelos"]:
        if modelo["nome"] in ("ridge", "random_forest"):
            modelo["ativo"] = True
            if modelo["nome"] in config_dict["tuning"]["parametros"]:
                config_dict["tuning"]["parametros"][modelo["nome"]]["ativo"] = True
        else:
            modelo["ativo"] = False
    config_dict["tuning"]["iteracoes"] = 2
    config_rapida = ConfiguracaoModelagem.model_validate(config_dict)

    resultado = treinador.treinar_e_ajustar(
        dados_treino=dados_sinteticos,
        coluna_alvo="Valor_da_Venda",
        config=config_rapida,
        folds=2,
    )

    assert isinstance(resultado, ResultadoTuning)
    assert len(resultado.modelos_treinados) == 2
    assert resultado.modelo_campeao in ("ridge", "random_forest")
    assert resultado.melhor_rmse > 0.0

    for modelo_res in resultado.modelos_treinados:
        assert modelo_res.framework == "sklearn"
        assert "RMSE" in modelo_res.metricas
        assert "MAE" in modelo_res.metricas
        assert modelo_res.metricas["RMSE"] > 0


def test_sklearn_treinamento_com_grid_e_metrica_customizada(
    dados_sinteticos: pd.DataFrame,
) -> None:
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config_dict = conf.obter_modelagem().model_dump()
    for modelo in config_dict["modelos"]:
        modelo["ativo"] = modelo["nome"] == "ridge"

    config_dict["tuning"]["metodo"] = "grid"
    config_dict["tuning"]["otimizar_metrica"] = "MAE"
    config_dict["tuning"]["manter_melhor"] = True
    config_dict["tuning"]["parametros"] = {"ridge": {"alpha": [0.1, 1.0]}}
    config_rapida = ConfiguracaoModelagem.model_validate(config_dict)

    resultado = treinador.treinar_e_ajustar(
        dados_treino=dados_sinteticos,
        coluna_alvo="Valor_da_Venda",
        config=config_rapida,
        folds=2,
    )

    assert isinstance(resultado, ResultadoTuning)
    assert len(resultado.modelos_treinados) == 1
    assert resultado.modelo_campeao == "ridge"
    assert "MAE" in resultado.modelos_treinados[0].metricas


def test_sklearn_escolha_tipo_preprocessador_por_modelo(dados_sinteticos: pd.DataFrame) -> None:
    """Verifica que o pré-processador e escalador são instanciados conforme a escolha no modelo."""
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
    treinador = TreinadorSklearn()
    conf = ConfYAML()
    config = conf.obter_modelagem()

    # 1. Teste de mapeamento do método _obter_escalador
    assert isinstance(treinador._obter_escalador("standard"), StandardScaler)
    assert isinstance(treinador._obter_escalador("robust"), RobustScaler)
    assert isinstance(treinador._obter_escalador("minmax"), MinMaxScaler)
    assert isinstance(treinador._obter_escalador("maxabs"), MaxAbsScaler)
    assert treinador._obter_escalador("nenhum") == "passthrough"
    assert treinador._obter_escalador("passthrough") == "passthrough"

    # 2. Teste de construção de ColumnTransformer com scaler robusto
    ct_robust = treinador._construir_preprocessador(
        cols_numericas=["Area_Privativa", "Quartos"],
        cols_categoricas=["Bairro"],
        config=config,
        tipo_preprocessador="robust",
    )
    pipe_num_robust = ct_robust.transformers[0][1]
    nomes_passos_robust = [nome for nome, _ in pipe_num_robust.steps]
    assert "scaler" in nomes_passos_robust
    assert isinstance(pipe_num_robust.named_steps["scaler"], RobustScaler)

    # 3. Teste de construção com 'nenhum' (sem scaler, ideal para árvores)
    ct_nenhum = treinador._construir_preprocessador(
        cols_numericas=["Area_Privativa", "Quartos"],
        cols_categoricas=["Bairro"],
        config=config,
        tipo_preprocessador="nenhum",
    )
    pipe_num_nenhum = ct_nenhum.transformers[0][1]
    nomes_passos_nenhum = [nome for nome, _ in pipe_num_nenhum.steps]
    assert "scaler" not in nomes_passos_nenhum
    assert "imputer" in nomes_passos_nenhum

    # 4. Modelos de árvore adotam 'nenhum' por padrão quando aplicável
    item_rf = config.obter_item_modelo("random_forest")
    assert item_rf is not None
    assert item_rf.obter_preprocessador() == "nenhum"

    # 5. Modelos lineares adotam 'standard' por padrão
    item_lr = config.obter_item_modelo("regressao_linear")
    assert item_lr is not None
    assert item_lr.obter_preprocessador() == "standard"


def test_sklearn_instanciar_estimador_sem_parametros_fixos() -> None:
    """Verifica que o instanciador cria os estimadores puros, sem parâmetros hardcoded."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    import lightgbm as lgb
    import xgboost as xgb

    treinador = TreinadorSklearn()

    # Modelos lineares e clássicos
    lr = treinador._instanciar_estimador("regressao_linear")
    assert isinstance(lr, LinearRegression)

    lasso = treinador._instanciar_estimador("lasso")
    assert isinstance(lasso, Lasso)

    ridge = treinador._instanciar_estimador("ridge")
    assert isinstance(ridge, Ridge)

    en = treinador._instanciar_estimador("elasticnet")
    assert isinstance(en, ElasticNet)

    # arvore_decisao deve instanciar DecisionTreeRegressor, não RandomForest
    dt = treinador._instanciar_estimador("arvore_decisao")
    assert isinstance(dt, DecisionTreeRegressor)

    # random_forest sem n_estimators hardcoded
    rf = treinador._instanciar_estimador("random_forest")
    assert isinstance(rf, RandomForestRegressor)

    svr = treinador._instanciar_estimador("svr")
    assert isinstance(svr, SVR)

    mlp = treinador._instanciar_estimador("rede_neural")
    assert isinstance(mlp, MLPRegressor)

    xg = treinador._instanciar_estimador("xgboost")
    assert isinstance(xg, xgb.XGBRegressor)

    lg = treinador._instanciar_estimador("lightgbm")
    assert isinstance(lg, lgb.LGBMRegressor)


def test_sklearn_instanciar_estimador_com_parametros_dinamicos() -> None:
    """Verifica que parâmetros vindos do YAML/dicionário são injetados no estimador."""
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    treinador = TreinadorSklearn()

    dt = treinador._instanciar_estimador("arvore_decisao", {"max_depth": 7, "min_samples_split": 4})
    assert isinstance(dt, DecisionTreeRegressor)
    assert dt.max_depth == 7
    assert dt.min_samples_split == 4

    rf = treinador._instanciar_estimador("random_forest", {"n_estimators": 25, "max_depth": 4})
    assert isinstance(rf, RandomForestRegressor)
    assert rf.n_estimators == 25
    assert rf.max_depth == 4

    with pytest.raises(ErroTreinadorSklearn, match="Modelo não suportado"):
        treinador._instanciar_estimador("modelo_inexistente")



