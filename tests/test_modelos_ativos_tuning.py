"""Testes para garantir que apenas os modelos com ativo: true no tuning são executados."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.esquema_configuracao import ConfiguracaoModelagem, ItemModelo, ParametrosTuningModelo
from app.motores_modelagem.treinador_sklearn import TreinadorSklearn
from app.servicos_avaliacao.testador_friedman import TestadorFriedman


def test_obter_modelos_ativos_filtra_tuning_desativado() -> None:
    """Garante que modelos com ativo: false em tuning.parametros são excluídos."""
    conf = ConfYAML()
    config_mod = conf.obter_modelagem()

    # Na configuração atual, regressao_linear está ativo=true e lasso está ativo=false
    modelos_ativos = config_mod.obter_modelos_ativos()
    nomes_ativos = [m.nome for m in modelos_ativos]

    assert "regressao_linear" in nomes_ativos
    assert "lasso" not in nomes_ativos
    assert "ridge" not in nomes_ativos
    assert "lightgbm" not in nomes_ativos
    assert len(modelos_ativos) == 1


def test_treinador_sklearn_executa_somente_modelos_ativos() -> None:
    """Verifica que o TreinadorSklearn treina exclusivamente os modelos ativos."""
    conf = ConfYAML()
    config_mod = conf.obter_modelagem()

    # Carrega dados reais de staging
    from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
    repo = RepositorioSqlite(caminho="storage-data/staging_imoveis.db", tabela="imoveis_limpos")
    assert repo.verificar_existencia(), "Tabela 'imoveis_limpos' deve existir no staging."
    df_real = repo.ler_dados().head(50)

    treinador = TreinadorSklearn()
    res_tuning = treinador.treinar_e_ajustar(
        dados_treino=df_real,
        coluna_alvo="Valor_da_Venda",
        config=config_mod,
        folds=2,
    )

    nomes_treinados = [m.nome_modelo for m in res_tuning.modelos_treinados]
    assert nomes_treinados == ["regressao_linear"]
    assert "lasso" not in nomes_treinados
    assert "ridge" not in nomes_treinados
    assert res_tuning.modelo_campeao == "regressao_linear"


def test_avaliacao_um_modelo_promocao_direta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que o pipeline promove diretamente o único modelo ativo sem falhar no Friedman."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo, ResultadoTuning
    from app.servicos_avaliacao.orquestrador_avaliacao import OrquestradorAvaliacao

    conf = ConfYAML()
    orquestrador = OrquestradorAvaliacao.da_configuracao(conf)

    # Simula 1 único modelo vindo do tuning
    modelo_unico = ResultadoTreinoModelo(
        nome_modelo="regressao_linear",
        framework="sklearn",
        melhores_parametros={"fit_intercept": True},
        metricas={"RMSE": 105000.0},
        duracao_segundos=0.1,
    )
    res_tuning = ResultadoTuning(
        metodo_tuning="random",
        iteracoes=1,
        modelos_treinados=[modelo_unico],
        modelo_campeao="regressao_linear",
        melhor_rmse=105000.0,
    )

    from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite

    repo = RepositorioSqlite(caminho="storage-data/staging_imoveis.db", tabela="imoveis_limpos")
    df_real = repo.ler_dados().head(50)

    resultado_etapa = orquestrador.executar_avaliacao_e_selecao(
        dados=df_real,
        resultado_tuning=res_tuning,
    )

    assert resultado_etapa.modelo_final_recomendado == "regressao_linear"
    assert resultado_etapa.comparacao_estatistica.modelo_campeao_estatistico == "regressao_linear"
    assert resultado_etapa.comparacao_estatistica.friedman.diferenca_significativa is False
    assert resultado_etapa.comparacao_estatistica.nemenyi is None
