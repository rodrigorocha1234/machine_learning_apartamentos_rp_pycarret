"""Testes para garantir que apenas os modelos com ativo: true no tuning são executados."""

from __future__ import annotations

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


def test_testador_friedman_um_modelo_promocao_direta() -> None:
    """Verifica que o TestadorFriedman promove diretamente quando há apenas 1 modelo."""
    testador = TestadorFriedman()
    matriz_cv = {"regressao_linear": [105000.0, 98000.0, 102000.0]}

    res = testador.comparar_modelos(matriz_cv=matriz_cv, metrica="RMSE")

    assert res.modelo_campeao_estatistico == "regressao_linear"
    assert res.candidatos_selecionados == ["regressao_linear"]
    assert res.friedman.diferenca_significativa is False
    assert res.nemenyi is None
    assert "Apenas o modelo 'regressao_linear' foi selecionado" in res.justificativa_decisao
