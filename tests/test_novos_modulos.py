from __future__ import annotations
import os
import tempfile
import pandas as pd
import pytest
from app.analises_eda.construtor_layout import ConstrutorLayout
from app.analises_eda.construtor_tabela import ConstrutorTabela
from app.contratos_dados.gerador_eda import ConfiguracaoAnaliseGrafica
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow
from app.motores_treino.motor_sklearn import MotorSklearn
from app.observabilidade_metricas.coletor_telemetria import ColetorTelemetria
from app.observabilidade_metricas.registrador_auditoria import RegistradorAuditoria
from app.orquestracao_ml.executor_avaliacao import ExecutorAvaliacao
from app.orquestracao_ml.montador_servico import MontadorServico
from app.orquestracao_ml.otimizador_campeao import OtimizadorCampeao
from app.orquestracao_ml.preparador_treino import PreparadorTreino


@pytest.fixture
def df_exemplo() -> pd.DataFrame:
    return pd.DataFrame({
        "Bairro": ["Centro", "Centro", "Jardim Paulista", "Jardim Paulista", "Ipiranga", "Ipiranga"],
        "Metragem": [50.0, 65.0, 110.0, 120.0, 80.0, 85.0],
        "Quartos": [1, 2, 3, 3, 2, 2],
        "Banheiros": [1, 1, 2, 2, 1, 2],
        "Valor_da_Venda": [200000.0, 250000.0, 500000.0, 550000.0, 320000.0, 340000.0],
    })


def test_construtor_tabela_agrupada(df_exemplo: pd.DataFrame) -> None:
    construtor = ConstrutorTabela()
    cfg = ConfiguracaoAnaliseGrafica(
        nome="resumo_bairro",
        tipo_grafico="tabela",
        titulo="Resumo por Bairro",
        agrupar_por="Bairro",
        coluna="Valor_da_Venda",
        metricas=["contagem", "media", "mediana"],
        limite=5,
    )
    fig = construtor.construir_tabela(dados=df_exemplo, configuracao=cfg, coluna_alvo="Valor_da_Venda")
    assert fig is not None
    assert fig.layout.title.text == "Resumo por Bairro"


def test_construtor_tabela_registros(df_exemplo: pd.DataFrame) -> None:
    construtor = ConstrutorTabela()
    cfg = ConfiguracaoAnaliseGrafica(
        nome="top_imoveis",
        tipo_grafico="tabela",
        titulo="Maiores Imóveis",
        colunas=["Bairro", "Metragem", "Valor_da_Venda"],
        ordenar_por="Metragem",
        direcao_ordenacao="desc",
        limite=3,
    )
    fig = construtor.construir_tabela(dados=df_exemplo, configuracao=cfg, coluna_alvo="Valor_da_Venda")
    assert fig is not None
    assert fig.layout.title.text == "Maiores Imóveis"


def test_construtor_layout() -> None:
    construtor = ConstrutorLayout()
    html = construtor.construir_documento(
        secoes_graficos=["<div>Grafico 1</div>", "<div>Grafico 2</div>"],
        total_amostras=100,
        coluna_alvo="Valor_da_Venda",
        total_analises=2,
    )
    assert "<!DOCTYPE html>" in html
    assert "Amostras: 100" in html
    assert "Gráficos Cadastrados no YAML: 2" in html
    assert "Grafico 1" in html


def test_coletor_telemetria() -> None:
    with tempfile.NamedTemporaryFile(suffix=".prom", delete=False) as tmp:
        caminho_prom = tmp.name

    try:
        coletor = ColetorTelemetria(caminho_metricas_prom=caminho_prom)
        evento_ok = EventoPublicacaoMLflow(
            tipo_evento="publicacao",
            status="sucesso",
            modelo_campeao="random_forest",
            modo_execucao="completo",
            experimento="imoveis_teste",
            nome_modelo_registrado="imoveis_rf",
            run_id="run_123",
            model_uri="runs:/run_123/model",
            mensagem="OK",
            timestamp_iso="2026-10-22T10:00:00Z",
        )
        coletor.notificar(evento_ok)
        assert coletor.total_sucessos == 1
        assert coletor.total_falhas == 0

        evento_erro = EventoPublicacaoMLflow(
            tipo_evento="publicacao",
            status="erro",
            modelo_campeao="random_forest",
            modo_execucao="completo",
            experimento="imoveis_teste",
            nome_modelo_registrado="imoveis_rf",
            run_id="run_124",
            model_uri=None,
            mensagem="Falha de conexão",
            timestamp_iso="2026-10-22T10:01:00Z",
        )
        coletor.notificar(evento_erro)
        assert coletor.total_sucessos == 1
        assert coletor.total_falhas == 1

        with open(caminho_prom, "r", encoding="utf-8") as f:
            conteudo = f.read()
        assert "mlflow_publicacao_sucesso_total 1" in conteudo
        assert "mlflow_publicacao_falha_total 1" in conteudo
    finally:
        if os.path.exists(caminho_prom):
            os.remove(caminho_prom)


def test_registrador_auditoria() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        caminho_jsonl = tmp.name

    try:
        registrador = RegistradorAuditoria(caminho_arquivo=caminho_jsonl)
        evento = EventoPublicacaoMLflow(
            tipo_evento="registro",
            status="sucesso",
            modelo_campeao="lightgbm",
            modo_execucao="completo",
            experimento="exp_rp",
            nome_modelo_registrado="modelo_lgbm",
            run_id="run_abc",
            model_uri="models:/modelo_lgbm/1",
            mensagem="Registrado com sucesso",
            timestamp_iso="2026-10-22T12:00:00Z",
        )
        registrador.notificar(evento)
        historico = registrador.obter_historico()
        assert len(historico) == 1
        assert historico[0].modelo_campeao == "lightgbm"

        with open(caminho_jsonl, "r", encoding="utf-8") as f:
            linhas = f.readlines()
        assert len(linhas) == 1
        assert "lightgbm" in linhas[0]
    finally:
        if os.path.exists(caminho_jsonl):
            os.remove(caminho_jsonl)


def test_preparador_treino_sklearn() -> None:
    preparador = PreparadorTreino()
    config = {
        "framework": {"motor": "sklearn"},
        "selecao": {
            "colunas": [
                {"nome": "Metragem", "papel": "numerica", "ativo": True},
                {"nome": "Bairro", "papel": "categorica", "ativo": True},
            ]
        },
        "preprocessamento": {
            "transformar_alvo": True,
            "metodo_transformacao_alvo": "log1p",
            "transformar_atributos": {
                "ativo": True,
                "metodo": "log1p",
                "colunas": ["Metragem"],
            },
        },
        "validacao": {"cv_interna": {"splits": 3, "metodo": "kfold", "embaralhar": True}},
        "modelos": [{"nome": "random_forest", "ativo": True}],
    }
    motor, nome, transf_alvo, met_alvo, transf_attr, met_attr, cols_attr = preparador.criar_motor(config, semente=42)
    assert isinstance(motor, MotorSklearn)
    assert nome == "sklearn"
    assert transf_alvo is True
    assert met_alvo == "log1p"
    assert transf_attr is True
    assert met_attr == "log1p"
    assert cols_attr == ["Metragem"]


def test_montador_servico(df_exemplo: pd.DataFrame) -> None:
    montador = MontadorServico()
    motor = MotorSklearn(
        semente=42,
        folds=2,
        colunas_numericas=["Metragem", "Quartos", "Banheiros"],
        colunas_categoricas=["Bairro"],
        modelos_configurados=[{"nome": "random_forest", "ativo": True}],
    )
    modelo_treinado = motor.treinar_final(df_exemplo, alvo="Valor_da_Venda", modelo_nome="random_forest")

    config_regras = {
        "coluna_alvo_predito": "preco_predito",
        "regras": [
            {
                "nome": "margem_seguranca",
                "ativo": True,
                "descricao": "Adiciona margem de 5%",
                "destino": "preco_com_margem",
                "expressao": "preco_predito * 1.05",
            }
        ],
    }

    servico = montador.montar_servico(modelo_final=modelo_treinado, config_regras=config_regras)
    predicoes = servico.predict(context=None, model_input=df_exemplo)
    assert "preco_predito" in predicoes.columns
    assert "preco_com_margem" in predicoes.columns
    assert (predicoes["preco_com_margem"] > predicoes["preco_predito"]).all()


def test_executor_avaliacao_e_otimizador_campeao(df_exemplo: pd.DataFrame) -> None:
    executor = ExecutorAvaliacao()
    otimizador = OtimizadorCampeao()

    motor = MotorSklearn(
        semente=42,
        folds=3,
        colunas_numericas=["Metragem", "Quartos", "Banheiros"],
        colunas_categoricas=["Bairro"],
        modelos_configurados=[
            {"nome": "regressao_linear", "ativo": True},
            {"nome": "ridge", "ativo": True},
            {"nome": "random_forest", "ativo": True},
        ],
    )

    avaliacao = executor.executar_avaliacao(
        motor_treino=motor,
        df_dev=df_exemplo,
        alvo="Valor_da_Venda",
        config_estatistica={"alpha": 0.05},
    )

    assert avaliacao.modelo_campeao in ["regressao_linear", "ridge", "random_forest"]
    assert "estatistica_friedman" in avaliacao.resumo_estatistico
    assert "postos_medios" in avaliacao.resumo_estatistico

    # Testa otimização de campeão individual
    otimizacao = otimizador.otimizar_ou_combinar(
        motor_treino=motor,
        df_dev=df_exemplo,
        dados_staging=df_exemplo,
        alvo="Valor_da_Venda",
        modelo_campeao_nome=avaliacao.modelo_campeao,
        config_tuning={"iteracoes": 2, "metodo": "random"},
        config_votacao={"ativo": False},
        resultado_estatistico=avaliacao.resultado_estatistico,
        perdas_pareadas=avaliacao.perdas_pareadas,
        cv_folds=3,
    )

    assert otimizacao.modelo_final is not None
    assert otimizacao.modelo_campeao_nome == avaliacao.modelo_campeao

    # Testa criação de ensemble de votação
    otimizacao_ens = otimizador.otimizar_ou_combinar(
        motor_treino=motor,
        df_dev=df_exemplo,
        dados_staging=df_exemplo,
        alvo="Valor_da_Venda",
        modelo_campeao_nome=avaliacao.modelo_campeao,
        config_tuning={},
        config_votacao={"ativo": True, "tecnica": "media_simples", "melhores_n": 2},
        resultado_estatistico=avaliacao.resultado_estatistico,
        perdas_pareadas=avaliacao.perdas_pareadas,
        cv_folds=3,
    )

    assert otimizacao_ens.modelo_final is not None
    assert "ensemble" in otimizacao_ens.modelo_campeao_nome

