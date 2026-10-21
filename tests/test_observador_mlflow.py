from __future__ import annotations
import json
import os
import tempfile
import pandas as pd
from sklearn.linear_model import LinearRegression
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow, ObservadorMLflow
from app.contratos_dados.regra_negocio import DefinicaoRegra
from app.orquestracao_ml.modelo_servico import ModeloServico
from app.orquestracao_ml.observador_auditoria import ObservadorAuditoria
from app.orquestracao_ml.observador_metricas import ObservadorMetricas
from app.orquestracao_ml.publicador_servico import ErroPublicacaoMLflow, PublicadorServico

class ObservadorEspiao(ObservadorMLflow):
    """Observador espião utilizado para capturar e validar eventos nos testes unitários."""
    def __init__(self) -> None:
        self.eventos_recebidos: list[EventoPublicacaoMLflow] = []

    def notificar(self, evento: EventoPublicacaoMLflow) -> None:
        self.eventos_recebidos.append(evento)

def _criar_modelo_servico_teste() -> ModeloServico:
    x = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0]})
    y = [250000.0, 500000.0, 750000.0]
    modelo = LinearRegression().fit(x, y)
    regras = [
        DefinicaoRegra(
            nome="margem_minima",
            ativo=True,
            descricao="Minimo",
            destino="preco_minimo",
            expressao="preco_predito * 0.95"
        )
    ]
    return ModeloServico(modelo_campeao=modelo, regras=regras, coluna_alvo_predito="preco_predito")

def test_contrato_protocolo_observador() -> None:
    espiao = ObservadorEspiao()
    assert isinstance(espiao, ObservadorMLflow)

    auditoria = ObservadorAuditoria("storage-data/auditoria_test.jsonl")
    assert isinstance(auditoria, ObservadorMLflow)

    metricas = ObservadorMetricas("storage-data/metricas_test.prom")
    assert isinstance(metricas, ObservadorMLflow)

def test_adicionar_e_remover_observadores() -> None:
    publicador = PublicadorServico("http://localhost:5000")
    obs1 = ObservadorEspiao()
    obs2 = ObservadorEspiao()

    publicador.adicionar_observador(obs1)
    publicador.adicionar_observador(obs2)
    # Não adiciona duplicado
    publicador.adicionar_observador(obs1)
    assert len(publicador._observadores) == 2

    publicador.remover_observador(obs1)
    assert len(publicador._observadores) == 1
    assert obs2 in publicador._observadores

def test_notificacao_sucesso_com_multiplos_observadores() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        caminho_db = os.path.join(tmp_dir, "mlflow.db")
        tracking_uri = f"sqlite:///{caminho_db}"
        publicador = PublicadorServico(tracking_uri=tracking_uri)

        obs_espiao = ObservadorEspiao()
        caminho_jsonl = os.path.join(tmp_dir, "auditoria.jsonl")
        obs_auditoria = ObservadorAuditoria(caminho_arquivo=caminho_jsonl)
        caminho_prom = os.path.join(tmp_dir, "metricas.prom")
        obs_metricas = ObservadorMetricas(caminho_metricas_prom=caminho_prom)

        publicador.adicionar_observador(obs_espiao)
        publicador.adicionar_observador(obs_auditoria)
        publicador.adicionar_observador(obs_metricas)

        servico = _criar_modelo_servico_teste()
        resultado = publicador.publicar_modelo_servico(
            modelo_servico=servico,
            experimento="exp_observer_teste",
            nome_modelo_registrado="modelo_observer",
            tags={"modelo_campeao": "random_forest", "modo_execucao": "apenas_campeao"},
            parametros={"n_estimators": 100},
            metricas={"rmse": 15000.5}
        )

        assert resultado["status"] == "sucesso"

        # 1. Valida ObservadorEspiao
        assert len(obs_espiao.eventos_recebidos) == 1
        ev = obs_espiao.eventos_recebidos[0]
        assert ev.tipo_evento == "modelo_publicado"
        assert ev.status == "sucesso"
        assert ev.modelo_campeao == "random_forest"
        assert ev.modo_execucao == "apenas_campeao"
        assert ev.run_id == resultado["run_id"]

        # 2. Valida ObservadorAuditoria (persistência em JSONL)
        assert len(obs_auditoria.obter_historico()) == 1
        assert os.path.exists(caminho_jsonl)
        with open(caminho_jsonl, "r", encoding="utf-8") as f:
            linhas = f.readlines()
            assert len(linhas) == 1
            dado_lido = json.loads(linhas[0])
            assert dado_lido["tipo_evento"] == "modelo_publicado"
            assert dado_lido["modelo_campeao"] == "random_forest"
            assert dado_lido["run_id"] == resultado["run_id"]

        # 3. Valida ObservadorMetricas (exportação Prometheus)
        assert obs_metricas.total_sucessos == 1
        assert obs_metricas.total_falhas == 0
        assert os.path.exists(caminho_prom)
        with open(caminho_prom, "r", encoding="utf-8") as f:
            conteudo_prom = f.read()
            assert "mlflow_publicacao_sucesso_total 1" in conteudo_prom
            assert 'modelo="random_forest"' in conteudo_prom

def test_notificacao_falha_observadores() -> None:
    # Tracking URI com esquema inválido dispara falha e notifica observadores
    publicador = PublicadorServico(tracking_uri="invalido://schema:9999")
    obs_espiao = ObservadorEspiao()
    obs_metricas = ObservadorMetricas("storage-data/metricas_tmp_falha.prom")

    publicador.adicionar_observador(obs_espiao)
    publicador.adicionar_observador(obs_metricas)

    servico = _criar_modelo_servico_teste()

    try:
        publicador.publicar_modelo_servico(
            modelo_servico=servico,
            experimento="exp_falha",
            nome_modelo_registrado="modelo_falha",
            tags={"modelo_campeao": "ridge", "modo_execucao": "completo"}
        )
        assert False, "Deveria ter lançado ErroPublicacaoMLflow"
    except ErroPublicacaoMLflow:
        pass

    assert len(obs_espiao.eventos_recebidos) == 1
    ev_erro = obs_espiao.eventos_recebidos[0]
    assert ev_erro.tipo_evento == "falha_publicacao"
    assert ev_erro.status == "erro"
    assert ev_erro.modelo_campeao == "ridge"
    assert obs_metricas.total_falhas == 1
