from __future__ import annotations
import os
import tempfile
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from app.contratos_dados.regra_negocio import DefinicaoRegra
from app.orquestracao_ml.modelo_servico import ModeloServico
from app.orquestracao_ml.publicador_servico import ErroPublicacaoMLflow, PublicadorServico

def test_publicador_mlflow_instanciacao() -> None:
    publicador = PublicadorServico(tracking_uri="http://localhost:5000")
    assert publicador._tracking_uri == "http://localhost:5000"

def test_publicador_mlflow_fallback_modelo_salvo(monkeypatch) -> None:
    # Com tracking URI que levanta erro de conexão, deve retornar o fallback seguro sem lançar exceção
    def falha_search(*args, **kwargs):
        raise ConnectionError("Falha simulada de conexao MLflow")

    monkeypatch.setattr(MlflowClient, "search_model_versions", falha_search)
    publicador = PublicadorServico(tracking_uri="http://127.0.0.1:5000")
    campeao = publicador.obter_modelo_campeao_salvo(
        nome_modelo_registrado="modelo_inexistente",
        fallback_padrao="random_forest"
    )
    assert campeao == "random_forest"

def test_publicador_mlflow_publicar_modelo_servico_local() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # MLflow 3.x com SQLite backend suporta Model Registry localmente
        caminho_db = os.path.join(tmp_dir, "mlflow.db")
        tracking_uri = f"sqlite:///{caminho_db}"
        publicador = PublicadorServico(tracking_uri=tracking_uri)

        # Monta modelo simples e regras
        x = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0]})
        y = [250000.0, 500000.0, 750000.0]
        modelo = LinearRegression().fit(x, y)

        regras = [
            DefinicaoRegra(
                nome="margem_inferior",
                ativo=True,
                descricao="Minimo",
                destino="preco_minimo",
                expressao="preco_predito * 0.95"
            )
        ]
        servico = ModeloServico(modelo_campeao=modelo, regras=regras, coluna_alvo_predito="preco_predito")

        resultado = publicador.publicar_modelo_servico(
            modelo_servico=servico,
            experimento="teste_publicacao_unitario",
            nome_modelo_registrado="modelo_teste_unitario",
            tags={"modelo_campeao": "linear_regression", "modo": "teste"},
            parametros={"alpha": 1.0}
        )

        assert resultado["status"] == "sucesso"
        assert "run_id" in resultado
        assert "models:/modelo_teste_unitario/latest" in resultado["model_uri"]

def test_publicador_mlflow_erro_tracking_invalido() -> None:
    # Tracking URI com esquema completamente absurdo/inválido deve lançar ErroPublicacaoMLflow
    publicador = PublicadorServico(tracking_uri="invalido://schema-nao-existente:99999")
    x = pd.DataFrame({"Metragem": [50.0]})
    y = [250000.0]
    modelo = LinearRegression().fit(x, y)
    servico = ModeloServico(modelo_campeao=modelo, regras=[], coluna_alvo_predito="preco_predito")

    try:
        publicador.publicar_modelo_servico(
            modelo_servico=servico,
            experimento="teste_falha",
            nome_modelo_registrado="modelo_falha"
        )
        assert False, "Deveria ter levantado ErroPublicacaoMLflow"
    except ErroPublicacaoMLflow:
        assert True
