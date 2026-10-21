import os
import tempfile
import yaml
from app.orquestracao_ml.main_pipeline import MainPipeline

def test_execucao_main_pipeline_end_to_end() -> None:
    os.environ["IMOVEIS_XLSX_PATH"] = "storage-data/imoveis_apartamentos_rp.xlsx"
    os.environ["STAGING_DATABASE_URL"] = "storage-data/staging_imoveis.db"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["GRAFANA_URL"] = "http://localhost:3000"

    pipeline = MainPipeline("configuracoes/projeto.yaml")
    resultado = pipeline.executar_pipeline()

    assert resultado["status"] == "sucesso"
    assert resultado["linhas_processadas"] > 0
    assert "estatistica" in resultado
    estat = resultado["estatistica"]
    assert isinstance(estat, dict)
    assert "modelo_campeao" in estat
    assert resultado["servico_mlflow_pronto"] is True

def test_execucao_main_pipeline_com_motor_polars() -> None:
    os.environ["IMOVEIS_XLSX_PATH"] = "storage-data/imoveis_apartamentos_rp.xlsx"
    os.environ["STAGING_DATABASE_URL"] = "storage-data/staging_imoveis.db"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["GRAFANA_URL"] = "http://localhost:3000"

    with open("configuracoes/projeto.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # Configura motor tabular para polars e usa sklearn para execução rápida
    conf["tabular"]["motor"] = "polars"
    conf["framework"]["motor"] = "sklearn"
    conf["tuning"]["iteracoes"] = 2

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", encoding="utf-8", delete=False) as tmp:
        yaml.dump(conf, tmp)
        caminho_tmp = tmp.name

    try:
        pipeline = MainPipeline(caminho_tmp)
        resultado = pipeline.executar_pipeline()
        assert resultado["status"] == "sucesso"
        assert resultado["linhas_processadas"] > 0
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)

def test_execucao_main_pipeline_com_transformacoes_e_atributos_novos() -> None:
    os.environ["IMOVEIS_XLSX_PATH"] = "storage-data/imoveis_apartamentos_rp.xlsx"
    os.environ["STAGING_DATABASE_URL"] = "storage-data/staging_imoveis.db"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["GRAFANA_URL"] = "http://localhost:3000"

    with open("configuracoes/projeto.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # Ativa transformações com clip e atributos novos
    conf["transformacoes"]["ativo"] = True
    conf["transformacoes"]["expressoes"][0]["ativo"] = True
    conf["atributos_novos"]["ativo"] = True
    conf["atributos_novos"]["expressoes"][0]["ativo"] = True
    conf["framework"]["motor"] = "sklearn"
    conf["tuning"]["iteracoes"] = 1

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", encoding="utf-8", delete=False) as tmp:
        yaml.dump(conf, tmp)
        caminho_tmp = tmp.name

    try:
        pipeline = MainPipeline(caminho_tmp)
        resultado = pipeline.executar_pipeline()
        assert resultado["status"] == "sucesso"
        assert resultado["linhas_processadas"] > 0
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)

def test_execucao_main_pipeline_com_ensemble_votacao() -> None:
    os.environ["IMOVEIS_XLSX_PATH"] = "storage-data/imoveis_apartamentos_rp.xlsx"
    os.environ["STAGING_DATABASE_URL"] = "storage-data/staging_imoveis.db"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["GRAFANA_URL"] = "http://localhost:3000"

    with open("configuracoes/projeto.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # Ativa técnica de votação mediana entre os melhores 3 modelos
    conf["votacao"]["ativo"] = True
    conf["votacao"]["tecnica"] = "mediana"
    conf["votacao"]["melhores_n"] = 3
    conf["framework"]["motor"] = "sklearn"
    conf["tuning"]["iteracoes"] = 1

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", encoding="utf-8", delete=False) as tmp:
        yaml.dump(conf, tmp)
        caminho_tmp = tmp.name

    try:
        pipeline = MainPipeline(caminho_tmp)
        resultado = pipeline.executar_pipeline()
        assert resultado["status"] == "sucesso"
        assert resultado["linhas_processadas"] > 0
        assert resultado["servico_mlflow_pronto"] is True
        assert resultado["tuning"]["ensemble"] == "mediana"
        assert resultado["estatistica"]["modelo_campeao"] == "ensemble_mediana"
    finally:
        if os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)


