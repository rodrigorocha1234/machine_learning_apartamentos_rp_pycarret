"""Testes automatizados para o orquestrador de implantação e promoção."""

from __future__ import annotations

from pathlib import Path
import pytest
from app.conf_yaml import ConfYAML
from app.servicos_implantacao.orquestrador_implantacao import OrquestradorImplantacao


def test_orquestrador_executa_implantacao(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    conf = ConfYAML()
    orquestrador = OrquestradorImplantacao.da_configuracao(conf)

    # Execução real apontando para diretório temporário para isolamento do teste
    resultado = orquestrador.executar_implantacao(diretorio_artefatos=tmp_path / "campeao")

    nomes_modelos_catalogo = tuple(m.nome for m in conf.obter_modelagem().modelos)
    assert resultado.modelo_promovido in nomes_modelos_catalogo
    assert resultado.versao_modelo == "1.0.0-champion"
    assert resultado.r2_holdout > 0.70
    assert resultado.rmse_holdout < 250_000.0
    assert resultado.total_regras_ativas >= 3

    # Verifica se os artefatos foram salvos
    assert Path(resultado.caminho_artefato_modelo).exists()
    assert Path(resultado.caminho_metadados).exists()

    # Validação do exemplo de predição
    ex = resultado.exemplo_predicao
    assert ex.preco_predito > 100_000.0
    assert "preco_minimo_sugerido" in ex.regras_calculadas
    assert "valor_comissao_estimada" in ex.regras_calculadas
    assert ex.regras_calculadas["valor_comissao_estimada"] == round(ex.preco_predito * 0.06, 2)


def test_orquestrador_implantacao_modelo_dinamico(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica se a seleção dinâmica promove o modelo especificado."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    conf = ConfYAML()
    orquestrador = OrquestradorImplantacao.da_configuracao(conf)

    resultado = orquestrador.executar_implantacao(
        diretorio_artefatos=tmp_path / "campeao_ridge",
        modelo_campeao_override="ridge",
    )

    assert resultado.modelo_promovido == "ridge"
    assert Path(resultado.caminho_artefato_modelo).exists()
    assert Path(resultado.caminho_metadados).exists()
