"""Testes automatizados para o orquestrador de implantação e promoção."""

from __future__ import annotations

from pathlib import Path
from app.conf_yaml import ConfYAML
from app.servicos_implantacao.orquestrador_implantacao import OrquestradorImplantacao


def test_orquestrador_executa_implantacao(tmp_path: Path) -> None:
    conf = ConfYAML()
    orquestrador = OrquestradorImplantacao.da_configuracao(conf)

    # Execução real apontando para diretório temporário para isolamento do teste
    resultado = orquestrador.executar_implantacao(diretorio_artefatos=tmp_path / "campeao")

    assert resultado.modelo_promovido == "lightgbm"
    assert resultado.versao_modelo == "1.0.0-champion"
    assert resultado.r2_holdout > 0.85
    assert resultado.rmse_holdout < 130_000.0
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
