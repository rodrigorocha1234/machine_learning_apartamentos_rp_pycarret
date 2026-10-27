"""Testes automatizados para os adaptadores e fábrica de fontes de dados."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from app.contratos_dados.fonte_dados import FonteDados
from app.enums_dominio import TipoFonteDados
from app.esquema_configuracao import ConfiguracaoFonte
from app.fontes_dados.fabrica_fontes import ErroFabricaFontes, FabricaFontes
from app.fontes_dados.fonte_excel import ErroFonteExcel, FonteExcel


def test_fonte_excel_carregamento_com_sucesso(tmp_path: Path) -> None:
    """Valida a leitura correta de um arquivo Excel válido."""
    caminho_arquivo = tmp_path / "imoveis_teste.xlsx"
    dados_esperados = pd.DataFrame(
        {
            "Valor_da_Venda": [500000, 750000],
            "Metragem": [85.0, 120.0],
            "Bairro": ["Centro", "Jardim Botânico"],
        }
    )
    dados_esperados.to_excel(caminho_arquivo, index=False, engine="openpyxl")

    fonte = FonteExcel(caminho=caminho_arquivo)
    assert isinstance(fonte, FonteDados)
    assert "imoveis_teste.xlsx" in fonte.obter_origem()

    dados_carregados = fonte.carregar_dados()
    assert len(dados_carregados) == 2
    assert list(dados_carregados.columns) == ["Valor_da_Venda", "Metragem", "Bairro"]
    assert dados_carregados["Valor_da_Venda"].tolist() == [500000, 750000]


def test_fonte_excel_arquivo_inexistente() -> None:
    """Deve levantar ErroFonteExcel quando o arquivo não existir."""
    fonte = FonteExcel("caminho_inexistente/dados.xlsx")
    with pytest.raises(ErroFonteExcel, match="não encontrado"):
        fonte.carregar_dados()


def test_fonte_excel_extensao_invalida(tmp_path: Path) -> None:
    """Deve levantar ErroFonteExcel quando a extensão não for .xlsx ou .xls."""
    arquivo_txt = tmp_path / "dados.txt"
    arquivo_txt.write_text("conteudo", encoding="utf-8")

    fonte = FonteExcel(arquivo_txt)
    with pytest.raises(ErroFonteExcel, match="Extensão inválida"):
        fonte.carregar_dados()


def test_fonte_excel_planilha_vazia(tmp_path: Path) -> None:
    """Deve levantar ErroFonteExcel se a planilha estiver totalmente vazia."""
    arquivo_vazio = tmp_path / "vazio.xlsx"
    pd.DataFrame().to_excel(arquivo_vazio, index=False, engine="openpyxl")

    fonte = FonteExcel(arquivo_vazio)
    with pytest.raises(ErroFonteExcel, match="está vazia"):
        fonte.carregar_dados()


def test_fabrica_fontes_criacao_sucesso(tmp_path: Path) -> None:
    """Valida a criação de FonteExcel via FabricaFontes."""
    arquivo_excel = tmp_path / "dados.xlsx"
    pd.DataFrame({"Valor_da_Venda": [100]}).to_excel(arquivo_excel, index=False, engine="openpyxl")

    config = ConfiguracaoFonte(tipo=TipoFonteDados.XLSX, caminho=str(arquivo_excel), aba=0)
    fonte = FabricaFontes.criar_fonte(config)

    assert isinstance(fonte, FonteExcel)
    assert len(fonte.carregar_dados()) == 1


def test_fabrica_fontes_tipo_invalido() -> None:
    """Deve levantar ErroFabricaFontes para tipos de fonte não suportados."""
    # Instanciamos diretamente ignorando validação do Pydantic para testar a fábrica
    config = ConfiguracaoFonte.model_construct(tipo="formato_inexistente", caminho="dados.csv", aba=0)
    with pytest.raises(ErroFabricaFontes, match="Tipo de fonte de dados não suportado"):
        FabricaFontes.criar_fonte(config)
