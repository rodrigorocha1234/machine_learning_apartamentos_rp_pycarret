"""Testes automatizados de integração para o serviço genérico ColetorDados."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from app.conf_yaml import ConfYAML
from app.fontes_dados.fonte_excel import FonteExcel
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_coleta.coletor_dados import ColetorDados, ErroColetaDados


def test_coletor_dados_execucao_completa(tmp_path: Path) -> None:
    """Valida o fluxo ponta a ponta de coleta e persistência no staging intermediário."""
    arquivo_excel = tmp_path / "amostra_imoveis.xlsx"
    banco_sqlite = tmp_path / "staging_teste.db"

    dados_simulados = pd.DataFrame(
        {
            "Codigo_Imovel": ["APT-01", "APT-02", "APT-03"],
            "Bairro": ["Centro", "Vila Tibério", "Bonfim"],
            "Zona": ["Centro", "Oeste", "Oeste"],
            "Metragem": [60.0, 75.0, 90.0],
            "Valor_da_Venda": [250000, 320000, 410000],
        }
    )
    dados_simulados.to_excel(arquivo_excel, index=False, engine="openpyxl")

    fonte = FonteExcel(arquivo_excel)
    repo = RepositorioSqlite(banco_sqlite, tabela="imoveis_preparados")
    coletor = ColetorDados(
        fonte=fonte,
        repositorio=repo,
        nome_alvo="Valor_da_Venda",
        tabela_destino="imoveis_preparados",
    )

    resultado = coletor.executar_coleta(substituir_staging=True)

    assert resultado.total_registros == 3
    assert resultado.total_colunas == 5
    assert resultado.alvo_presente is True
    assert resultado.tabela_destino == "imoveis_preparados"
    assert resultado.tempo_execucao_segundos >= 0.0

    # Confirma que os dados foram fisicamente gravados no SQLite
    dados_persistidos = repo.ler_dados()
    assert len(dados_persistidos) == 3
    assert "Valor_da_Venda" in dados_persistidos.columns
    assert dados_persistidos["Valor_da_Venda"].sum() == 250000 + 320000 + 410000


def test_coletor_dados_alvo_ausente(tmp_path: Path) -> None:
    """Deve levantar ErroColetaDados se a planilha não contiver a coluna alvo."""
    arquivo_sem_alvo = tmp_path / "sem_alvo.xlsx"
    pd.DataFrame({"Metragem": [50.0], "Bairro": ["Centro"]}).to_excel(
        arquivo_sem_alvo, index=False, engine="openpyxl"
    )

    fonte = FonteExcel(arquivo_sem_alvo)
    repo = RepositorioSqlite(tmp_path / "staging.db")
    coletor = ColetorDados(fonte=fonte, repositorio=repo, nome_alvo="Valor_da_Venda")

    with pytest.raises(ErroColetaDados, match="coluna alvo obrigatória 'Valor_da_Venda' não está presente"):
        coletor.executar_coleta()


def test_coletor_dados_instanciacao_da_configuracao(tmp_path: Path) -> None:
    """Valida a instanciação desacoplada de ColetorDados a partir de ConfYAML."""
    arquivo_excel = tmp_path / "imoveis.xlsx"
    pd.DataFrame({"Valor_da_Venda": [100000]}).to_excel(arquivo_excel, index=False, engine="openpyxl")

    pasta_conf = tmp_path / "configuracoes"
    pasta_conf.mkdir()
    conteudo_coleta = f"""
projeto:
  nome: "teste_coleta"
  alvo: "Valor_da_Venda"
fonte:
  tipo: "xlsx"
  caminho: "{arquivo_excel}"
repositorio_staging:
  tipo: "sqlite"
  caminho: "{tmp_path}/staging.db"
  tabela: "imoveis_preparados"
selecao:
  ativo: true
  colunas:
    - {{nome: "Valor_da_Venda", tipo: "quantitativa_continua", papel: "alvo"}}
    - {{nome: "Metragem", tipo: "quantitativa_continua", papel: "numerica"}}
modelagem:
  motor_treino: "sklearn"
  modelos:
    - {{nome: "linear", pycaret_id: "lr", sklearn_class: "LinearRegression"}}
avaliacao:
  validacao:
    holdout: 0.2
"""
    (pasta_conf / "01_coleta_dados.yaml").write_text(conteudo_coleta, encoding="utf-8")

    conf = ConfYAML(pasta_conf)
    coletor = ColetorDados.da_configuracao(conf)

    resultado = coletor.executar_coleta()
    assert resultado.total_registros == 1
    assert resultado.alvo_presente is True


def test_coletor_colunas_obrigatorias_ausentes(tmp_path: Path) -> None:
    """Deve levantar ErroColetaDados se colunas de negócio obrigatórias estiverem ausentes."""
    arquivo_excel = tmp_path / "faltando_colunas.xlsx"
    pd.DataFrame({"Valor_da_Venda": [100000]}).to_excel(arquivo_excel, index=False, engine="openpyxl")

    fonte = FonteExcel(arquivo_excel)
    repo = RepositorioSqlite(tmp_path / "staging.db")
    coletor = ColetorDados(
        fonte=fonte,
        repositorio=repo,
        nome_alvo="Valor_da_Venda",
        colunas_obrigatorias=["Valor_da_Venda", "Metragem", "Bairro"],
    )

    with pytest.raises(ErroColetaDados, match="colunas de negócio obrigatórias configuradas no YAML estão ausentes"):
        coletor.executar_coleta()


def test_coletor_validar_nulos_no_alvo(tmp_path: Path) -> None:
    """Deve levantar ErroColetaDados se a flag validar_nulos_alvo estiver ativa e houver nulos."""
    arquivo_excel = tmp_path / "com_nulos_alvo.xlsx"
    pd.DataFrame({"Valor_da_Venda": [100000, None, 200000]}).to_excel(
        arquivo_excel, index=False, engine="openpyxl"
    )

    fonte = FonteExcel(arquivo_excel)
    repo = RepositorioSqlite(tmp_path / "staging.db")
    coletor = ColetorDados(
        fonte=fonte,
        repositorio=repo,
        nome_alvo="Valor_da_Venda",
        validar_nulos_alvo=True,
    )

    with pytest.raises(ErroColetaDados, match="possui 1 valores ausentes"):
        coletor.executar_coleta()
