"""Bateria de testes automatizados para leitura, validação e integridade de ConfYAML.

Cobre cenários de configuração válida exclusivamente a partir dos arquivos modulares por etapa,
interpolação de variáveis de ambiente, erros sintáticos, falhas de esquema, referências cruzadas
inconsistentes e rejeição de expressões maliciosas ou inseguras via AST.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from app.conf_yaml import ConfYAML, ErroConfiguracao
from conf_yaml import ConfYAML as ConfYAMLCompativel


def test_carregar_configuracao_padrao_sem_argumentos() -> None:
    """Valida o carregamento padrão a partir da pasta 'configuracoes' sem argumentos."""
    conf = ConfYAML()
    config = conf.obter_configuracao_completa()

    # Validação do Projeto (Etapa 1)
    assert config.projeto.nome == "previsao_imoveis_rp"
    assert config.projeto.alvo == "Valor_da_Venda"
    assert config.projeto.semente == 42

    # Validação da Fonte e Staging (Etapa 1)
    assert config.fonte.tipo == "xlsx"
    assert "imoveis_apartamentos_rp.xlsx" in config.fonte.caminho
    assert config.repositorio_staging.tipo == "sqlite"
    assert config.repositorio_staging.tabela == "imoveis_preparados"

    # Validação do Motor Tabular e Seleção (Etapa 2)
    assert config.tabular.motor == "pandas"
    assert len(config.selecao.colunas) >= 5
    alvo_encontrado = [col for col in config.selecao.colunas if col.papel == "alvo"]
    assert len(alvo_encontrado) == 1
    assert alvo_encontrado[0].nome == "Valor_da_Venda"

    # Validação da EDA (Etapa 3)
    assert config.eda.motor_grafico in {"grafana", "seaborn", "plotly", "altair"}
    assert len(config.eda.analises) >= 3

    # Validação da Modelagem (Etapa 4)
    assert config.modelagem.motor_treino in {"pycaret", "sklearn"}
    assert len(config.modelagem.modelos) >= 5
    assert config.modelagem.tuning.ativo is True
    assert config.modelagem.tuning.iteracoes >= 5

    # Validação da Avaliação (Etapa 5)
    assert config.avaliacao.validacao.holdout == 0.2
    assert config.avaliacao.estatistica.teste == "friedman"
    assert config.avaliacao.estatistica.pos_teste == "nemenyi"

    # Validação das Regras de Negócio (Etapa 6)
    assert config.implantacao.ativo is True
    assert len(config.implantacao.regras) >= 2

    # Validação de Monitoramento e Drift (Etapa 7)
    assert config.monitoramento.drift.ativo is True


def test_carregar_diretorio_modular_explicito(diretorio_configuracoes: Path) -> None:
    """Valida a unificação modular dos arquivos YAML por etapa via caminho explícito."""
    conf = ConfYAML(diretorio_configuracoes)
    config = conf.obter_configuracao_completa()

    assert config.projeto.nome == "previsao_imoveis_rp"
    assert config.fonte.tipo == "xlsx"
    assert config.tabular.motor == "pandas"
    assert config.modelagem.motor_treino == "pycaret"
    assert config.avaliacao.estatistica.teste == "friedman"
    assert len(config.implantacao.regras) >= 2
    assert config.monitoramento.drift.ativo is True


def test_importacao_modulo_raiz_compatibilidade() -> None:
    """Garante que a classe pode ser importada via `from conf_yaml import ConfYAML`."""
    conf = ConfYAMLCompativel()
    assert conf.obter_projeto().nome == "previsao_imoveis_rp"


def test_acessadores_fortemente_tipados() -> None:
    """Verifica todos os métodos acessadores específicos da classe ConfYAML."""
    conf = ConfYAML()

    assert conf.obter_projeto().alvo == "Valor_da_Venda"
    assert conf.obter_fonte().tipo == "xlsx"
    assert conf.obter_staging().tipo == "sqlite"
    assert conf.obter_tabular().motor in {"pandas", "polars"}
    assert conf.obter_selecao().ativo is True
    assert conf.obter_filtros().ativo is True
    assert conf.obter_transformacoes().ativo is True
    assert conf.obter_atributos_novos().ativo is False
    assert conf.obter_eda().motor_grafico in {"grafana", "seaborn", "plotly", "altair"}
    assert conf.obter_modelagem().motor_treino in {"pycaret", "sklearn"}
    assert conf.obter_avaliacao().validacao.metrica == "RMSE"
    assert conf.obter_implantacao().coluna_alvo_predito == "preco_predito"
    assert len(conf.obter_monitoramento().observabilidade.metricas) >= 3


def test_interpolacao_variaveis_ambiente(tmp_path: Path) -> None:
    """Testa a correta interpolação de variáveis de ambiente com e sem valores padrão."""
    os.environ["VAR_TESTE_EXP"] = "caminho/customizado/dados.xlsx"
    conteudo = """
projeto:
  nome: "teste_env"
  alvo: "preco"
fonte:
  tipo: "xlsx"
  caminho: "${VAR_TESTE_EXP}"
repositorio_staging:
  tipo: "sqlite"
  caminho: "${VAR_INEXISTENTE:caminho/padrao/staging.db}"
selecao:
  ativo: true
  colunas:
    - {nome: "preco", tipo: "quantitativa_continua", papel: "alvo"}
    - {nome: "area", tipo: "quantitativa_continua", papel: "numerica"}
modelagem:
  motor_treino: "sklearn"
  modelos:
    - {nome: "linear", pycaret_id: "lr", sklearn_class: "LinearRegression"}
avaliacao:
  validacao:
    holdout: 0.2
"""
    pasta_teste = tmp_path / "modular_env"
    pasta_teste.mkdir()
    (pasta_teste / "01_dados.yaml").write_text(conteudo, encoding="utf-8")

    conf = ConfYAML(pasta_teste)
    assert conf.obter_fonte().caminho == "caminho/customizado/dados.xlsx"
    assert conf.obter_staging().caminho == "caminho/padrao/staging.db"


def test_erro_arquivo_inexistente() -> None:
    """Deve levantar ErroConfiguracao quando o arquivo não existir."""
    with pytest.raises(ErroConfiguracao, match="Arquivo ou diretório de configuração não encontrado"):
        ConfYAML("arquivo_totalmente_inexistente.yaml")


def test_erro_diretorio_vazio(tmp_path: Path) -> None:
    """Deve levantar ErroConfiguracao se o diretório não tiver arquivos YAML."""
    pasta_vazia = tmp_path / "vazia"
    pasta_vazia.mkdir()

    with pytest.raises(ErroConfiguracao, match="Nenhum arquivo YAML encontrado"):
        ConfYAML(pasta_vazia)


def test_erro_sintaxe_yaml(diretorio_copia_modular: Path) -> None:
    """Deve levantar ErroConfiguracao ao encontrar sintaxe YAML inválida."""
    arquivo_invalido = diretorio_copia_modular / "01_coleta_dados.yaml"
    arquivo_invalido.write_text("chave_invalida: [item_nao_fechado", encoding="utf-8")

    with pytest.raises(ErroConfiguracao, match="Erro de sintaxe YAML"):
        ConfYAML(diretorio_copia_modular)


def test_erro_motor_tabular_invalido(diretorio_copia_modular: Path) -> None:
    """Deve rejeitar motor tabular não suportado (ex: spark)."""
    arquivo_prep = diretorio_copia_modular / "02_preparacao_dados.yaml"
    conteudo = arquivo_prep.read_text(encoding="utf-8")
    arquivo_prep.write_text(conteudo.replace('motor: "pandas"', 'motor: "spark"'), encoding="utf-8")

    with pytest.raises(ErroConfiguracao, match="Erro de validação do esquema"):
        ConfYAML(diretorio_copia_modular)


def test_erro_motor_treino_invalido(diretorio_copia_modular: Path) -> None:
    """Deve rejeitar motor de treino não suportado (ex: tensorflow)."""
    arquivo_mod = diretorio_copia_modular / "04_modelagem_regressao.yaml"
    conteudo = arquivo_mod.read_text(encoding="utf-8")
    arquivo_mod.write_text(
        conteudo.replace('motor_treino: "pycaret"', 'motor_treino: "tensorflow"'), encoding="utf-8"
    )

    with pytest.raises(ErroConfiguracao, match="Erro de validação do esquema"):
        ConfYAML(diretorio_copia_modular)


def test_erro_holdout_invalido(diretorio_copia_modular: Path) -> None:
    """Deve rejeitar frações de holdout fora do intervalo (0, 1)."""
    arquivo_aval = diretorio_copia_modular / "05_avaliacao_selecao.yaml"
    conteudo = arquivo_aval.read_text(encoding="utf-8")
    arquivo_aval.write_text(conteudo.replace("holdout: 0.2", "holdout: 1.5"), encoding="utf-8")

    with pytest.raises(ErroConfiguracao, match="Erro de validação do esquema"):
        ConfYAML(diretorio_copia_modular)


def test_erro_conflito_nome_alvo(diretorio_copia_modular: Path) -> None:
    """Deve levantar erro quando o alvo do projeto diferir do alvo das colunas."""
    arquivo_coleta = diretorio_copia_modular / "01_coleta_dados.yaml"
    conteudo = arquivo_coleta.read_text(encoding="utf-8")
    arquivo_coleta.write_text(
        conteudo.replace('alvo: "Valor_da_Venda"', 'alvo: "Outro_Alvo"'), encoding="utf-8"
    )

    with pytest.raises(ErroConfiguracao, match="Conflito de alvo"):
        ConfYAML(diretorio_copia_modular)


def test_erro_nenhuma_coluna_preditora(diretorio_copia_modular: Path) -> None:
    """Deve falhar quando não há colunas ativas para predição."""
    arquivo_prep = diretorio_copia_modular / "02_preparacao_dados.yaml"
    conteudo_so_alvo = """
tabular:
  motor: "pandas"
selecao:
  ativo: true
  colunas:
    - {nome: "Valor_da_Venda", ativo: true, tipo: "quantitativa_continua", papel: "alvo"}
"""
    arquivo_prep.write_text(conteudo_so_alvo, encoding="utf-8")

    with pytest.raises(ErroConfiguracao, match="Nenhuma coluna preditora"):
        ConfYAML(diretorio_copia_modular)


def test_erro_nenhum_modelo_ativo(diretorio_copia_modular: Path) -> None:
    """Deve falhar quando todos os modelos estiverem desativados."""
    arquivo_mod = diretorio_copia_modular / "04_modelagem_regressao.yaml"
    conteudo = arquivo_mod.read_text(encoding="utf-8")
    arquivo_mod.write_text(
        conteudo.replace("ativo: true, pycaret_id", "ativo: false, pycaret_id"), encoding="utf-8"
    )

    with pytest.raises(ErroConfiguracao, match="Nenhum modelo de regressão está ativo"):
        ConfYAML(diretorio_copia_modular)


def test_erro_tuning_modelo_inexistente(diretorio_copia_modular: Path) -> None:
    """Deve falhar se a parametrização de tuning referenciar modelo ausente."""
    arquivo_mod = diretorio_copia_modular / "04_modelagem_regressao.yaml"
    conteudo = arquivo_mod.read_text(encoding="utf-8")
    arquivo_mod.write_text(
        conteudo.replace(
            "regressao_linear:\n        fit_intercept:",
            "modelo_fantasma:\n        fit_intercept:",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ErroConfiguracao, match="referencia modelo inexistente: 'modelo_fantasma'"):
        ConfYAML(diretorio_copia_modular)


def test_erro_votacao_melhores_n_excedente(diretorio_copia_modular: Path) -> None:
    """Deve falhar se a votação solicitar mais modelos do que os ativos no catálogo."""
    arquivo_aval = diretorio_copia_modular / "05_avaliacao_selecao.yaml"
    conteudo = arquivo_aval.read_text(encoding="utf-8")
    conteudo_alterado = conteudo.replace("ativo: false", "ativo: true").replace(
        "melhores_n: 3", "melhores_n: 99"
    )
    arquivo_aval.write_text(conteudo_alterado, encoding="utf-8")

    with pytest.raises(ErroConfiguracao, match="Votação configurada para 99 melhores modelos"):
        ConfYAML(diretorio_copia_modular)


def test_erro_destino_duplicado_regras(diretorio_copia_modular: Path) -> None:
    """Deve falhar quando duas regras de negócio definem o mesmo campo destino."""
    arquivo_regras = diretorio_copia_modular / "06_implantacao_regras.yaml"
    conteudo = arquivo_regras.read_text(encoding="utf-8")
    arquivo_regras.write_text(
        conteudo.replace('destino: "preco_maximo_sugerido"', 'destino: "preco_minimo_sugerido"'),
        encoding="utf-8",
    )

    with pytest.raises(ErroConfiguracao, match="Destino duplicado detectado em regra de negócio"):
        ConfYAML(diretorio_copia_modular)


def test_rejeicao_expressao_maliciosa_import(diretorio_copia_modular: Path) -> None:
    """Bloqueia tentativas de importar módulos ou executar chamadas perigosas em expressões."""
    arquivo_regras = diretorio_copia_modular / "06_implantacao_regras.yaml"
    conteudo = arquivo_regras.read_text(encoding="utf-8")
    arquivo_regras.write_text(
        conteudo.replace(
            'expressao: "preco_predito * 0.95"', 'expressao: "__import__(\'os\').system(\'ls\')"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ErroConfiguracao, match="Expressão insegura"):
        ConfYAML(diretorio_copia_modular)


def test_rejeicao_expressao_maliciosa_eval(diretorio_copia_modular: Path) -> None:
    """Bloqueia chamadas da função builtin eval."""
    arquivo_regras = diretorio_copia_modular / "06_implantacao_regras.yaml"
    conteudo = arquivo_regras.read_text(encoding="utf-8")
    arquivo_regras.write_text(
        conteudo.replace(
            'expressao: "preco_predito * 0.95"', 'expressao: "eval(\'preco_predito\')"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ErroConfiguracao, match="Expressão insegura"):
        ConfYAML(diretorio_copia_modular)


def test_rejeicao_expressao_maliciosa_introspeccao(diretorio_copia_modular: Path) -> None:
    """Bloqueia acesso a atributos de introspecção como __class__ ou __subclasses__."""
    arquivo_regras = diretorio_copia_modular / "06_implantacao_regras.yaml"
    conteudo = arquivo_regras.read_text(encoding="utf-8")
    arquivo_regras.write_text(
        conteudo.replace(
            'expressao: "preco_predito * 0.95"', 'expressao: "preco_predito.__class__.__name__"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ErroConfiguracao, match="Expressão insegura"):
        ConfYAML(diretorio_copia_modular)
