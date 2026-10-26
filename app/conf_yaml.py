"""Centralizador de leitura, interpolação e validação de configurações YAML.

Garante tipagem estrita, resolução segura de variáveis de ambiente e validação
de integridade cruzada e segurança de expressões antes da execução das etapas do pipeline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import cast, final
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv()

from app.esquema_configuracao import (
    ConfiguracaoAtributosNovos,
    ConfiguracaoAvaliacao,
    ConfiguracaoColeta,
    ConfiguracaoCompleta,
    ConfiguracaoEda,
    ConfiguracaoFiltros,
    ConfiguracaoFonte,
    ConfiguracaoImplantacao,
    ConfiguracaoModelagem,
    ConfiguracaoMonitoramento,
    ConfiguracaoProjeto,
    ConfiguracaoRepositorioStaging,
    ConfiguracaoSelecao,
    ConfiguracaoTabular,
    ConfiguracaoTransformacoes,
)
from app.validador_ast import ErroExpressaoInsegura, ValidadorAst


class ErroConfiguracao(Exception):
    """Exceção levantada quando há falha de leitura, sintaxe ou validação da configuração."""


@final
class ConfYAML:
    """Classe centralizadora de configurações do projeto imobiliário."""

    _PADRAO_VAR_ENV = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")

    def __init__(self, caminho_ou_diretorio: str | Path | None = None) -> None:
        self._validador_ast = ValidadorAst()
        self._caminho_ou_diretorio = Path(
            caminho_ou_diretorio if caminho_ou_diretorio is not None else "configuracoes"
        )
        self._configuracao: ConfiguracaoCompleta = self._carregar_inicial()

    @classmethod
    def de_dicionario(cls, dados: dict[str, object]) -> ConfYAML:
        """Cria uma instância de ConfYAML diretamente a partir de um dicionário validado."""
        instancia = cls.__new__(cls)
        instancia._validador_ast = ValidadorAst()
        instancia._caminho_ou_diretorio = Path("em_memoria")
        instancia._configuracao = instancia._construir_e_validar(dados)
        return instancia

    def _carregar_inicial(self) -> ConfiguracaoCompleta:
        alvo = self._caminho_ou_diretorio
        if alvo.is_dir():
            return self.carregar_diretorio(alvo)
        if alvo.is_file():
            return self.carregar_arquivo(alvo)
        raise ErroConfiguracao(
            f"Arquivo ou diretório de configuração não encontrado: '{alvo}'"
        )

    def carregar_arquivo(self, caminho: str | Path) -> ConfiguracaoCompleta:
        """Carrega e valida a configuração a partir de um arquivo YAML consolidado."""
        caminho_path = Path(caminho)
        if not caminho_path.is_file():
            raise ErroConfiguracao(f"Arquivo de configuração inexistente: '{caminho_path}'")

        try:
            conteudo_bruto = caminho_path.read_text(encoding="utf-8")
        except OSError as erro_io:
            raise ErroConfiguracao(
                f"Erro ao ler arquivo YAML '{caminho_path}': {erro_io}"
            ) from erro_io

        conteudo_interpolado = self._interpolar_variaveis(conteudo_bruto)

        try:
            dados_brutos = yaml.safe_load(conteudo_interpolado)
        except yaml.YAMLError as erro_yaml:
            raise ErroConfiguracao(
                f"Erro de sintaxe no arquivo YAML '{caminho_path}': {erro_yaml}"
            ) from erro_yaml

        if not isinstance(dados_brutos, dict):
            raise ErroConfiguracao(
                f"Conteúdo raiz do YAML em '{caminho_path}' deve ser um mapeamento (chave-valor)."
            )

        dados_dict = cast(dict[object, object], dados_brutos)
        mapeamento_validado: dict[str, object] = {
            str(chave): valor for chave, valor in dados_dict.items()
        }
        return self._construir_e_validar(mapeamento_validado)

    def carregar_diretorio(self, diretorio: str | Path) -> ConfiguracaoCompleta:
        """Carrega e unifica múltiplos arquivos YAML modulares organizados por etapa."""
        diretorio_path = Path(diretorio)
        if not diretorio_path.is_dir():
            raise ErroConfiguracao(f"Diretório de configurações inexistente: '{diretorio_path}'")

        arquivos_yaml = sorted(
            [f for f in diretorio_path.iterdir() if f.is_file() and f.suffix in {".yaml", ".yml"}]
        )
        if not arquivos_yaml:
            raise ErroConfiguracao(
                f"Nenhum arquivo YAML encontrado no diretório: '{diretorio_path}'"
            )

        dados_unificados: dict[str, object] = {}

        for arquivo in arquivos_yaml:
            try:
                conteudo_bruto = arquivo.read_text(encoding="utf-8")
            except OSError as erro_io:
                raise ErroConfiguracao(
                    f"Erro de leitura no arquivo '{arquivo}': {erro_io}"
                ) from erro_io

            conteudo_interpolado = self._interpolar_variaveis(conteudo_bruto)

            try:
                dados_arquivo = yaml.safe_load(conteudo_interpolado)
            except yaml.YAMLError as erro_yaml:
                raise ErroConfiguracao(
                    f"Erro de sintaxe YAML no arquivo modular '{arquivo}': {erro_yaml}"
                ) from erro_yaml

            if isinstance(dados_arquivo, dict):
                dados_dict_arquivo = cast(dict[object, object], dados_arquivo)
                for chave, valor in dados_dict_arquivo.items():
                    dados_unificados[str(chave)] = valor

        if not dados_unificados:
            raise ErroConfiguracao(
                f"Nenhum dado válido extraído dos arquivos YAML modulares em '{diretorio_path}'"
            )

        return self._construir_e_validar(dados_unificados)

    def _interpolar_variaveis(self, texto: str) -> str:
        """Substitui padrões ${VAR} ou ${VAR:default} pelos valores de variáveis de ambiente."""

        def substituir(match: re.Match[str]) -> str:
            nome_var = match.group(1)
            valor_padrao = match.group(2) if match.group(2) is not None else ""
            return os.getenv(nome_var, valor_padrao)

        return self._PADRAO_VAR_ENV.sub(substituir, texto)

    def _construir_e_validar(self, dados: dict[str, object]) -> ConfiguracaoCompleta:
        """Instancia o modelo Pydantic e realiza validações de integridade e segurança."""
        try:
            configuracao = ConfiguracaoCompleta.model_validate(dados)
        except ValidationError as erro_validacao:
            raise ErroConfiguracao(
                f"Erro de validação do esquema de configuração: {erro_validacao}"
            ) from erro_validacao

        self._validar_referencias_cruzadas(configuracao)
        self._validar_seguranca_expressoes(configuracao)
        return configuracao

    def _validar_referencias_cruzadas(self, config: ConfiguracaoCompleta) -> None:
        """Verifica a consistência semântica e integridade relacional entre as seções."""
        nome_alvo = config.projeto.alvo
        colunas_ativas = [col for col in config.selecao.colunas if col.ativo]

        # 1. Alvo deve estar presente nas colunas selecionadas com papel 'alvo'
        colunas_alvo = [col for col in colunas_ativas if col.papel == "alvo"]
        if not colunas_alvo:
            raise ErroConfiguracao(
                f"Nenhuma coluna ativa foi configurada com papel 'alvo'. Alvo esperado: '{nome_alvo}'."
            )

        if colunas_alvo[0].nome != nome_alvo:
            raise ErroConfiguracao(
                f"Conflito de alvo: projeto.alvo é '{nome_alvo}', mas a coluna com papel 'alvo' é '{colunas_alvo[0].nome}'."
            )

        # 2. Deve existir pelo menos uma coluna preditora ativa
        colunas_preditoras = [
            col for col in colunas_ativas if col.papel in {"numerica", "categorica"}
        ]
        if not colunas_preditoras:
            raise ErroConfiguracao(
                "Nenhuma coluna preditora (numerica ou categorica) ativa foi selecionada."
            )

        # 3. Modelagem: deve haver ao menos um modelo ativo
        modelos_ativos = [mod for mod in config.modelagem.modelos if mod.ativo]
        if not modelos_ativos:
            raise ErroConfiguracao(
                "Nenhum modelo de regressão está ativo em 'modelagem.modelos'."
            )

        nomes_modelos = {mod.nome for mod in config.modelagem.modelos}

        # 4. Tuning: parâmetros devem referenciar modelos existentes
        if config.modelagem.tuning.ativo:
            for nome_modelo in config.modelagem.tuning.parametros:
                if nome_modelo not in nomes_modelos:
                    raise ErroConfiguracao(
                        f"Configuração de tuning referencia modelo inexistente: '{nome_modelo}'."
                    )

        # 5. Votação: se ativa, melhores_n deve ser menor ou igual ao número de modelos ativos
        if config.avaliacao.votacao.ativo:
            if config.avaliacao.votacao.melhores_n > len(modelos_ativos):
                raise ErroConfiguracao(
                    f"Votação configurada para {config.avaliacao.votacao.melhores_n} melhores modelos, "
                    f"mas apenas {len(modelos_ativos)} modelos estão ativos."
                )

        # 6. Regras de negócio: destinos não podem colidir entre regras
        destinos_regras: set[str] = set()
        for regra in config.implantacao.regras:
            if regra.ativo:
                if regra.destino in destinos_regras:
                    raise ErroConfiguracao(
                        f"Destino duplicado detectado em regra de negócio: '{regra.destino}'."
                    )
                destinos_regras.add(regra.destino)

    def _validar_seguranca_expressoes(self, config: ConfiguracaoCompleta) -> None:
        """Valida que todas as expressões configuradas passam pela whitelist do parser AST."""
        # Filtros
        if config.filtros.ativo:
            for filtro in config.filtros.expressoes:
                if filtro.ativo:
                    for motor, expr in filtro.expressao.items():
                        try:
                            self._validador_ast.validar_expressao(expr)
                        except ErroExpressaoInsegura as erro_ast:
                            raise ErroConfiguracao(
                                f"Expressão insegura no filtro '{filtro.nome}' (motor '{motor}'): {erro_ast}"
                            ) from erro_ast

        # Transformações
        if config.transformacoes.ativo:
            for transformacao in config.transformacoes.expressoes:
                if transformacao.ativo:
                    for motor, expr in transformacao.expressao.items():
                        try:
                            self._validador_ast.validar_expressao(expr)
                        except ErroExpressaoInsegura as erro_ast:
                            raise ErroConfiguracao(
                                f"Expressão insegura na transformação '{transformacao.nome}' (motor '{motor}'): {erro_ast}"
                            ) from erro_ast

        # Atributos novos
        if config.atributos_novos.ativo:
            for atributo in config.atributos_novos.expressoes:
                if atributo.ativo:
                    for motor, expr in atributo.expressao.items():
                        try:
                            self._validador_ast.validar_expressao(expr)
                        except ErroExpressaoInsegura as erro_ast:
                            raise ErroConfiguracao(
                                f"Expressão insegura no novo atributo '{atributo.nome}' (motor '{motor}'): {erro_ast}"
                            ) from erro_ast

        # Regras de negócio
        if config.implantacao.ativo:
            for regra in config.implantacao.regras:
                if regra.ativo:
                    try:
                        self._validador_ast.validar_expressao(regra.expressao)
                    except ErroExpressaoInsegura as erro_ast:
                        raise ErroConfiguracao(
                            f"Expressão insegura na regra de negócio '{regra.nome}': {erro_ast}"
                        ) from erro_ast

        # Análises EDA
        for analise in config.eda.analises:
            if analise.ativo and analise.expressao:
                try:
                    self._validador_ast.validar_expressao(analise.expressao)
                except ErroExpressaoInsegura as erro_ast:
                    raise ErroConfiguracao(
                        f"Expressão insegura na análise EDA '{analise.nome}': {erro_ast}"
                    ) from erro_ast

    # --------------------------------------------------------------------------
    # Acessadores fortemente tipados
    # --------------------------------------------------------------------------

    def obter_configuracao_completa(self) -> ConfiguracaoCompleta:
        """Retorna o objeto completo e tipado de configuração."""
        return self._configuracao

    def obter_projeto(self) -> ConfiguracaoProjeto:
        """Retorna as configurações de identificação do projeto."""
        return self._configuracao.projeto

    def obter_fonte(self) -> ConfiguracaoFonte:
        """Retorna as configurações da fonte primária de dados."""
        return self._configuracao.fonte

    def obter_staging(self) -> ConfiguracaoRepositorioStaging:
        """Retorna as configurações do repositório de staging intermediário."""
        return self._configuracao.repositorio_staging

    def obter_coleta(self) -> ConfiguracaoColeta:
        """Retorna as configurações e validações de atributos de negócio da etapa de coleta."""
        return self._configuracao.coleta

    def obter_tabular(self) -> ConfiguracaoTabular:
        """Retorna o motor de processamento tabular configurado."""
        return self._configuracao.tabular

    def obter_selecao(self) -> ConfiguracaoSelecao:
        """Retorna a seleção e tipagem de colunas ativas."""
        return self._configuracao.selecao

    def obter_filtros(self) -> ConfiguracaoFiltros:
        """Retorna as regras de filtragem condicional."""
        return self._configuracao.filtros

    def obter_transformacoes(self) -> ConfiguracaoTransformacoes:
        """Retorna as transformações de atributos configuradas."""
        return self._configuracao.transformacoes

    def obter_atributos_novos(self) -> ConfiguracaoAtributosNovos:
        """Retorna a especificação de criação de novos atributos."""
        return self._configuracao.atributos_novos

    def obter_eda(self) -> ConfiguracaoEda:
        """Retorna as definições da análise exploratória de dados."""
        return self._configuracao.eda

    def obter_modelagem(self) -> ConfiguracaoModelagem:
        """Retorna as configurações completas de modelagem e tuning."""
        return self._configuracao.modelagem

    def obter_avaliacao(self) -> ConfiguracaoAvaliacao:
        """Retorna as configurações de validação cruzada, estatística e votação."""
        return self._configuracao.avaliacao

    def obter_implantacao(self) -> ConfiguracaoImplantacao:
        """Retorna as configurações de implantação e regras de negócio."""
        return self._configuracao.implantacao

    def obter_monitoramento(self) -> ConfiguracaoMonitoramento:
        """Retorna as configurações de observabilidade e detecção de drift."""
        return self._configuracao.monitoramento
