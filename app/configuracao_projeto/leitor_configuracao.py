from __future__ import annotations
import os
import re
from typing import final
import yaml

type ValorConfiguracao = (
    str
    | int
    | float
    | bool
    | None
    | list[ValorConfiguracao]
    | dict[str, ValorConfiguracao]
)

class ErroConfiguracao(Exception):
    """Exceção para erros de carregamento ou validação do arquivo de configuração."""
    pass

@final
class LeitorConfiguracao:
    """Carregador de arquivos YAML com interpolação de variáveis de ambiente."""

    _PADRAO_VAR_ENV = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")

    def carregar_configuracao(self, caminho_yaml: str) -> dict[str, ValorConfiguracao]:
        if not os.path.exists(caminho_yaml):
            raise ErroConfiguracao(f"Arquivo de configuração não encontrado em: '{caminho_yaml}'")

        try:
            with open(caminho_yaml, "r", encoding="utf-8") as f:
                conteudo_bruto = f.read()
        except Exception as erro:
            raise ErroConfiguracao(f"Erro ao ler arquivo YAML '{caminho_yaml}': {erro}") from erro

        conteudo_interpolado = self._interpolar_variaveis(conteudo_bruto)

        try:
            configuracao = yaml.safe_load(conteudo_interpolado)
            if not isinstance(configuracao, dict):
                raise ErroConfiguracao("Estrutura raiz do YAML deve ser um mapeamento (dict).")
            return configuracao
        except yaml.YAMLError as erro_yaml:
            raise ErroConfiguracao(f"Erro de sintaxe YAML: {erro_yaml}") from erro_yaml

    def _interpolar_variaveis(self, texto: str) -> str:
        def substituir(match: re.Match[str]) -> str:
            nome_var = match.group(1)
            valor_padrao = match.group(2) if match.group(2) is not None else ""
            return os.getenv(nome_var, valor_padrao)

        return self._PADRAO_VAR_ENV.sub(substituir, texto)
