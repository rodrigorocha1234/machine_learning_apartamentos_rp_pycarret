"""Motor de cálculo de regras de negócio baseado em Polars e execução segura AST."""

from __future__ import annotations

from types import CodeType
from typing import cast, final, override
import pandas as pd
import polars as pl

from app.contratos_implantacao.avaliador_regras import AvaliadorRegras
from app.esquema_configuracao import ConfiguracaoImplantacao, RegraNegocioItem
from app.validador_ast import ErroExpressaoInsegura, ValidadorAst


class ErroAvaliadorPolars(Exception):
    """Exceção levantada em falhas durante a validação ou cálculo de regras com Polars."""


@final
class AvaliadorPolars(AvaliadorRegras):
    """Implementação do AvaliadorRegras utilizando Polars e execução segura."""

    def __init__(self, config_implantacao: ConfiguracaoImplantacao) -> None:
        self._config = config_implantacao
        self._validador_ast = ValidadorAst()
        self._regras_compiladas: list[tuple[RegraNegocioItem, CodeType]] = []
        self._preparar_regras()

    def _preparar_regras(self) -> None:
        """Valida e pré-compila as expressões das regras ativas."""
        if not self._config.ativo:
            return

        for regra in self._config.regras:
            if not regra.ativo:
                continue
            try:
                self._validador_ast.validar_expressao(regra.expressao)
                codigo_compilado = compile(regra.expressao, f"<regra_{regra.nome}>", "eval")
                self._regras_compiladas.append((regra, codigo_compilado))
            except ErroExpressaoInsegura as erro_ast:
                raise ErroAvaliadorPolars(
                    f"Expressão insegura na regra '{regra.nome}': {erro_ast}"
                ) from erro_ast
            except SyntaxError as erro_sintaxe:
                raise ErroAvaliadorPolars(
                    f"Erro de sintaxe na expressão da regra '{regra.nome}': {erro_sintaxe}"
                ) from erro_sintaxe

    @override
    def avaliar_regras_registro(
        self,
        contexto: dict[str, float | int | str],
    ) -> dict[str, float]:
        """Calcula todas as regras ativas sobre as variáveis de um registro individual."""
        if not self._config.ativo or not self._regras_compiladas:
            return {}

        resultados: dict[str, float] = {}
        ambiente_execucao: dict[str, object] = {"__builtins__": {}}
        variaveis_locais: dict[str, object] = dict(contexto)

        for regra, codigo in self._regras_compiladas:
            try:
                valor_calculado = eval(codigo, ambiente_execucao, variaveis_locais)
                valor_float = float(cast(float | int, valor_calculado))
                resultados[regra.destino] = round(valor_float, 2)
                variaveis_locais[regra.destino] = valor_float
            except ZeroDivisionError:
                resultados[regra.destino] = 0.0
                variaveis_locais[regra.destino] = 0.0
            except Exception as erro:
                raise ErroAvaliadorPolars(
                    f"Falha ao calcular regra '{regra.nome}' com Polars: {erro}"
                ) from erro

        return resultados

    @override
    def avaliar_regras_tabela(
        self,
        dados: pd.DataFrame,
        coluna_predita: str,
    ) -> pd.DataFrame:
        """Aplica as regras sobre a tabela convertendo temporariamente para Polars se necessário."""
        if not self._config.ativo or not self._regras_compiladas:
            return dados.copy()

        if coluna_predita not in dados.columns:
            raise ErroAvaliadorPolars(
                f"Coluna de preço predito '{coluna_predita}' não encontrada na tabela."
            )

        # Conversão para Polars para processamento eficiente
        df_polars = pl.from_pandas(dados)
        linhas_regras: list[dict[str, float]] = []

        for dict_linha in df_polars.to_dicts():
            contexto_linha: dict[str, float | int | str] = {}
            for k, val in dict_linha.items():
                if isinstance(val, (int, float, str)):
                    contexto_linha[str(k)] = val
                else:
                    try:
                        contexto_linha[str(k)] = float(str(val))
                    except (ValueError, TypeError):
                        contexto_linha[str(k)] = str(val)

            res = self.avaliar_regras_registro(contexto_linha)
            linhas_regras.append(res)

        df_novas = pd.DataFrame(linhas_regras, index=dados.index)
        df_final = dados.copy()
        for col_obj in cast(list[object], list(df_novas.columns)):
            col_nome = str(col_obj)
            df_final[col_nome] = df_novas[col_nome]

        return df_final
