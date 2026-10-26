"""Implementação de MotorTabela baseada em Polars LazyFrame com AST seguro."""

from __future__ import annotations

import ast
from collections.abc import Callable
import operator
from typing import final, override
import pandas as pd
import polars as pl

from app.contratos_dados.motor_tabela import MotorTabela


class ErroMotorPolars(Exception):
    """Exceção levantada em falhas de processamento ou avaliação do motor Polars."""


@final
class MotorPolars(MotorTabela):
    """Motor tabular de alta performance baseado em Polars LazyFrame."""

    _OPERADORES_BINARIOS: dict[
        type[ast.operator], Callable[[pl.Expr, pl.Expr], pl.Expr]
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _OPERADORES_UNARIOS: dict[type[ast.unaryop], Callable[[pl.Expr], pl.Expr]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    _OPERADORES_COMPARACAO: dict[
        type[ast.cmpop], Callable[[pl.Expr, pl.Expr], pl.Expr]
    ] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    _METODOS_PERMITIDOS: frozenset[str] = frozenset({"clip", "abs", "round"})
    _NOMES_DATAFRAME_PERMITIDOS: frozenset[str] = frozenset({"df", "dados", "data", "tabela"})

    @override
    def obter_nome_motor(self) -> str:
        return "polars"

    @override
    def filtrar_dados(self, dados: pd.DataFrame, expressao: str) -> pd.DataFrame:
        if not expressao.strip():
            return dados.copy()

        try:
            pl_df = pl.from_pandas(dados)
            colunas = set(pl_df.columns)
            expressao_polars = self._avaliar_expressao(expressao, colunas)
            resultado_pl = pl_df.lazy().filter(expressao_polars).collect()
            return resultado_pl.to_pandas().reset_index(drop=True)
        except Exception as erro:
            raise ErroMotorPolars(
                f"Falha ao filtrar dados no Polars com '{expressao}': {erro}"
            ) from erro

    @override
    def transformar_dados(
        self, dados: pd.DataFrame, expressao: str, destino: str
    ) -> pd.DataFrame:
        try:
            pl_df = pl.from_pandas(dados)
            colunas = set(pl_df.columns)
            expressao_polars = self._avaliar_expressao(expressao, colunas)
            resultado_pl = (
                pl_df.lazy()
                .with_columns(expressao_polars.alias(destino))
                .collect()
            )
            return resultado_pl.to_pandas()
        except Exception as erro:
            raise ErroMotorPolars(
                f"Falha ao transformar dados no Polars para '{destino}' com '{expressao}': {erro}"
            ) from erro

    @override
    def selecionar_colunas(self, dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
        pl_df = pl.from_pandas(dados)
        colunas_inexistentes = [c for c in colunas if c not in pl_df.columns]
        if colunas_inexistentes:
            raise ErroMotorPolars(
                f"Colunas não encontradas no DataFrame Polars: {colunas_inexistentes}"
            )
        resultado_pl = pl_df.lazy().select([pl.col(c) for c in colunas]).collect()
        return resultado_pl.to_pandas()

    def _avaliar_expressao(self, expressao: str, colunas_validas: set[str]) -> pl.Expr:
        try:
            arvore = ast.parse(expressao, mode="eval")
        except SyntaxError as erro_sintaxe:
            raise ErroMotorPolars(
                f"Sintaxe de expressão inválida: '{expressao}'"
            ) from erro_sintaxe
        return self._visitar_no(arvore.body, colunas_validas)

    def _obter_nome_coluna_subscript(self, no_slice: ast.AST) -> str:
        if isinstance(no_slice, ast.Constant) and isinstance(no_slice.value, str):
            return no_slice.value
        raise ErroMotorPolars(
            "O índice da coluna no DataFrame Polars deve ser uma string literal (ex: df['Metragem'])."
        )

    def _visitar_no(self, no: ast.AST, colunas_validas: set[str]) -> pl.Expr:
        if isinstance(no, ast.Constant):
            if isinstance(no.value, (int, float, bool, str)):
                return pl.lit(no.value)
            raise ErroMotorPolars(f"Tipo de constante não suportado: {type(no.value).__name__}")

        if isinstance(no, ast.Subscript):
            if isinstance(no.value, ast.Name) and no.value.id in self._NOMES_DATAFRAME_PERMITIDOS:
                coluna = self._obter_nome_coluna_subscript(no.slice)
                if coluna not in colunas_validas:
                    raise ErroMotorPolars(f"Coluna '{coluna}' inexistente no DataFrame Polars.")
                return pl.col(coluna)
            raise ErroMotorPolars("Indexação permitida apenas no formato df['nome_coluna'].")

        if isinstance(no, ast.Name):
            if no.id in self._NOMES_DATAFRAME_PERMITIDOS or no.id == "pl":
                raise ErroMotorPolars(
                    f"Identificador '{no.id}' deve ser usado como {no.id}['coluna'] ou pl.col('coluna')."
                )
            if no.id not in colunas_validas:
                raise ErroMotorPolars(f"Coluna '{no.id}' inexistente no DataFrame Polars.")
            return pl.col(no.id)

        if isinstance(no, ast.BinOp):
            tipo_op = type(no.op)
            if tipo_op not in self._OPERADORES_BINARIOS:
                raise ErroMotorPolars(f"Operador binário proibido no Polars: {tipo_op.__name__}")
            esq = self._visitar_no(no.left, colunas_validas)
            dir_ = self._visitar_no(no.right, colunas_validas)
            funcao_bin = self._OPERADORES_BINARIOS[tipo_op]
            return funcao_bin(esq, dir_)

        if isinstance(no, ast.UnaryOp):
            tipo_un = type(no.op)
            if tipo_un not in self._OPERADORES_UNARIOS:
                raise ErroMotorPolars(f"Operador unário proibido no Polars: {tipo_un.__name__}")
            op = self._visitar_no(no.operand, colunas_validas)
            funcao_un = self._OPERADORES_UNARIOS[tipo_un]
            return funcao_un(op)

        if isinstance(no, ast.Compare):
            if len(no.ops) != 1 or len(no.comparators) != 1:
                raise ErroMotorPolars("Comparações múltiplas encadeadas não são suportadas.")
            tipo_cmp = type(no.ops[0])
            if tipo_cmp not in self._OPERADORES_COMPARACAO:
                raise ErroMotorPolars(f"Operador de comparação proibido no Polars: {tipo_cmp.__name__}")
            esq = self._visitar_no(no.left, colunas_validas)
            dir_ = self._visitar_no(no.comparators[0], colunas_validas)
            funcao_cmp = self._OPERADORES_COMPARACAO[tipo_cmp]
            return funcao_cmp(esq, dir_)

        if isinstance(no, ast.BoolOp):
            valores = [self._visitar_no(v, colunas_validas) for v in no.values]
            if not valores:
                raise ErroMotorPolars("Operação booleana sem operandos no Polars.")
            resultado_bool = valores[0]
            for prox in valores[1:]:
                if isinstance(no.op, ast.And):
                    resultado_bool = resultado_bool & prox
                elif isinstance(no.op, ast.Or):
                    resultado_bool = resultado_bool | prox
                else:
                    raise ErroMotorPolars(
                        f"Operador lógico não suportado no Polars: {type(no.op).__name__}"
                    )
            return resultado_bool

        if isinstance(no, ast.Call):
            # Suporte a pl.col('coluna')
            if (
                isinstance(no.func, ast.Attribute)
                and isinstance(no.func.value, ast.Name)
                and no.func.value.id == "pl"
                and no.func.attr == "col"
            ):
                if (
                    not no.args
                    or not isinstance(no.args[0], ast.Constant)
                    or not isinstance(no.args[0].value, str)
                ):
                    raise ErroMotorPolars(
                        "pl.col deve receber uma string literal com o nome da coluna."
                    )
                nome_col = no.args[0].value
                if nome_col not in colunas_validas:
                    raise ErroMotorPolars(f"Coluna '{nome_col}' inexistente no DataFrame Polars.")
                return pl.col(nome_col)

            # Chamadas de métodos em colunas (ex: df['Metragem'].clip(...) ou pl.col('Metragem').clip(...))
            if not isinstance(no.func, ast.Attribute):
                raise ErroMotorPolars(
                    "Apenas chamadas de métodos permitidos em colunas são suportadas no Polars."
                )
            objeto = self._visitar_no(no.func.value, colunas_validas)
            nome_metodo = no.func.attr
            if nome_metodo not in self._METODOS_PERMITIDOS:
                raise ErroMotorPolars(
                    f"Método '{nome_metodo}' não é permitido por política de segurança no Polars."
                )

            metodo = getattr(objeto, nome_metodo, None)
            if not callable(metodo):
                raise ErroMotorPolars(f"Método '{nome_metodo}' não encontrado no objeto Polars.")

            args = [self._visitar_no(arg, colunas_validas) for arg in no.args]
            kwargs: dict[str, pl.Expr] = {}
            for kw in no.keywords:
                if kw.arg is None:
                    continue
                nome_arg = kw.arg
                if nome_metodo == "clip":
                    if nome_arg == "lower":
                        nome_arg = "lower_bound"
                    elif nome_arg == "upper":
                        nome_arg = "upper_bound"
                kwargs[nome_arg] = self._visitar_no(kw.value, colunas_validas)

            resultado: object = metodo(*args, **kwargs)
            if isinstance(resultado, pl.Expr):
                return resultado
            raise ErroMotorPolars(
                f"Retorno do método '{nome_metodo}' no Polars não é uma pl.Expr válida."
            )

        raise ErroMotorPolars(f"Expressão proibida por segurança no Polars: {type(no).__name__}")
