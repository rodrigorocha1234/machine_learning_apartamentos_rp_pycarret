from __future__ import annotations
import ast
from collections.abc import Callable
import functools
import operator
from typing import final, override
import pandas as pd
from app.contratos_dados.motor_tabela import MotorTabela

class ErroMotorTabela(Exception):
    """Exceção levantada em operações do motor tabular."""
    pass

type OperandoAst = pd.Series[float] | pd.Series[int] | pd.Series[bool] | pd.Series[str] | float | int | bool | str

@final
class MotorPandas(MotorTabela):
    """Implementação vetorizada em pandas do protocolo MotorTabela com AST seguro."""

    _OPERADORES_BINARIOS: dict[type[ast.operator], Callable[[OperandoAst, OperandoAst], OperandoAst]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
    }

    _OPERADORES_UNARIOS: dict[type[ast.unaryop], Callable[[OperandoAst], OperandoAst]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    _OPERADORES_COMPARACAO: dict[type[ast.cmpop], Callable[[OperandoAst, OperandoAst], OperandoAst]] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    _METODOS_PERMITIDOS: set[str] = {"clip", "abs", "round"}
    _NOMES_DATAFRAME_PERMITIDOS: set[str] = {"df", "dados", "data", "tabela"}

    @override
    def filtrar_dados(self, dados: pd.DataFrame, expressao: str) -> pd.DataFrame:
        if not expressao.strip():
            return dados.copy()
        try:
            mascara = self._avaliar_expressao(expressao, dados)
            if isinstance(mascara, pd.Series):
                return dados.loc[mascara.astype(bool)].reset_index(drop=True)
            if isinstance(mascara, bool):
                return dados.copy() if mascara else dados.iloc[0:0].copy()
            raise ErroMotorTabela(f"A expressão '{expressao}' não resultou em máscara booleana.")
        except Exception as erro:
            raise ErroMotorTabela(f"Falha ao filtrar dados com expressão '{expressao}': {erro}") from erro

    @override
    def transformar_dados(self, dados: pd.DataFrame, expressao: str, destino: str) -> pd.DataFrame:
        try:
            serie = self._avaliar_expressao(expressao, dados)
            return dados.assign(**{destino: serie})
        except Exception as erro:
            raise ErroMotorTabela(f"Falha ao transformar dados para '{destino}' com '{expressao}': {erro}") from erro

    @override
    def selecionar_colunas(self, dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
        colunas_inexistentes = [c for c in colunas if c not in dados.columns]
        if colunas_inexistentes:
            raise ErroMotorTabela(f"Colunas não encontradas no DataFrame: {colunas_inexistentes}")
        return dados[colunas].copy()

    def _avaliar_expressao(self, expressao: str, dados: pd.DataFrame) -> OperandoAst:
        try:
            arvore = ast.parse(expressao, mode="eval")
        except SyntaxError as erro_sintaxe:
            raise ErroMotorTabela(f"Expressão com sintaxe inválida: '{expressao}'") from erro_sintaxe
        return self._visitar_no(arvore.body, dados)

    def _obter_nome_coluna_subscript(self, no_slice: ast.AST) -> str:
        if isinstance(no_slice, ast.Constant) and isinstance(no_slice.value, str):
            return no_slice.value
        raise ErroMotorTabela("O índice da coluna no DataFrame deve ser uma string literal (ex: df['Metragem']).")

    def _visitar_no(self, no: ast.AST, dados: pd.DataFrame) -> OperandoAst:
        if isinstance(no, ast.Constant):
            if isinstance(no.value, (int, float, bool, str)):
                return no.value
            raise ErroMotorTabela(f"Tipo de constante proibido: {type(no.value).__name__}")

        if isinstance(no, ast.Subscript):
            if isinstance(no.value, ast.Name) and no.value.id in self._NOMES_DATAFRAME_PERMITIDOS:
                coluna = self._obter_nome_coluna_subscript(no.slice)
                if coluna not in dados.columns:
                    raise ErroMotorTabela(f"Coluna '{coluna}' não existe no DataFrame.")
                return dados[coluna]
            raise ErroMotorTabela("Indexação permitida apenas no formato df['nome_coluna'].")

        if isinstance(no, ast.Name):
            if no.id in self._NOMES_DATAFRAME_PERMITIDOS:
                raise ErroMotorTabela(f"O identificador '{no.id}' deve ser indexado por coluna (ex: {no.id}['coluna']).")
            if no.id not in dados.columns:
                raise ErroMotorTabela(f"Coluna '{no.id}' não existe no DataFrame.")
            return dados[no.id]

        if isinstance(no, ast.BinOp):
            tipo_op = type(no.op)
            if tipo_op not in self._OPERADORES_BINARIOS:
                raise ErroMotorTabela(f"Operador binário proibido: {tipo_op.__name__}")
            esq = self._visitar_no(no.left, dados)
            dir_ = self._visitar_no(no.right, dados)
            funcao_bin = self._OPERADORES_BINARIOS[tipo_op]
            return funcao_bin(esq, dir_)

        if isinstance(no, ast.UnaryOp):
            tipo_un = type(no.op)
            if tipo_un not in self._OPERADORES_UNARIOS:
                raise ErroMotorTabela(f"Operador unário proibido: {tipo_un.__name__}")
            op = self._visitar_no(no.operand, dados)
            funcao_un = self._OPERADORES_UNARIOS[tipo_un]
            return funcao_un(op)

        if isinstance(no, ast.Compare):
            if len(no.ops) != 1 or len(no.comparators) != 1:
                raise ErroMotorTabela("Comparações múltiplas encadeadas não são suportadas.")
            tipo_cmp = type(no.ops[0])
            if tipo_cmp not in self._OPERADORES_COMPARACAO:
                raise ErroMotorTabela(f"Operador de comparação proibido: {tipo_cmp.__name__}")
            esq = self._visitar_no(no.left, dados)
            dir_ = self._visitar_no(no.comparators[0], dados)
            funcao_cmp = self._OPERADORES_COMPARACAO[tipo_cmp]
            return funcao_cmp(esq, dir_)

        if isinstance(no, ast.BoolOp):
            valores = [self._visitar_no(v, dados) for v in no.values]
            if isinstance(no.op, ast.And):
                return functools.reduce(operator.and_, valores)
            if isinstance(no.op, ast.Or):
                return functools.reduce(operator.or_, valores)
            raise ErroMotorTabela(f"Operador lógico não suportado: {type(no.op).__name__}")

        if isinstance(no, ast.Call):
            if not isinstance(no.func, ast.Attribute):
                raise ErroMotorTabela("Apenas chamadas de métodos permitidos em colunas são suportadas.")
            objeto = self._visitar_no(no.func.value, dados)
            nome_metodo = no.func.attr
            if nome_metodo not in self._METODOS_PERMITIDOS:
                raise ErroMotorTabela(f"Método '{nome_metodo}' não é permitido por política de segurança.")
            metodo = getattr(objeto, nome_metodo, None)
            if not callable(metodo):
                raise ErroMotorTabela(f"Método '{nome_metodo}' não encontrado no objeto.")
            args = [self._visitar_no(arg, dados) for arg in no.args]
            kwargs = {
                kw.arg: self._visitar_no(kw.value, dados)
                for kw in no.keywords
                if kw.arg is not None
            }
            resultado = metodo(*args, **kwargs)
            if isinstance(resultado, (pd.Series, int, float, bool, str)):
                return resultado
            raise ErroMotorTabela(f"Retorno do método '{nome_metodo}' tem tipo inesperado: {type(resultado).__name__}")

        raise ErroMotorTabela(f"Expressão proibida por segurança: {type(no).__name__}")
