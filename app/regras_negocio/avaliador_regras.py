import ast
from collections.abc import Callable
import operator
from typing import final, override
import pandas as pd
from app.contratos_dados.regra_negocio import DefinicaoRegra, RegraNegocio

class ErroRegraNegocio(Exception):
    """Exceção levantada em caso de falha na validação ou avaliação da regra de negócio."""
    pass

type OperandoVetorizado = pd.Series[float] | pd.Series[int] | pd.Series[bool] | float | int | bool

@final
class AvaliadorRegras(RegraNegocio):
    """Avaliador de regras de negócio dinâmicas via AST seguro sobre DataFrames."""

    _OPERADORES_BINARIOS: dict[type[ast.operator], Callable[[OperandoVetorizado, OperandoVetorizado], OperandoVetorizado]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _OPERADORES_UNARIOS: dict[type[ast.unaryop], Callable[[OperandoVetorizado], OperandoVetorizado]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    _OPERADORES_COMPARACAO: dict[type[ast.cmpop], Callable[[OperandoVetorizado, OperandoVetorizado], OperandoVetorizado]] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    @override
    def aplicar_regra(self, dados: pd.DataFrame, regras: list[DefinicaoRegra]) -> pd.DataFrame:
        resultado = dados.copy()
        for regra in regras:
            if not regra.ativo:
                continue
            try:
                serie_calculada = self._avaliar_expressao(regra.expressao, resultado)
                resultado = resultado.assign(**{regra.destino: serie_calculada})
            except Exception as erro:
                raise ErroRegraNegocio(
                    f"Falha ao avaliar a regra de negócio '{regra.nome}' com a expressão '{regra.expressao}': {erro}"
                ) from erro
        return resultado

    def _avaliar_expressao(self, expressao: str, dados: pd.DataFrame) -> OperandoVetorizado:
        try:
            arvore = ast.parse(expressao, mode="eval")
        except SyntaxError as erro_sintaxe:
            raise ErroRegraNegocio(f"Expressão com sintaxe inválida: {expressao}") from erro_sintaxe

        return self._visitar_no(arvore.body, dados)

    def _visitar_no(self, no: ast.AST, dados: pd.DataFrame) -> OperandoVetorizado:
        if isinstance(no, ast.Constant):
            if isinstance(no.value, (int, float, bool)):
                return no.value
            raise ErroRegraNegocio(f"Constante do tipo {type(no.value).__name__} não permitida.")

        if isinstance(no, ast.Name):
            if no.id not in dados.columns:
                raise ErroRegraNegocio(f"Coluna ou identificador não encontrado no DataFrame: {no.id}")
            coluna: pd.Series[float] = dados[no.id]
            return coluna

        if isinstance(no, ast.BinOp):
            tipo_operador = type(no.op)
            if tipo_operador not in self._OPERADORES_BINARIOS:
                raise ErroRegraNegocio(f"Operador binário não permitido: {tipo_operador.__name__}")
            esquerda = self._visitar_no(no.left, dados)
            direita = self._visitar_no(no.right, dados)
            funcao_binaria = self._OPERADORES_BINARIOS[tipo_operador]
            return funcao_binaria(esquerda, direita)

        if isinstance(no, ast.UnaryOp):
            tipo_unario = type(no.op)
            if tipo_unario not in self._OPERADORES_UNARIOS:
                raise ErroRegraNegocio(f"Operador unário não permitido: {tipo_unario.__name__}")
            operando = self._visitar_no(no.operand, dados)
            funcao_unaria = self._OPERADORES_UNARIOS[tipo_unario]
            return funcao_unaria(operando)

        if isinstance(no, ast.Compare):
            if len(no.ops) != 1 or len(no.comparators) != 1:
                raise ErroRegraNegocio("Comparações encadeadas múltiplas não são permitidas.")
            tipo_comp = type(no.ops[0])
            if tipo_comp not in self._OPERADORES_COMPARACAO:
                raise ErroRegraNegocio(f"Operador de comparação não permitido: {tipo_comp.__name__}")
            esquerda = self._visitar_no(no.left, dados)
            direita = self._visitar_no(no.comparators[0], dados)
            funcao_comp = self._OPERADORES_COMPARACAO[tipo_comp]
            return funcao_comp(esquerda, direita)

        raise ErroRegraNegocio(f"Operação proibida por política de segurança: {type(no).__name__}")
