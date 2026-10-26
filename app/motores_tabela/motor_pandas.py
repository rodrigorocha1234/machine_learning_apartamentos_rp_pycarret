"""Implementação de MotorTabela baseada em Pandas vetorizado com AST seguro."""

from __future__ import annotations

import ast
from typing import cast, final, override
import pandas as pd

from app.contratos_dados.motor_tabela import MotorTabela


class ErroMotorPandas(Exception):
    """Exceção levantada em falhas de processamento ou avaliação do motor Pandas."""


type OperandoPandas = pd.Series | float | int | bool | str


@final
class MotorPandas(MotorTabela):
    """Motor tabular vetorizado de alta performance para Pandas."""

    _METODOS_PERMITIDOS: frozenset[str] = frozenset({"clip", "abs", "round"})
    _NOMES_DATAFRAME_PERMITIDOS: frozenset[str] = frozenset({"df", "dados", "data", "tabela"})

    @override
    def obter_nome_motor(self) -> str:
        return "pandas"

    @override
    def filtrar_dados(self, dados: pd.DataFrame, expressao: str) -> pd.DataFrame:
        if not expressao.strip():
            return dados.copy()

        try:
            mascara = self._avaliar_expressao(expressao, dados)
            if isinstance(mascara, pd.Series):
                mascara_booleana = mascara.astype(bool)
                resultado_filtrado = cast(
                    pd.DataFrame, dados.loc[mascara_booleana]
                ).reset_index(drop=True)
                return resultado_filtrado
            if isinstance(mascara, bool):
                if mascara:
                    return dados.copy()
                df_vazio = cast(pd.DataFrame, dados.iloc[0:0]).copy()
                return df_vazio
            raise ErroMotorPandas(
                f"A expressão '{expressao}' não retornou uma máscara booleana."
            )
        except Exception as erro:
            raise ErroMotorPandas(
                f"Falha ao filtrar dados no Pandas com expressão '{expressao}': {erro}"
            ) from erro

    @override
    def transformar_dados(
        self, dados: pd.DataFrame, expressao: str, destino: str
    ) -> pd.DataFrame:
        try:
            serie = self._avaliar_expressao(expressao, dados)
            return dados.assign(**{destino: serie})
        except Exception as erro:
            raise ErroMotorPandas(
                f"Falha ao transformar coluna '{destino}' no Pandas com '{expressao}': {erro}"
            ) from erro

    @override
    def selecionar_colunas(self, dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
        colunas_inexistentes = [c for c in colunas if c not in dados.columns]
        if colunas_inexistentes:
            raise ErroMotorPandas(
                f"Colunas não encontradas no DataFrame: {colunas_inexistentes}"
            )
        colunas_dict: dict[str, pd.Series] = {}
        for col in colunas:
            coluna_serie = dados[col]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(coluna_serie, pd.Series):
                colunas_dict[col] = coluna_serie.copy()
            else:
                raise ErroMotorPandas(f"Coluna duplicada '{col}' no DataFrame.")
        return pd.DataFrame(colunas_dict)

    def _avaliar_expressao(self, expressao: str, dados: pd.DataFrame) -> OperandoPandas:
        try:
            arvore = ast.parse(expressao, mode="eval")
        except SyntaxError as erro_sintaxe:
            raise ErroMotorPandas(
                f"Expressão com sintaxe inválida: '{expressao}'"
            ) from erro_sintaxe
        return self._visitar_no(arvore.body, dados)

    def _obter_nome_coluna_subscript(self, no_slice: ast.AST) -> str:
        if isinstance(no_slice, ast.Constant) and isinstance(no_slice.value, str):
            return no_slice.value
        raise ErroMotorPandas(
            "O índice da coluna no DataFrame deve ser uma string literal (ex: df['Metragem'])."
        )

    def _visitar_no(self, no: ast.AST, dados: pd.DataFrame) -> OperandoPandas:
        if isinstance(no, ast.Constant):
            if isinstance(no.value, (int, float, bool, str)):
                return no.value
            raise ErroMotorPandas(f"Tipo de constante não suportado: {type(no.value).__name__}")

        if isinstance(no, ast.Subscript):
            if isinstance(no.value, ast.Name) and no.value.id in self._NOMES_DATAFRAME_PERMITIDOS:
                coluna = self._obter_nome_coluna_subscript(no.slice)
                if coluna not in dados.columns:
                    raise ErroMotorPandas(f"Coluna '{coluna}' não encontrada no DataFrame.")
                serie_obj = dados[coluna]  # pyright: ignore[reportUnknownVariableType]
                if isinstance(serie_obj, pd.Series):
                    return serie_obj
                raise ErroMotorPandas(f"Coluna duplicada '{coluna}' no DataFrame.")
            raise ErroMotorPandas("Indexação permitida apenas no formato df['nome_coluna'].")

        if isinstance(no, ast.Name):
            if no.id in self._NOMES_DATAFRAME_PERMITIDOS:
                raise ErroMotorPandas(
                    f"O identificador '{no.id}' deve ser indexado por coluna (ex: {no.id}['coluna'])."
                )
            if no.id not in dados.columns:
                raise ErroMotorPandas(f"Coluna '{no.id}' não encontrada no DataFrame.")
            serie_col_obj = dados[no.id]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(serie_col_obj, pd.Series):
                return serie_col_obj
            raise ErroMotorPandas(f"Coluna duplicada '{no.id}' no DataFrame.")

        if isinstance(no, ast.BinOp):
            esq = self._visitar_no(no.left, dados)
            dir_ = self._visitar_no(no.right, dados)
            return self._executar_binop(no.op, esq, dir_)

        if isinstance(no, ast.UnaryOp):
            operando = self._visitar_no(no.operand, dados)
            return self._executar_unaryop(no.op, operando)

        if isinstance(no, ast.Compare):
            if len(no.ops) != 1 or len(no.comparators) != 1:
                raise ErroMotorPandas("Comparações múltiplas encadeadas não são suportadas.")
            esq = self._visitar_no(no.left, dados)
            dir_ = self._visitar_no(no.comparators[0], dados)
            return self._executar_cmpop(no.ops[0], esq, dir_)

        if isinstance(no, ast.BoolOp):
            valores = [self._visitar_no(v, dados) for v in no.values]
            if not valores:
                raise ErroMotorPandas("Operação booleana sem operandos.")
            resultado_bool: OperandoPandas = valores[0]
            for proximo in valores[1:]:
                if isinstance(no.op, ast.And):
                    if isinstance(resultado_bool, pd.Series):
                        resultado_bool = cast(pd.Series, resultado_bool & proximo)
                    elif isinstance(proximo, pd.Series):
                        resultado_bool = cast(pd.Series, proximo & resultado_bool)
                    else:
                        resultado_bool = bool(resultado_bool) and bool(proximo)
                elif isinstance(no.op, ast.Or):
                    if isinstance(resultado_bool, pd.Series):
                        resultado_bool = cast(pd.Series, resultado_bool | proximo)
                    elif isinstance(proximo, pd.Series):
                        resultado_bool = cast(pd.Series, proximo | resultado_bool)
                    else:
                        resultado_bool = bool(resultado_bool) or bool(proximo)
                else:
                    raise ErroMotorPandas(f"Operador lógico não suportado: {type(no.op).__name__}")
            return resultado_bool

        if isinstance(no, ast.Call):
            if not isinstance(no.func, ast.Attribute):
                raise ErroMotorPandas(
                    "Apenas chamadas de métodos permitidos em colunas são suportadas."
                )
            objeto = self._visitar_no(no.func.value, dados)
            nome_metodo = no.func.attr
            if nome_metodo not in self._METODOS_PERMITIDOS:
                raise ErroMotorPandas(
                    f"Método '{nome_metodo}' não é permitido por política de segurança."
                )

            metodo = getattr(objeto, nome_metodo, None)
            if not callable(metodo):
                raise ErroMotorPandas(f"Método '{nome_metodo}' não encontrado no objeto.")

            args = [self._visitar_no(arg, dados) for arg in no.args]
            kwargs = {
                kw.arg: self._visitar_no(kw.value, dados)
                for kw in no.keywords
                if kw.arg is not None
            }
            resultado_metodo: object = metodo(*args, **kwargs)
            if isinstance(resultado_metodo, pd.Series):
                return resultado_metodo
            if isinstance(resultado_metodo, (int, float, bool, str)):
                return resultado_metodo

            raise ErroMotorPandas(
                f"Retorno do método '{nome_metodo}' tem tipo inesperado: {type(resultado_metodo).__name__}"
            )

        raise ErroMotorPandas(f"Expressão proibida por segurança: {type(no).__name__}")

    def _executar_binop(
        self, op: ast.operator, esq: OperandoPandas, dir_: OperandoPandas
    ) -> OperandoPandas:
        if isinstance(esq, pd.Series):
            if isinstance(op, ast.Add):
                return cast(pd.Series, esq.add(dir_))
            if isinstance(op, ast.Sub):
                return cast(pd.Series, esq.sub(dir_))
            if isinstance(op, ast.Mult):
                return cast(pd.Series, esq.mul(dir_))
            if isinstance(op, ast.Div):
                return cast(pd.Series, esq.div(dir_))
            if isinstance(op, ast.FloorDiv):
                return cast(pd.Series, esq.floordiv(dir_))
            if isinstance(op, ast.Mod):
                return cast(pd.Series, esq.mod(dir_))
            if isinstance(op, ast.Pow):
                return cast(pd.Series, esq.pow(dir_))
            if isinstance(op, ast.BitAnd):
                return cast(pd.Series, esq & dir_)
            if isinstance(op, ast.BitOr):
                return cast(pd.Series, esq | dir_)
        elif isinstance(dir_, pd.Series):
            if isinstance(op, ast.Add):
                return cast(pd.Series, dir_.radd(esq))
            if isinstance(op, ast.Sub):
                return cast(pd.Series, dir_.rsub(esq))
            if isinstance(op, ast.Mult):
                return cast(pd.Series, dir_.rmul(esq))
            if isinstance(op, ast.Div):
                return cast(pd.Series, dir_.rtruediv(esq))
            if isinstance(op, ast.FloorDiv):
                return cast(pd.Series, dir_.rfloordiv(esq))
            if isinstance(op, ast.Mod):
                return cast(pd.Series, dir_.rmod(esq))
            if isinstance(op, ast.Pow):
                return cast(pd.Series, dir_.rpow(esq))
            if isinstance(op, ast.BitAnd):
                return cast(pd.Series, dir_ & esq)
            if isinstance(op, ast.BitOr):
                return cast(pd.Series, dir_ | esq)
        elif isinstance(esq, (int, float)) and isinstance(dir_, (int, float)):
            if isinstance(op, ast.Add):
                return esq + dir_
            if isinstance(op, ast.Sub):
                return esq - dir_
            if isinstance(op, ast.Mult):
                return esq * dir_
            if isinstance(op, ast.Div):
                return esq / dir_
            if isinstance(op, ast.FloorDiv):
                return esq // dir_
            if isinstance(op, ast.Mod):
                return esq % dir_
            if isinstance(op, ast.Pow):
                return esq**dir_
            if isinstance(esq, int) and isinstance(dir_, int):
                if isinstance(op, ast.BitAnd):
                    return esq & dir_
                if isinstance(op, ast.BitOr):
                    return esq | dir_
        elif isinstance(esq, str) and isinstance(dir_, str) and isinstance(op, ast.Add):
            return esq + dir_
        raise ErroMotorPandas(
            f"Operação binária inválida entre tipos {type(esq).__name__} e {type(dir_).__name__}"
        )

    def _executar_unaryop(
        self, op: ast.unaryop, operando: OperandoPandas
    ) -> OperandoPandas:
        if isinstance(operando, pd.Series):
            if isinstance(op, ast.UAdd):
                return operando
            if isinstance(op, ast.USub):
                return -operando
            if isinstance(op, ast.Not):
                return ~operando
        elif isinstance(operando, bool):
            if isinstance(op, ast.Not):
                return not operando
        elif isinstance(operando, (int, float)):
            if isinstance(op, ast.UAdd):
                return +operando
            if isinstance(op, ast.USub):
                return -operando
            if isinstance(op, ast.Not):
                return not operando
        raise ErroMotorPandas(
            f"Operador unário proibido ou incompatível: {type(op).__name__}"
        )

    def _executar_cmpop(
        self, op: ast.cmpop, esq: OperandoPandas, dir_: OperandoPandas
    ) -> OperandoPandas:
        if isinstance(esq, pd.Series):
            if isinstance(op, ast.Eq):
                return cast(pd.Series, esq.eq(dir_))
            if isinstance(op, ast.NotEq):
                return cast(pd.Series, esq.ne(dir_))
            if isinstance(op, ast.Lt):
                return cast(pd.Series, esq.lt(dir_))
            if isinstance(op, ast.LtE):
                return cast(pd.Series, esq.le(dir_))
            if isinstance(op, ast.Gt):
                return cast(pd.Series, esq.gt(dir_))
            if isinstance(op, ast.GtE):
                return cast(pd.Series, esq.ge(dir_))
        elif isinstance(dir_, pd.Series):
            if isinstance(op, ast.Eq):
                return cast(pd.Series, dir_.eq(esq))
            if isinstance(op, ast.NotEq):
                return cast(pd.Series, dir_.ne(esq))
            if isinstance(op, ast.Lt):
                return cast(pd.Series, dir_.gt(esq))
            if isinstance(op, ast.LtE):
                return cast(pd.Series, dir_.ge(esq))
            if isinstance(op, ast.Gt):
                return cast(pd.Series, dir_.lt(esq))
            if isinstance(op, ast.GtE):
                return cast(pd.Series, dir_.le(esq))
        elif isinstance(esq, (int, float)) and isinstance(dir_, (int, float)):
            if isinstance(op, ast.Eq):
                return esq == dir_
            if isinstance(op, ast.NotEq):
                return esq != dir_
            if isinstance(op, ast.Lt):
                return esq < dir_
            if isinstance(op, ast.LtE):
                return esq <= dir_
            if isinstance(op, ast.Gt):
                return esq > dir_
            if isinstance(op, ast.GtE):
                return esq >= dir_
        elif isinstance(esq, str) and isinstance(dir_, str):
            if isinstance(op, ast.Eq):
                return esq == dir_
            if isinstance(op, ast.NotEq):
                return esq != dir_
        raise ErroMotorPandas(
            f"Operação de comparação inválida entre {type(esq).__name__} e {type(dir_).__name__}"
        )
