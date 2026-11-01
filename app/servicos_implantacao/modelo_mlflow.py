"""Modelo PyFunc autônomo para MLflow Serving com avaliação dinâmica de regras de negócio."""

from __future__ import annotations

import ast
from types import CodeType
from typing import cast
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext
import numpy as np
import pandas as pd


class ErroSegurancaExpressao(Exception):
    """Exceção levantada quando uma expressão contém nós AST ou operações não permitidas."""


class ValidadorAstInterno:
    """Validador sintático com lista positiva (whitelist) para expressões de configuração."""

    _NOS_PERMITIDOS: frozenset[type[ast.AST]] = frozenset(
        {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Subscript,
            ast.Slice,
            ast.Attribute,
            ast.Call,
            ast.keyword,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.And,
            ast.Or,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
        }
    )

    _METODOS_PERMITIDOS: frozenset[str] = frozenset(
        {"clip", "isin", "isna", "notna", "fillna", "astype", "round", "abs", "log", "log1p", "exp"}
    )

    _FUNCOES_PERMITIDAS: frozenset[str] = frozenset(
        {"abs", "round", "min", "max", "int", "float", "str", "bool"}
    )

    def validar_expressao(self, expressao: str) -> bool:
        """Valida se uma expressão é sintaticamente segura conforme lista positiva."""
        expressao_limpa = expressao.strip()
        if not expressao_limpa:
            raise ErroSegurancaExpressao("Expressão não pode ser vazia.")

        try:
            arvore = ast.parse(expressao_limpa, mode="eval")
        except SyntaxError as erro_sintaxe:
            raise ErroSegurancaExpressao(
                f"Erro de sintaxe na expressão '{expressao_limpa}': {erro_sintaxe}"
            ) from erro_sintaxe

        for no in ast.walk(arvore):
            tipo_no = type(no)
            if tipo_no not in self._NOS_PERMITIDOS:
                raise ErroSegurancaExpressao(
                    f"Tipo de nó AST não permitido: '{tipo_no.__name__}' na expressão: '{expressao_limpa}'"
                )

            if isinstance(no, ast.Name):
                if no.id.startswith("__") or no.id in {"eval", "exec", "open", "compile", "globals", "locals"}:
                    raise ErroSegurancaExpressao(
                        f"Identificador proibido detectado: '{no.id}' na expressão: '{expressao_limpa}'"
                    )

            if isinstance(no, ast.Attribute):
                if no.attr.startswith("__") or (
                    no.attr not in self._METODOS_PERMITIDOS and no.attr not in {"str", "dt", "cat"}
                ):
                    raise ErroSegurancaExpressao(
                        f"Acesso a atributo ou método não permitido: '{no.attr}' na expressão: '{expressao_limpa}'"
                    )

            if isinstance(no, ast.Call):
                func = no.func
                if isinstance(func, ast.Name):
                    if func.id not in self._FUNCOES_PERMITIDAS:
                        raise ErroSegurancaExpressao(
                            f"Chamada de função não permitida: '{func.id}()' na expressão: '{expressao_limpa}'"
                        )
                elif isinstance(func, ast.Attribute):
                    if func.attr not in self._METODOS_PERMITIDOS:
                        raise ErroSegurancaExpressao(
                            f"Chamada de método não permitida: '.{func.attr}()' na expressão: '{expressao_limpa}'"
                        )
                else:
                    raise ErroSegurancaExpressao(
                        f"Formato de chamada inválido na expressão: '{expressao_limpa}'"
                    )

        return True


class ModeloComRegrasNegocio(PythonModel):
    """Encapsula o estimador campeão e avalia regras de negócio de 06_implantacao_regras.yaml."""

    def __init__(
        self,
        modelo_estimador: object,
        regras_configuradas: list[dict[str, object]],
        coluna_alvo_predito: str = "preco_predito",
    ) -> None:
        self.modelo_estimador = modelo_estimador
        self.regras_configuradas = regras_configuradas
        self.coluna_alvo_predito = coluna_alvo_predito
        self._codigos_compilados: list[tuple[str, str, CodeType]] = []
        self._inicializar_regras()

    def _inicializar_regras(self) -> None:
        """Valida com AST seguro e compila as expressões das regras de negócio ativas."""
        validador = ValidadorAstInterno()
        for r in self.regras_configuradas:
            if not bool(r.get("ativo", True)):
                continue
            nome = str(r.get("nome", ""))
            destino = str(r.get("destino", ""))
            expressao = str(r.get("expressao", ""))
            validador.validar_expressao(expressao)
            codigo = compile(expressao, f"<regra_{nome}>", "eval")
            self._codigos_compilados.append((nome, destino, codigo))

    def predict(
        self,
        context: PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Executa a predição do modelo base e enriquece com os resultados das regras de negócio."""
        if not hasattr(self.modelo_estimador, "predict"):
            raise ValueError("O estimador configurado não implementa método 'predict'.")

        preditor = self.modelo_estimador
        predicoes_brutas = getattr(preditor, "predict")(model_input)
        array_predicoes = np.asarray(predicoes_brutas, dtype=np.float64)
        array_predicoes = np.clip(array_predicoes, a_min=10_000.0, a_max=None)

        df_resultado = pd.DataFrame(index=model_input.index)
        df_resultado[self.coluna_alvo_predito] = np.round(array_predicoes, 2)

        # Preparar dados de entrada para o contexto das regras
        registros_input = cast(list[dict[str, object]], model_input.to_dict(orient="records"))
        predicoes_lista = [round(float(v), 2) for v in array_predicoes.tolist()]

        linhas_regras: list[dict[str, float]] = []
        ambiente_execucao: dict[str, object] = {"__builtins__": {}}

        for idx, reg in enumerate(registros_input):
            contexto_linha: dict[str, object] = {
                self.coluna_alvo_predito: predicoes_lista[idx]
            }
            for k, val in reg.items():
                if isinstance(val, (int, float, str)):
                    contexto_linha[str(k)] = val
                else:
                    try:
                        contexto_linha[str(k)] = float(str(val))
                    except (ValueError, TypeError):
                        contexto_linha[str(k)] = str(val)

            linha_calculada: dict[str, float] = {}
            for _nome, destino, codigo in self._codigos_compilados:
                try:
                    res = eval(codigo, ambiente_execucao, contexto_linha)
                    val_float = round(float(cast(float | int, res)), 2)
                    linha_calculada[destino] = val_float
                    contexto_linha[destino] = val_float
                except ZeroDivisionError:
                    linha_calculada[destino] = 0.0
                    contexto_linha[destino] = 0.0
                except Exception as erro:
                    raise RuntimeError(f"Falha ao calcular regra '{destino}': {erro}") from erro

            linhas_regras.append(linha_calculada)

        if linhas_regras:
            df_regras = pd.DataFrame(linhas_regras, index=model_input.index)
            for col in df_regras.columns:
                df_resultado[col] = df_regras[col]

        return df_resultado
