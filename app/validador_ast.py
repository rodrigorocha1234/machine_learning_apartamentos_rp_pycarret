"""Módulo para validação estática segura de expressões Python via AST.

Garante que nenhuma expressão configurada em arquivos YAML contenha chamadas perigosas,
acesso a sistema de arquivos, rede, comandos do sistema operacional ou introspecção maliciosa.
"""

from __future__ import annotations

import ast
from typing import final


class ErroExpressaoInsegura(Exception):
    """Exceção levantada quando uma expressão contém nós AST ou operações não permitidas."""


@final
class ValidadorAst:
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
            # Operadores Aritméticos
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            # Operadores Unários
            ast.UAdd,
            ast.USub,
            ast.Not,
            # Operadores Booleanos
            ast.And,
            ast.Or,
            # Operadores de Comparação
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
        {
            "clip",
            "isin",
            "isna",
            "notna",
            "fillna",
            "astype",
            "round",
            "abs",
            "log",
            "log1p",
            "exp",
        }
    )

    _FUNCOES_PERMITIDAS: frozenset[str] = frozenset(
        {
            "abs",
            "round",
            "min",
            "max",
            "int",
            "float",
            "str",
            "bool",
        }
    )

    def validar_expressao(self, expressao: str) -> bool:
        """Valida se uma expressão é sintaticamente segura conforme lista positiva.

        Lança ErroExpressaoInsegura caso a expressão viole as restrições de segurança.
        """
        expressao_limpa = expressao.strip()
        if not expressao_limpa:
            raise ErroExpressaoInsegura("Expressão não pode ser vazia.")

        try:
            arvore = ast.parse(expressao_limpa, mode="eval")
        except SyntaxError as erro_sintaxe:
            raise ErroExpressaoInsegura(
                f"Erro de sintaxe na expressão '{expressao_limpa}': {erro_sintaxe}"
            ) from erro_sintaxe

        for no in ast.walk(arvore):
            tipo_no = type(no)
            if tipo_no not in self._NOS_PERMITIDOS:
                raise ErroExpressaoInsegura(
                    f"Tipo de nó AST não permitido: '{tipo_no.__name__}' na expressão: '{expressao_limpa}'"
                )

            # Bloqueio estrito de introspecção e atributos privados/duplos sublinhados
            if isinstance(no, ast.Name):
                if no.id.startswith("__") or no.id in {"eval", "exec", "open", "compile", "globals", "locals"}:
                    raise ErroExpressaoInsegura(
                        f"Identificador proibido detectado: '{no.id}' na expressão: '{expressao_limpa}'"
                    )

            if isinstance(no, ast.Attribute):
                if no.attr.startswith("__") or (no.attr not in self._METODOS_PERMITIDOS and no.attr not in {"str", "dt", "cat"}):
                    raise ErroExpressaoInsegura(
                        f"Acesso a atributo ou método não permitido: '{no.attr}' na expressão: '{expressao_limpa}'"
                    )

            if isinstance(no, ast.Call):
                # Validar se a função chamada é uma função ou método seguro
                func = no.func
                if isinstance(func, ast.Name):
                    if func.id not in self._FUNCOES_PERMITIDAS:
                        raise ErroExpressaoInsegura(
                            f"Chamada de função não permitida: '{func.id}()' na expressão: '{expressao_limpa}'"
                        )
                elif isinstance(func, ast.Attribute):
                    if func.attr not in self._METODOS_PERMITIDOS:
                        raise ErroExpressaoInsegura(
                            f"Chamada de método não permitida: '.{func.attr}()' na expressão: '{expressao_limpa}'"
                        )
                else:
                    raise ErroExpressaoInsegura(
                        f"Formato de chamada inválido na expressão: '{expressao_limpa}'"
                    )

        return True
