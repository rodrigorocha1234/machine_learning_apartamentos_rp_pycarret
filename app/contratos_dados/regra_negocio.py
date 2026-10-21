from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import pandas as pd

@dataclass(frozen=True)
class DefinicaoRegra:
    nome: str
    ativo: bool
    descricao: str
    destino: str
    expressao: str

@runtime_checkable
class RegraNegocio(Protocol):
    def aplicar_regra(self, dados: pd.DataFrame, regras: list[DefinicaoRegra]) -> pd.DataFrame: ...

