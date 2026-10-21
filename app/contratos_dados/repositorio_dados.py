from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class RepositorioDados(Protocol):
    def gravar_dados(self, dados: pd.DataFrame) -> None: ...
    def ler_dados(self) -> pd.DataFrame: ...
