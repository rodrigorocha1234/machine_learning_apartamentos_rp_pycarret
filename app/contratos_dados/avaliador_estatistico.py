from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import pandas as pd


@dataclass(frozen=True)
class ResultadoEstatistico:
    estatistica_friedman: float
    valor_p_friedman: float
    hipotese_nula_rejeitada: bool
    postos_medios: pd.Series[float]
    comparacao_pos_hoc: pd.DataFrame
    modelo_campeao: str

@runtime_checkable
class AvaliadorEstatistico(Protocol):
    def testar_diferencas(self, perdas_pareadas: pd.DataFrame, alpha: float) -> ResultadoEstatistico: ...
