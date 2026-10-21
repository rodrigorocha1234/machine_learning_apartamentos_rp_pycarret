from typing import final
import pandas as pd

@final
class AnaliseBasica:
    def resumir_dados(self, dados: pd.DataFrame) -> pd.DataFrame:
        return dados.describe(include="all").transpose()
