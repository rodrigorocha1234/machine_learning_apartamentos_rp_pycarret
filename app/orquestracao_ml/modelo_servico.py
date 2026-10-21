from __future__ import annotations
from typing import final
import mlflow.pyfunc
import pandas as pd
from app.contratos_dados.motor_treino import ModeloPreditor
from app.contratos_dados.regra_negocio import DefinicaoRegra
from app.regras_negocio.avaliador_regras import AvaliadorRegras

@final
class ModeloServico(mlflow.pyfunc.PythonModel):
    """Wrapper PyFunc nativo do MLflow integrando modelo campeão e regras de negócio dinâmicas via AST."""

    def __init__(
        self,
        modelo_campeao: ModeloPreditor,
        regras: list[DefinicaoRegra],
        coluna_alvo_predito: str = "preco_predito"
    ) -> None:
        self._modelo_campeao = modelo_campeao
        self._regras = regras
        self._coluna_alvo_predito = coluna_alvo_predito
        self._avaliador = AvaliadorRegras()

    def predict(self, context: mlflow.pyfunc.PythonModelContext | None, model_input: pd.DataFrame) -> pd.DataFrame:
        dados = model_input.copy()
        
        # Suporta predição tanto com scikit-learn Pipeline quanto com objeto com método predict
        if hasattr(self._modelo_campeao, "predict"):
            predicoes = self._modelo_campeao.predict(dados)
        else:
            raise RuntimeError("Modelo campeão não possui método predict compatível.")

        dados[self._coluna_alvo_predito] = predicoes

        # Aplica dinamicamente as regras de negócio configuradas no YAML via AST vetorizado
        resultado = self._avaliador.aplicar_regra(dados, self._regras)
        return resultado
