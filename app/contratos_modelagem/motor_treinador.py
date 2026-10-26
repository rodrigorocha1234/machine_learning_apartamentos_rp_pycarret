"""Contrato e modelos de dados para motores de treinamento e tuning de regressão."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd
from pydantic import BaseModel, Field

from app.esquema_configuracao import ConfiguracaoModelagem


class ResultadoTreinoModelo(BaseModel):
    """Métricas e metadados de um modelo específico treinado e ajustado."""

    nome_modelo: str = Field(min_length=1, description="Identificador do modelo")
    framework: str = Field(description="Framework utilizado (pycaret ou sklearn)")
    melhores_parametros: dict[str, object] = Field(
        default={}, description="Hiperparâmetros otimizados durante o tuning"
    )
    metricas: dict[str, float] = Field(
        default={}, description="Métricas de avaliação na validação cruzada (RMSE, MAE, R2)"
    )
    duracao_segundos: float = Field(
        ge=0.0, description="Duração do treinamento e tuning do modelo em segundos"
    )


class ResultadoTuning(BaseModel):
    """Sumário comparativo do tuning de hiperparâmetros de todos os modelos avaliados."""

    metodo_tuning: str = Field(description="Algoritmo de busca utilizado (random, grid, bayesian)")
    iteracoes: int = Field(ge=1, description="Número de iterações configuradas")
    modelos_treinados: list[ResultadoTreinoModelo] = Field(
        description="Lista de resultados por modelo"
    )
    modelo_campeao: str = Field(description="Nome do modelo com melhor performance (menor RMSE)")
    melhor_rmse: float = Field(description="Menor valor de RMSE obtido")


class ResultadoModelagem(BaseModel):
    """Evidências completas da execução da etapa de modelagem e registro no MLflow."""

    motor_utilizado: str = Field(description="Identificador do framework executor")
    total_modelos: int = Field(ge=1, description="Total de modelos ajustados")
    experimento_mlflow: str = Field(description="Nome do experimento registrado no MLflow")
    run_id_pai: str = Field(description="ID do run principal no MLflow")
    runs_filhos: dict[str, str] = Field(
        description="Mapeamento de nome de modelo para ID do run filho no MLflow"
    )
    modelo_campeao: str = Field(description="Modelo selecionado como campeão da rodada")
    duracao_total_segundos: float = Field(
        ge=0.0, description="Tempo total decorrido na etapa de modelagem"
    )


@runtime_checkable
class MotorTreinador(Protocol):
    """Protocolo GoF Strategy para motores de treinamento, pré-processamento e tuning."""

    def treinar_e_ajustar(
        self,
        dados_treino: pd.DataFrame,
        coluna_alvo: str,
        config: ConfiguracaoModelagem,
        folds: int = 3,
    ) -> ResultadoTuning:
        """Executa pré-processamento, treinamento e tuning dos modelos configurados."""
        ...

    def obter_nome_motor(self) -> str:
        """Retorna o identificador textual do motor (pycaret ou sklearn)."""
        ...
