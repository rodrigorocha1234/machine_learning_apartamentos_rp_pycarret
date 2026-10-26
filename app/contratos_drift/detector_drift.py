"""Contratos e esquemas de dados tipados para detecção estatística de drift e qualidade."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd
from pydantic import BaseModel, Field


class MetricasQualidadeDados(BaseModel):
    """Métricas de qualidade, integridade e conformidade de esquema dos dados."""

    total_amostras: int = Field(ge=0, description="Total de observações na janela analisada")
    taxa_nulos_geral: float = Field(ge=0.0, le=1.0, description="Proporção geral de valores nulos")
    contagem_nulos_por_coluna: dict[str, int] = Field(
        default={}, description="Nulos absolutos por feature"
    )
    novas_categorias_detectadas: dict[str, list[str]] = Field(
        default={}, description="Categorias observadas em produção ausentes na referência"
    )
    valores_fora_limites: dict[str, int] = Field(
        default={}, description="Quantidade de valores fora dos limites mínimo/máximo históricos"
    )


class ResultadoDriftVariavel(BaseModel):
    """Diagnóstico estatístico detalhado de drift em uma feature ou alvo."""

    nome_variavel: str = Field(description="Nome da coluna avaliada")
    tipo_variavel: str = Field(description="Papel da variável (numerica, categorica, alvo, predicao)")
    metodo_estatistico: str = Field(description="Teste aplicado (kolmogorov_smirnov, psi, wasserstein)")
    estatistica_teste: float = Field(description="Valor numérico da estatística calculada")
    p_valor: float | None = Field(default=None, description="P-valor do teste (quando aplicável)")
    limiar_alerta: float = Field(description="Limiar configurado no YAML para acionamento de alerta")
    drift_detectado: bool = Field(description="Verdadeiro se o desvio ultrapassou o limiar de significância")
    distribuicao_referencia: dict[str, float] = Field(
        default={}, description="Métricas de resumo da distribuição de referência (média, desvio, mediana)"
    )
    distribuicao_producao: dict[str, float] = Field(
        default={}, description="Métricas de resumo da distribuição de produção (média, desvio, mediana)"
    )


class ResultadoRelatorioDrift(BaseModel):
    """Relatório consolidado de observabilidade contínua e detecção de drift."""

    timestamp_analise: str = Field(description="Data e hora da verificação em formato ISO UTC")
    total_amostras_referencia: int = Field(description="Total de amostras no baseline de treino")
    total_amostras_producao: int = Field(description="Total de amostras na janela de inferência")
    qualidade_dados: MetricasQualidadeDados = Field(description="Auditoria de qualidade dos dados")
    covariate_drift_detectado: bool = Field(description="Mudança estatística nas variáveis preditoras P(X)")
    prediction_drift_detectado: bool = Field(description="Mudança na distribuição das predições P(Y_hat)")
    target_drift_detectado: bool | None = Field(
        default=None, description="Mudança no alvo real P(Y) se rótulos estiverem disponíveis"
    )
    concept_drift_detectado: bool | None = Field(
        default=None, description="Degradação na relação funcional P(Y|X) se rótulos estiverem disponíveis"
    )
    classificacao_drift: str = Field(
        description="Diagnóstico taxonômico (sem_drift, virtual_drift, real_concept_drift, sudden_drift, gradual_drift)"
    )
    degradacao_rmse_percentual: float | None = Field(
        default=None, description="Variação percentual do erro RMSE em relação ao baseline"
    )
    variaveis_com_drift: list[str] = Field(
        default=[], description="Lista de features que apresentaram drift estatístico"
    )
    detalhes_variaveis: list[ResultadoDriftVariavel] = Field(
        default=[], description="Detalhamento individual por variável"
    )
    recomendacao_acao: str = Field(description="Orientação operacional prescritiva de engenharia")


@runtime_checkable
class DetectorDrift(Protocol):
    """Protocolo GoF Strategy para detecção estatística de desvios e degradação."""

    def analisar_drift(
        self,
        dados_referencia: pd.DataFrame,
        dados_producao: pd.DataFrame,
        y_referencia: pd.Series | None = None,
        y_producao_real: pd.Series | None = None,
        predicoes_referencia: pd.Series | None = None,
        predicoes_producao: pd.Series | None = None,
        rmse_baseline: float | None = None,
    ) -> ResultadoRelatorioDrift:
        """Executa a bateria de testes estatísticos entre a referência e a produção."""
        ...
