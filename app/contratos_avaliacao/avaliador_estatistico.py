"""Contratos e modelos para testes estatísticos pareados (Friedman e Nemenyi)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field


class ResultadoFriedman(BaseModel):
    """Resultados do teste não-paramétrico global de Friedman."""

    estatistica_chi2: float = Field(ge=0.0, description="Estatística qui-quadrado de Friedman")
    p_valor: float = Field(ge=0.0, le=1.0, description="Valor-p assintótico do teste")
    graus_liberdade: int = Field(ge=1, description="Graus de liberdade (k - 1)")
    diferenca_significativa: bool = Field(
        description="Indica se há diferença estatisticamente significativa ao nível alpha"
    )
    hipotese_nula_rejeitada: bool = Field(
        description="Verdadeiro se rejeitamos H0 de equivalência entre modelos"
    )
    ranks_medios: dict[str, float] = Field(
        description="Ranking médio de cada modelo nas dobras pareadas (menor é melhor)"
    )


class ComparacaoPar(BaseModel):
    """Resultado do teste pareado post-hoc de Nemenyi entre dois modelos."""

    modelo_a: str = Field(description="Primeiro modelo da comparação")
    modelo_b: str = Field(description="Segundo modelo da comparação")
    diferenca_rank: float = Field(ge=0.0, description="Diferença absoluta entre os ranks médios")
    estatistica_q: float = Field(ge=0.0, description="Estatística q de Studentized Range")
    p_valor: float = Field(ge=0.0, le=1.0, description="Valor-p ajustado para comparação múltipla")
    significativo: bool = Field(
        description="Indica se a diferença entre o par excede a Diferença Crítica (CD)"
    )


class ResultadoNemenyi(BaseModel):
    """Resultados do pós-teste de Nemenyi com Diferença Crítica e comparações de pares."""

    diferenca_critica: float = Field(
        ge=0.0, description="Distância mínima entre ranks para significância ao nível alpha (CD)"
    )
    comparacoes: list[ComparacaoPar] = Field(
        description="Lista de comparações pareadas entre todos os modelos"
    )
    modelos_equivalentes_ao_campeao: list[str] = Field(
        description="Modelos cuja diferença para o campeão não ultrapassa a Diferença Crítica"
    )


class ResultadoComparacaoEstatistica(BaseModel):
    """Evidência consolidada dos testes estatísticos e seleção fundamentada."""

    metrica_avaliada: str = Field(description="Métrica base da comparação pareada (ex: RMSE)")
    total_amostras_folds: int = Field(ge=2, description="Número de medições pareadas por modelo")
    friedman: ResultadoFriedman = Field(description="Resultado do teste de Friedman")
    nemenyi: ResultadoNemenyi | None = Field(
        default=None, description="Resultado do pós-teste de Nemenyi se Friedman rejeitou H0"
    )
    modelo_campeao_estatistico: str = Field(
        description="Modelo com o melhor rank médio geral"
    )
    candidatos_selecionados: list[str] = Field(
        description="Candidatos elegíveis para comitê ou promoção final"
    )
    justificativa_decisao: str = Field(
        description="Justificativa formal baseada em evidência empírica e limites estatísticos"
    )


@runtime_checkable
class AvaliadorEstatistico(Protocol):
    """Protocolo GoF Strategy para execução dos testes estatísticos pareados."""

    def comparar_modelos(
        self,
        matriz_cv: dict[str, list[float]],
        metrica: str = "RMSE",
        alpha: float = 0.05,
    ) -> ResultadoComparacaoEstatistica:
        """Executa os testes de Friedman e Nemenyi sobre as medições pareadas de CV."""
        ...
