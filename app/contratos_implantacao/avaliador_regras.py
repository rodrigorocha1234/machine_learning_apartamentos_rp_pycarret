"""Contratos estruturais e modelos de dados para avaliação de regras de negócio e inferência."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import pandas as pd
from pydantic import BaseModel, Field


class RequisicaoImovel(BaseModel):
    """Esquema de entrada para requisição de avaliação e inferência de imóvel."""

    Metragem: float = Field(gt=0.0, description="Área total ou útil do apartamento em m²")
    Quartos: int = Field(ge=0, description="Quantidade de dormitórios")
    Banheiros: int = Field(ge=0, description="Quantidade de banheiros")
    Vagas: int = Field(ge=0, description="Quantidade de vagas de garagem")
    Bairro: str = Field(min_length=1, description="Bairro de localização em Ribeirão Preto")
    Zona: str = Field(default="Sul", description="Zona da cidade (Sul, Leste, Oeste, Norte, Centro)")
    Valor_Condominio: float = Field(default=0.0, ge=0.0, description="Valor mensal da taxa condominial em R$")
    Valor_Iptu: float = Field(default=0.0, ge=0.0, description="Valor anual ou mensal do IPTU em R$")
    Area_Privativa: float = Field(default=0.0, ge=0.0, description="Área privativa útil em m² (0 se não informado)")


class RespostaRegra(BaseModel):
    """Detalhamento individual de uma regra de negócio calculada."""

    nome: str = Field(description="Identificador da regra de negócio")
    destino: str = Field(description="Nome da variável gerada no payload de saída")
    descricao: str = Field(description="Explicação da lógica de negócio")
    valor: float = Field(description="Resultado numérico do cálculo")


class RespostaPredicao(BaseModel):
    """Esquema tipado retornado pelo endpoint de inferência da API."""

    preco_predito: float = Field(description="Preço de venda estimado pelo modelo campeão em R$")
    regras_calculadas: dict[str, float] = Field(
        description="Mapeamento de campo destino para valor calculado das regras ativas"
    )
    detalhes_regras: list[RespostaRegra] = Field(
        description="Lista descritiva com detalhes de cada regra de negócio processada"
    )
    modelo_campeao: str = Field(description="Nome do algoritmo selecionado para predição")
    versao_modelo: str = Field(description="Identificador de versão do modelo em produção")
    data_hora_iso: str = Field(description="Timestamp ISO da predição")


class StatusServico(BaseModel):
    """Esquema de status operacional retornado pela rota de health check."""

    status: str = Field(description="Estado de prontidão do serviço ('operacional')")
    versao_api: str = Field(description="Versão semântica da API de inferência")
    modelo_campeao: str = Field(description="Nome do modelo atualmente carregado")
    versao_modelo: str = Field(description="Versão do modelo em produção")
    total_regras_ativas: int = Field(description="Quantidade de regras de negócio em vigor")
    ambiente: str = Field(description="Ambiente de execução")


@runtime_checkable
class AvaliadorRegras(Protocol):
    """Protocolo GoF Strategy para cálculo seguro de regras de negócio dinâmicas."""

    def avaliar_regras_registro(
        self,
        contexto: dict[str, float | int | str],
    ) -> dict[str, float]:
        """Calcula as regras ativas sobre um registro individual."""
        ...

    def avaliar_regras_tabela(
        self,
        dados: pd.DataFrame,
        coluna_predita: str,
    ) -> pd.DataFrame:
        """Calcula e anexa as colunas de regras ativas a um DataFrame tabular."""
        ...
