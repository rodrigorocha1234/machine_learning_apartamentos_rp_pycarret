"""Contrato e modelos para motores de visualização gráfica desacoplados."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field

from app.esquema_configuracao import ConfiguracaoEda


class ResultadoAnaliseEda(BaseModel):
    """Representação estruturada do cálculo estatístico de uma análise EDA."""

    nome: str = Field(min_length=1, description="Identificador da análise")
    titulo: str = Field(description="Título descritivo do gráfico ou painel")
    tipo_grafico: str = Field(description="Tipo de gráfico (histograma, boxplot, dispersao, etc.)")
    descricao: str = Field(default="", description="Descrição analítica do objetivo")
    coluna_principal: str | None = Field(default=None, description="Coluna focal da análise")
    dados_estruturados: dict[str, object] = Field(
        default={}, description="Dados tabulares ou séries numéricas sumarizadas"
    )
    metricas_resumo: dict[str, float] = Field(
        default={}, description="Métricas estatísticas escalares (média, mediana, IQR, etc.)"
    )


class ResultadoVisualizacao(BaseModel):
    """Metadados e evidências da renderização ou publicação dos artefatos visuais."""

    motor_utilizado: str = Field(description="Nome do motor gráfico executor")
    total_paineis: int = Field(ge=0, description="Quantidade de painéis ou figuras geradas")
    arquivos_gerados: list[str] = Field(
        default=[], description="Caminhos absolutos ou relativos dos artefatos produzidos"
    )
    sucesso: bool = Field(default=True, description="Indicador de sucesso da renderização")
    mensagem: str = Field(default="", description="Mensagem de status ou diagnóstico")


@runtime_checkable
class AdaptadorVisualizacao(Protocol):
    """Protocolo GoF Strategy para renderizadores e publicadores de gráficos e dashboards."""

    def renderizar_analises(
        self, analises: list[ResultadoAnaliseEda], configuracao: ConfiguracaoEda
    ) -> ResultadoVisualizacao:
        """Renderiza ou publica as análises no destino configurado (Grafana, HTML, etc.)."""
        ...

    def obter_nome_motor(self) -> str:
        """Retorna o identificador textual do motor gráfico."""
        ...
