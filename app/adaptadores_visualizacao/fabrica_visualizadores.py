"""Fábrica GoF Factory Method para adaptadores de visualização gráfica."""

from __future__ import annotations

from typing import final
from app.adaptadores_visualizacao.adaptador_grafana import AdaptadorGrafana
from app.adaptadores_visualizacao.adaptador_relatorios import AdaptadorRelatorios
from app.contratos_visualizacao.adaptador_visualizacao import AdaptadorVisualizacao
from app.esquema_configuracao import ConfiguracaoEda


class ErroFabricaVisualizadores(Exception):
    """Exceção levantada quando o motor gráfico solicitado não é suportado."""


@final
class FabricaVisualizadores:
    """Fábrica responsável pela instanciação desacoplada de motores de visualização."""

    @classmethod
    def criar_visualizador(cls, config: ConfiguracaoEda) -> AdaptadorVisualizacao:
        """Cria e retorna a implementação de AdaptadorVisualizacao correspondente."""
        motor = config.motor_grafico.strip().lower()

        if motor == "grafana":
            return AdaptadorGrafana()

        if motor in {"relatorios", "seaborn", "plotly", "altair"}:
            return AdaptadorRelatorios()

        raise ErroFabricaVisualizadores(
            f"Motor gráfico não suportado pela fábrica: '{config.motor_grafico}'."
        )
