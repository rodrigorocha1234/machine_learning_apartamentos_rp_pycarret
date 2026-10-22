from __future__ import annotations
from typing import final
from app.contratos_dados.gerador_eda import ErroGeradorEda, GeradorEda
from app.contratos_dados.motor_tabela import MotorTabela
from app.motores_tabela.fabrica_motores import FabricaMotores
from app.analises_eda.gerador_sweetviz import GeradorSweetviz
from app.analises_eda.gerador_plotly import GeradorPlotly
from app.analises_eda.gerador_nativo import GeradorNativo
from app.analises_eda.gerador_ydata import GeradorYData
from app.analises_eda.gerador_altair import GeradorAltair
from app.analises_eda.gerador_bokeh import GeradorBokeh


@final
class FabricaEda:
    """Fábrica GoF para instanciação desacoplada de motores de Análise Exploratória (EDA)."""

    _MOTORES: dict[str, type[GeradorEda]] = {
        "sweetviz": GeradorSweetviz,
        "plotly": GeradorPlotly,
        "nativo": GeradorNativo,
        "json": GeradorNativo,
        "ydata": GeradorYData,
        "ydata_profiling": GeradorYData,
        "pandas_profiling": GeradorYData,
        "altair": GeradorAltair,
        "vega": GeradorAltair,
        "bokeh": GeradorBokeh,
    }

    @staticmethod
    def criar_gerador(
        motor: str, motor_tabular: str | MotorTabela | None = None
    ) -> GeradorEda:
        """Instancia o motor de EDA correspondente à chave informada no arquivo YAML com injeção do motor tabular.

        Args:
            motor: Identificador do motor visual ('plotly', 'altair', 'bokeh', 'nativo', etc.).
            motor_tabular: Motor tabular para expressões AST ('pandas', 'polars' ou instância de MotorTabela).

        Returns:
            Instância que implementa o protocolo GeradorEda.

        Raises:
            ErroGeradorEda: Se o motor especificado não for suportado.
        """
        chave_normalizada = motor.strip().lower()
        classe_gerador = FabricaEda._MOTORES.get(chave_normalizada)
        if classe_gerador is None:
            opcoes_validas = list(FabricaEda._MOTORES.keys())
            raise ErroGeradorEda(
                f"Motor de EDA '{motor}' não suportado. "
                f"Opções disponíveis no projeto: {opcoes_validas}"
            )

        motor_instancia: MotorTabela | None = None
        if isinstance(motor_tabular, str) and motor_tabular.strip():
            motor_instancia = FabricaMotores.criar_motor(motor_tabular)
        elif motor_tabular is not None:
            motor_instancia = motor_tabular

        if classe_gerador in (GeradorPlotly, GeradorAltair, GeradorBokeh, GeradorNativo):
            return classe_gerador(motor_tabela=motor_instancia)
        return classe_gerador()
