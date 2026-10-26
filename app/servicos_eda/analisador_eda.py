"""Serviço orquestrador para a Etapa 3: Análise Exploratória de Dados (EDA)."""

from __future__ import annotations

import time
from typing import final
import pandas as pd
from pydantic import BaseModel, Field

from app.adaptadores_visualizacao.fabrica_visualizadores import FabricaVisualizadores
from app.conf_yaml import ConfYAML
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.contratos_visualizacao.adaptador_visualizacao import (
    AdaptadorVisualizacao,
    ResultadoAnaliseEda,
    ResultadoVisualizacao,
)
from app.esquema_configuracao import ConfiguracaoEda
from app.repositorios_dados.fabrica_repositorios import FabricaRepositorios
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_eda.calculador_estatistico import CalculadorEstatistico


class ErroAnalisadorEda(Exception):
    """Exceção levantada em falhas durante a orquestração da EDA."""


class ResultadoEda(BaseModel):
    """Sumário executivo do processamento da etapa de EDA."""

    total_analises_configuradas: int = Field(ge=0, description="Total de itens cadastrados no YAML")
    total_analises_processadas: int = Field(ge=0, description="Total de análises ativas executadas")
    nomes_analises_processadas: list[str] = Field(description="Identificadores das análises concluídas")
    motor_visualizacao: str = Field(description="Motor gráfico utilizado")
    arquivos_gerados: list[str] = Field(description="Artefatos e relatórios produzidos")
    tempo_execucao_segundos: float = Field(ge=0.0, description="Duração total da execução em segundos")


@final
class AnalisadorEda:
    """Orquestrador desacoplado responsável por coordenar cálculos estatísticos e renderização."""

    def __init__(
        self,
        adaptador: AdaptadorVisualizacao,
        calculador: CalculadorEstatistico,
        configuracao_eda: ConfiguracaoEda,
        repositorio_staging: RepositorioDados | None = None,
    ) -> None:
        self._adaptador = adaptador
        self._calculador = calculador
        self._configuracao = configuracao_eda
        self._repositorio = repositorio_staging

    @classmethod
    def da_configuracao(
        cls,
        conf: ConfYAML,
        adaptador_sobrescrito: AdaptadorVisualizacao | None = None,
    ) -> AnalisadorEda:
        """Instancia o orquestrador a partir das definições do ConfYAML."""
        config_eda = conf.obter_eda()
        adaptador = (
            adaptador_sobrescrito
            if adaptador_sobrescrito is not None
            else FabricaVisualizadores.criar_visualizador(config_eda)
        )
        calculador = CalculadorEstatistico()
        repositorio = FabricaRepositorios.criar_repositorio(conf.obter_staging())

        return cls(
            adaptador=adaptador,
            calculador=calculador,
            configuracao_eda=config_eda,
            repositorio_staging=repositorio,
        )

    def executar_eda(
        self, dados: pd.DataFrame | None = None
    ) -> tuple[list[ResultadoAnaliseEda], ResultadoVisualizacao, ResultadoEda]:
        """Executa todas as análises EDA ativas e envia ao adaptador de visualização."""
        inicio = time.perf_counter()

        # 1. Carregamento dos dados tabulares
        df_base = self._obter_dados_entrada(dados)
        if df_base.empty:
            raise ErroAnalisadorEda("Conjunto de dados para análise EDA está vazio.")

        # 2. Execução das análises ativas
        analises_calculadas: list[ResultadoAnaliseEda] = []
        for item in self._configuracao.analises:
            if not item.ativo:
                continue
            try:
                res_item = self._calculador.calcular_analise(df_base, item)
                analises_calculadas.append(res_item)
            except Exception as erro_item:
                raise ErroAnalisadorEda(
                    f"Falha ao executar cálculo da análise '{item.nome}': {erro_item}"
                ) from erro_item

        # 3. Renderização/Publicação no motor gráfico
        try:
            res_vis = self._adaptador.renderizar_analises(
                analises_calculadas, self._configuracao
            )
        except Exception as erro_vis:
            raise ErroAnalisadorEda(
                f"Falha na renderização visual com motor '{self._adaptador.obter_nome_motor()}': {erro_vis}"
            ) from erro_vis

        duracao = time.perf_counter() - inicio

        resultado_eda = ResultadoEda(
            total_analises_configuradas=len(self._configuracao.analises),
            total_analises_processadas=len(analises_calculadas),
            nomes_analises_processadas=[a.nome for a in analises_calculadas],
            motor_visualizacao=self._adaptador.obter_nome_motor(),
            arquivos_gerados=res_vis.arquivos_gerados,
            tempo_execucao_segundos=round(duracao, 4),
        )

        return analises_calculadas, res_vis, resultado_eda

    def _obter_dados_entrada(self, dados: pd.DataFrame | None) -> pd.DataFrame:
        if dados is not None:
            return dados

        if self._repositorio is None:
            raise ErroAnalisadorEda(
                "Nenhum repositório de staging configurado e nenhum DataFrame fornecido."
            )

        # Se for repositório SQLite, verifica se há a tabela 'imoveis_limpos' preparada na Etapa 2
        if isinstance(self._repositorio, RepositorioSqlite):
            repo_limpos = RepositorioSqlite(
                caminho=self._repositorio.caminho,
                tabela="imoveis_limpos",
            )
            if repo_limpos.verificar_existencia():
                dados_limpos = repo_limpos.ler_dados()
                if not dados_limpos.empty:
                    return dados_limpos

        return self._repositorio.ler_dados()
