from __future__ import annotations
import json
from typing import final
import pandas as pd
from app.contratos_dados.repositorio_dados import RepositorioDados

type ValorRelatorioEda = (
    int
    | float
    | str
    | None
    | dict[str, int]
    | dict[str, float]
    | dict[str, dict[str, float | str | int | None]]
)

class ErroAnaliseEda(Exception):
    """Exceção levantada em operações de análise exploratória de dados."""
    pass

@final
class AnaliseExploratoria:
    """Gerador de análises exploratórias (EDA) a partir de repositório de staging SQLite."""

    def __init__(self, repositorio: RepositorioDados) -> None:
        self._repositorio = repositorio

    def executar_eda(self, coluna_alvo: str) -> dict[str, ValorRelatorioEda]:
        try:
            dados = self._repositorio.ler_dados()
        except Exception as erro:
            raise ErroAnaliseEda(f"Falha ao ler dados do staging para EDA: {erro}") from erro

        if dados.empty:
            raise ErroAnaliseEda("Base de dados no staging está vazia para realizar EDA.")

        resumo_estatistico = dados.describe(include="all").to_dict()
        nulos = dados.isna().sum().to_dict()
        
        colunas_numericas = dados.select_dtypes(include=["number"]).columns.tolist()
        correlacoes: dict[str, dict[str, float]] = {}
        if len(colunas_numericas) > 1:
            correlacoes = dados[colunas_numericas].corr().round(4).to_dict()

        distribuicao_alvo: dict[str, float] = {}
        if coluna_alvo in dados.columns and pd.api.types.is_numeric_dtype(dados[coluna_alvo]):
            serie_alvo = dados[coluna_alvo].dropna()
            distribuicao_alvo = {
                "media": float(serie_alvo.mean()),
                "mediana": float(serie_alvo.median()),
                "desvio_padrao": float(serie_alvo.std()),
                "minimo": float(serie_alvo.min()),
                "maximo": float(serie_alvo.max()),
                "p25": float(serie_alvo.quantile(0.25)),
                "p75": float(serie_alvo.quantile(0.75)),
            }

        return {
            "total_linhas": int(len(dados)),
            "total_colunas": int(len(dados.columns)),
            "resumo": resumo_estatistico,
            "nulos": nulos,
            "distribuicao_alvo": distribuicao_alvo,
            "correlacoes": correlacoes,
        }

    def calcular_mediana_agrupada(self, coluna_alvo: str, agrupador: str) -> dict[str, float]:
        dados = self._repositorio.ler_dados()
        if coluna_alvo not in dados.columns or agrupador not in dados.columns:
            raise ErroAnaliseEda(f"Colunas '{coluna_alvo}' ou '{agrupador}' inexistentes para agregação.")
        agrupado = dados.groupby(agrupador, observed=True)[coluna_alvo].median().to_dict()
        return {str(k): float(v) for k, v in agrupado.items()}

    def salvar_relatorio_json(self, relatorio: dict[str, ValorRelatorioEda], caminho_destino: str) -> None:
        try:
            with open(caminho_destino, "w", encoding="utf-8") as f:
                json.dump(relatorio, f, ensure_ascii=False, indent=2)
        except Exception as erro:
            raise ErroAnaliseEda(f"Falha ao salvar relatório EDA em '{caminho_destino}': {erro}") from erro

    def gerar_relatorio_visual(
        self,
        coluna_alvo: str,
        motor: str = "sweetviz",
        caminho_saida: str = "storage-data/relatorio_eda.html",
        analises_configuradas: list[dict[str, ValorConfiguracaoEda]] | None = None,
        motor_tabular: str | None = None,
    ) -> str:
        """Gera o relatório visual de EDA utilizando o motor configurado via FabricaEda."""
        from app.analises_eda.fabrica_eda import FabricaEda
        try:
            dados = self._repositorio.ler_dados()
            gerador = FabricaEda.criar_gerador(motor=motor, motor_tabular=motor_tabular)
            return gerador.gerar_relatorio(
                dados=dados,
                coluna_alvo=coluna_alvo,
                caminho_saida=caminho_saida,
                analises_configuradas=analises_configuradas,
            )
        except Exception as erro:
            raise ErroAnaliseEda(f"Falha ao gerar relatório visual EDA com motor '{motor}': {erro}") from erro

