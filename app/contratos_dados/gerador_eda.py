from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import pandas as pd

type ValorConfiguracaoEda = (
    str
    | int
    | float
    | bool
    | None
    | list[str]
    | list[int]
    | list[float]
    | dict[str, str | int | float | bool | None]
)


class ErroGeradorEda(Exception):
    """Exceção levantada em operações de geração de análise exploratória (EDA)."""
    pass


@dataclass(frozen=True)
class ConfiguracaoAnaliseGrafica:
    """Configuração imutável de uma análise, gráfico ou tabela cadastrado declarativamente no YAML."""
    nome: str
    tipo_grafico: str
    ativo: bool = True
    titulo: str = ""
    coluna: str | None = None
    coluna_x: str | None = None
    coluna_y: str | None = None
    cor: str | None = None
    nbins: int = 30
    marginal: str | None = None
    escala: str = "linear"
    agregacao: str = "mediana"
    linha_tendencia: str | None = None
    expressao_filtro: str | None = None
    expressao_calculo: str | None = None
    coluna_destino: str | None = None
    agrupar_por: str | None = None
    colunas: list[str] | None = None
    metricas: list[str] | None = None
    ordenar_por: str | None = None
    direcao_ordenacao: str = "asc"
    limite: int | None = None
    motor_tabular: str | None = None

    @staticmethod
    def de_dicionario(dados: dict[str, ValorConfiguracaoEda]) -> ConfiguracaoAnaliseGrafica:
        """Converte um dicionário extraído do YAML em uma instância tipada e validada."""
        nome = str(dados.get("nome", "analise_sem_nome"))
        tipo = str(dados.get("tipo_grafico", dados.get("tipo", "histograma"))).strip().lower()
        ativo = bool(dados.get("ativo", True))
        titulo = str(dados.get("titulo", nome.replace("_", " ").title()))
        
        coluna = dados.get("coluna")
        coluna_x = dados.get("coluna_x", dados.get("x", coluna))
        coluna_y = dados.get("coluna_y", dados.get("y"))
        cor = dados.get("cor")
        marginal = dados.get("marginal")
        escala = str(dados.get("escala", "linear")).lower()
        agregacao = str(dados.get("agregacao", "mediana")).lower()
        linha_tendencia = dados.get("linha_tendencia")

        expressao_filtro = dados.get("expressao_filtro", dados.get("filtro"))
        expressao_calculo = dados.get("expressao_calculo", dados.get("expressao"))
        coluna_destino = dados.get("coluna_destino", dados.get("destino"))
        agrupar_por = dados.get("agrupar_por", dados.get("grupo"))
        ordenar_por = dados.get("ordenar_por")
        direcao_ordenacao = str(dados.get("direcao_ordenacao", "asc")).lower()
        motor_tabular = dados.get("motor_tabular")

        nbins_raw = dados.get("nbins", 30)
        nbins = int(nbins_raw) if isinstance(nbins_raw, (int, float, str)) and str(nbins_raw).isdigit() else 30

        limite_raw = dados.get("limite")
        limite = int(limite_raw) if isinstance(limite_raw, (int, float, str)) and str(limite_raw).isdigit() else None

        colunas_raw = dados.get("colunas")
        colunas: list[str] | None = None
        if isinstance(colunas_raw, list):
            colunas = [str(c) for c in colunas_raw]

        metricas_raw = dados.get("metricas")
        metricas: list[str] | None = None
        if isinstance(metricas_raw, list):
            metricas = [str(m) for m in metricas_raw]

        return ConfiguracaoAnaliseGrafica(
            nome=nome,
            tipo_grafico=tipo,
            ativo=ativo,
            titulo=titulo,
            coluna=str(coluna) if coluna is not None else None,
            coluna_x=str(coluna_x) if coluna_x is not None else None,
            coluna_y=str(coluna_y) if coluna_y is not None else None,
            cor=str(cor) if cor is not None else None,
            nbins=nbins,
            marginal=str(marginal) if marginal is not None else None,
            escala=escala,
            agregacao=agregacao,
            linha_tendencia=str(linha_tendencia) if linha_tendencia is not None else None,
            expressao_filtro=str(expressao_filtro) if expressao_filtro is not None else None,
            expressao_calculo=str(expressao_calculo) if expressao_calculo is not None else None,
            coluna_destino=str(coluna_destino) if coluna_destino is not None else None,
            agrupar_por=str(agrupar_por) if agrupar_por is not None else None,
            colunas=colunas,
            metricas=metricas,
            ordenar_por=str(ordenar_por) if ordenar_por is not None else None,
            direcao_ordenacao=direcao_ordenacao,
            limite=limite,
            motor_tabular=str(motor_tabular).strip().lower() if motor_tabular is not None else None,
        )


@runtime_checkable
class GeradorEda(Protocol):
    """Contrato abstrato (GoF Strategy) para motores de geração de relatórios de EDA."""

    def gerar_relatorio(
        self,
        dados: pd.DataFrame,
        coluna_alvo: str,
        caminho_saida: str,
        analises_configuradas: list[dict[str, ValorConfiguracaoEda]] | None = None,
    ) -> str:
        """Gera o relatório visual de EDA e retorna o caminho do arquivo produzido.

        Args:
            dados: DataFrame com os dados a serem analisados.
            coluna_alvo: Nome da variável dependente (alvo da análise).
            caminho_saida: Caminho do arquivo de saída (ex: .html ou .json).
            analises_configuradas: Lista opcional de análises gráficas cadastradas no YAML.

        Returns:
            Caminho absoluto ou relativo do artefato gerado.

        Raises:
            ErroGeradorEda: Em caso de falhas durante o processamento ou exportação.
        """
        ...
