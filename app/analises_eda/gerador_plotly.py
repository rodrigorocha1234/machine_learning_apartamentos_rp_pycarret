from __future__ import annotations
import os
from typing import final, override
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.analises_eda.construtor_layout import ConstrutorLayout
from app.analises_eda.construtor_tabela import ConstrutorTabela
from app.contratos_dados.gerador_eda import (
    ConfiguracaoAnaliseGrafica,
    ErroGeradorEda,
    GeradorEda,
    ValorConfiguracaoEda,
)
from app.contratos_dados.motor_tabela import MotorTabela
from app.motores_tabela.fabrica_motores import FabricaMotores
from app.motores_tabela.motor_pandas import MotorPandas


@final
class GeradorPlotly(GeradorEda):
    """Implementação de GeradorEda gerando gráficos interativos via Plotly baseados em análises cadastradas no YAML."""

    def __init__(self, motor_tabela: MotorTabela | None = None) -> None:
        self._construtor_tabela = ConstrutorTabela()
        self._construtor_layout = ConstrutorLayout()
        self._motor_tabela: MotorTabela = motor_tabela or MotorPandas()

    @override
    def gerar_relatorio(
        self,
        dados: pd.DataFrame,
        coluna_alvo: str,
        caminho_saida: str,
        analises_configuradas: list[dict[str, ValorConfiguracaoEda]] | None = None,
    ) -> str:
        if dados.empty:
            raise ErroGeradorEda("O DataFrame fornecido para geração do EDA via Plotly está vazio.")

        try:
            diretorio_destino = os.path.dirname(caminho_saida)
            if diretorio_destino and not os.path.exists(diretorio_destino):
                os.makedirs(diretorio_destino, exist_ok=True)

            secoes_html: list[str] = []
            colunas_num = dados.select_dtypes(include=["number"]).columns.tolist()

            # Processa as análises declaradas no YAML se fornecidas
            configs: list[ConfiguracaoAnaliseGrafica] = []
            if analises_configuradas:
                for item in analises_configuradas:
                    cfg = ConfiguracaoAnaliseGrafica.de_dicionario(item)
                    if cfg.ativo:
                        configs.append(cfg)

            # Caso não haja análises customizadas cadastradas, usa catálogo padrão
            if not configs:
                configs = [
                    ConfiguracaoAnaliseGrafica(
                        nome="distribuicao_alvo",
                        tipo_grafico="histograma",
                        coluna=coluna_alvo,
                        titulo=f"Distribuição do Alvo: {coluna_alvo}",
                        marginal="box",
                    ),
                    ConfiguracaoAnaliseGrafica(
                        nome="matriz_correlacao",
                        tipo_grafico="correlacao",
                        titulo="Matriz de Correlação das Variáveis Numéricas",
                    ),
                ]
                outras_num = [c for c in colunas_num if c != coluna_alvo]
                if outras_num:
                    configs.append(
                        ConfiguracaoAnaliseGrafica(
                            nome="dispersao_metragem",
                            tipo_grafico="dispersao",
                            coluna_x=outras_num[0],
                            coluna_y=coluna_alvo,
                            titulo=f"Relação: {outras_num[0]} vs {coluna_alvo}",
                        )
                    )

            # Renderiza cada análise, gráfico ou tabela cadastrado no YAML
            primeiro_grafico = True

            for cfg in configs:
                fig = None
                df_item = dados.copy()
                motor_item = (
                    FabricaMotores.criar_motor(cfg.motor_tabular)
                    if cfg.motor_tabular
                    else self._motor_tabela
                )

                # 1. Aplica expressão de cálculo de novo atributo via AST seguro se configurada
                if cfg.expressao_calculo and cfg.coluna_destino:
                    try:
                        df_item = motor_item.transformar_dados(
                            df_item, cfg.expressao_calculo, cfg.coluna_destino
                        )
                    except Exception as erro_calc:
                        raise ErroGeradorEda(
                            f"Falha ao avaliar expressão de cálculo '{cfg.expressao_calculo}' na análise '{cfg.nome}': {erro_calc}"
                        ) from erro_calc

                # 2. Aplica expressão de filtro via AST seguro se configurada
                if cfg.expressao_filtro:
                    try:
                        df_item = motor_item.filtrar_dados(df_item, cfg.expressao_filtro)
                    except Exception as erro_filtro:
                        raise ErroGeradorEda(
                            f"Falha ao avaliar expressão de filtro '{cfg.expressao_filtro}' na análise '{cfg.nome}': {erro_filtro}"
                        ) from erro_filtro

                if df_item.empty:
                    continue

                colunas_num_item = df_item.select_dtypes(include=["number"]).columns.tolist()

                if cfg.tipo_grafico in ("histograma", "histogram"):
                    col = cfg.coluna or cfg.coluna_x or coluna_alvo
                    if col in df_item.columns:
                        log_x = cfg.escala == "log"
                        fig = px.histogram(
                            df_item,
                            x=col,
                            nbins=cfg.nbins,
                            marginal=cfg.marginal if cfg.marginal in ("box", "violin", "rug") else None,
                            log_x=log_x,
                            title=cfg.titulo,
                            template="plotly_dark",
                            color_discrete_sequence=["#38bdf8"],
                        )

                elif cfg.tipo_grafico in ("boxplot", "box"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if y_col in df_item.columns:
                        fig = px.box(
                            df_item,
                            x=x_col if x_col in df_item.columns else None,
                            y=y_col,
                            title=cfg.titulo,
                            template="plotly_dark",
                            color_discrete_sequence=["#818cf8"],
                        )

                elif cfg.tipo_grafico in ("dispersao", "scatter"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if x_col in df_item.columns and y_col in df_item.columns:
                        cor_col = cfg.cor if cfg.cor in df_item.columns else None
                        has_trend = cfg.linha_tendencia == "ols" and len(df_item) >= 5
                        fig = px.scatter(
                            df_item,
                            x=x_col,
                            y=y_col,
                            color=cor_col,
                            trendline="ols" if has_trend else None,
                            title=cfg.titulo,
                            template="plotly_dark",
                            color_discrete_sequence=["#34d399", "#38bdf8", "#fbbf24"],
                        )

                elif cfg.tipo_grafico in ("correlacao", "heatmap"):
                    if len(colunas_num_item) > 1:
                        matriz_corr = df_item[colunas_num_item].corr().round(3)
                        fig = px.imshow(
                            matriz_corr,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Blues",
                            title=cfg.titulo,
                            template="plotly_dark",
                        )

                elif cfg.tipo_grafico in ("barras", "bar"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if x_col in df_item.columns:
                        if y_col in df_item.columns and pd.api.types.is_numeric_dtype(df_item[y_col]):
                            if cfg.agregacao == "media":
                                df_agg = df_item.groupby(x_col, observed=True)[y_col].mean().reset_index()
                            elif cfg.agregacao == "contagem":
                                df_agg = df_item.groupby(x_col, observed=True)[y_col].count().reset_index()
                            else:
                                df_agg = df_item.groupby(x_col, observed=True)[y_col].median().reset_index()
                            fig = px.bar(
                                df_agg,
                                x=x_col,
                                y=y_col,
                                title=cfg.titulo,
                                template="plotly_dark",
                                color_discrete_sequence=["#fbbf24"],
                            )
                        else:
                            df_cont = df_item[x_col].value_counts().reset_index()
                            df_cont.columns = [x_col, "Contagem"]
                            fig = px.bar(
                                df_cont,
                                x=x_col,
                                y="Contagem",
                                title=cfg.titulo,
                                template="plotly_dark",
                                color_discrete_sequence=["#fbbf24"],
                            )

                elif cfg.tipo_grafico in ("tabela", "table"):
                    fig = self._construtor_tabela.construir_tabela(
                        dados=df_item,
                        configuracao=cfg,
                        coluna_alvo=coluna_alvo,
                    )

                if fig is not None:
                    # Inclui a biblioteca PlotlyJS apenas na primeira figura para otimizar o tamanho do HTML
                    include_js = "cdn" if primeiro_grafico else False
                    primeiro_grafico = False
                    secoes_html.append(fig.to_html(full_html=False, include_plotlyjs=include_js))

            documento_html = self._construtor_layout.construir_documento(
                secoes_graficos=secoes_html,
                total_amostras=len(dados),
                coluna_alvo=coluna_alvo,
                total_analises=len(configs),
                titulo="Relatório de Análise Exploratória (Plotly)",
            )

            with open(caminho_saida, "w", encoding="utf-8") as f:
                f.write(documento_html)

            return caminho_saida
        except Exception as erro:
            raise ErroGeradorEda(f"Falha ao gerar relatório Plotly em '{caminho_saida}': {erro}") from erro
