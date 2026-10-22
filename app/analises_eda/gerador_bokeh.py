from __future__ import annotations
import os
from typing import final, override
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.themes import built_in_themes
import numpy as np
import pandas as pd
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
class GeradorBokeh(GeradorEda):
    """Implementação de GeradorEda gerando gráficos interativos via biblioteca Bokeh."""

    def __init__(self, motor_tabela: MotorTabela | None = None) -> None:
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
            raise ErroGeradorEda("O DataFrame fornecido para geração do EDA via Bokeh está vazio.")

        try:
            diretorio_destino = os.path.dirname(caminho_saida)
            if diretorio_destino and not os.path.exists(diretorio_destino):
                os.makedirs(diretorio_destino, exist_ok=True)

            configs: list[ConfiguracaoAnaliseGrafica] = []
            if analises_configuradas:
                for item in analises_configuradas:
                    cfg = ConfiguracaoAnaliseGrafica.de_dicionario(item)
                    if cfg.ativo:
                        configs.append(cfg)

            colunas_num = [str(c) for c in dados.select_dtypes(include=["number"]).columns]
            if not configs:
                configs = [
                    ConfiguracaoAnaliseGrafica(
                        nome="distribuicao_alvo",
                        tipo_grafico="histograma",
                        coluna=coluna_alvo,
                        titulo=f"Distribuição do Alvo: {coluna_alvo}",
                    )
                ]
                outras_num = [c for c in colunas_num if c != coluna_alvo]
                if outras_num:
                    configs.append(
                        ConfiguracaoAnaliseGrafica(
                            nome="dispersao_alvo",
                            tipo_grafico="dispersao",
                            coluna_x=outras_num[0],
                            coluna_y=coluna_alvo,
                            titulo=f"Relação: {outras_num[0]} vs {coluna_alvo}",
                        )
                    )

            plots: list[figure] = []

            for cfg in configs:
                df_item = dados.copy()
                motor_item = (
                    FabricaMotores.criar_motor(cfg.motor_tabular)
                    if cfg.motor_tabular
                    else self._motor_tabela
                )

                if cfg.expressao_calculo and cfg.coluna_destino:
                    try:
                        df_item = motor_item.transformar_dados(
                            df_item, cfg.expressao_calculo, cfg.coluna_destino
                        )
                    except Exception as erro_calc:
                        raise ErroGeradorEda(
                            f"Falha ao avaliar expressão de cálculo '{cfg.expressao_calculo}' na análise '{cfg.nome}': {erro_calc}"
                        ) from erro_calc

                if cfg.expressao_filtro:
                    try:
                        df_item = motor_item.filtrar_dados(df_item, cfg.expressao_filtro)
                    except Exception as erro_filtro:
                        raise ErroGeradorEda(
                            f"Falha ao avaliar expressão de filtro '{cfg.expressao_filtro}' na análise '{cfg.nome}': {erro_filtro}"
                        ) from erro_filtro

                if df_item.empty:
                    continue

                fig: figure | None = None

                if cfg.tipo_grafico in ("histograma", "histogram"):
                    col = cfg.coluna or cfg.coluna_x or coluna_alvo
                    if col in df_item.columns:
                        serie_col = df_item[col]
                        if isinstance(serie_col, pd.Series):
                            valores_brutos = pd.to_numeric(serie_col, errors="coerce").dropna().to_numpy()
                            if len(valores_brutos) > 0:
                                hist, edges = np.histogram(valores_brutos, bins=cfg.nbins)
                                fig = figure(
                                    title=cfg.titulo,
                                    width=850,
                                    height=300,
                                    tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                                )
                                fig.quad(
                                    top=hist,
                                    bottom=0,
                                    left=edges[:-1],
                                    right=edges[1:],
                                    fill_color="#38bdf8",
                                    line_color="#0284c7",
                                    alpha=0.85,
                                )
                                fig.xaxis.axis_label = col
                                fig.yaxis.axis_label = "Frequência"

                elif cfg.tipo_grafico in ("dispersao", "scatter"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if x_col and y_col and x_col in df_item.columns and y_col in df_item.columns:
                        serie_x = df_item[x_col]
                        serie_y = df_item[y_col]
                        if isinstance(serie_x, pd.Series) and isinstance(serie_y, pd.Series):
                            zona_lista = df_item["Zona"].astype(str).tolist() if "Zona" in df_item.columns else ["" for _ in range(len(df_item))]
                            bairro_lista = df_item["Bairro"].astype(str).tolist() if "Bairro" in df_item.columns else ["" for _ in range(len(df_item))]
                            fonte = ColumnDataSource(
                                data={
                                    "x": pd.to_numeric(serie_x, errors="coerce").fillna(0).tolist(),
                                    "y": pd.to_numeric(serie_y, errors="coerce").fillna(0).tolist(),
                                    "zona": zona_lista,
                                    "bairro": bairro_lista,
                                }
                            )
                            fig = figure(
                                title=cfg.titulo,
                                width=850,
                                height=340,
                                tools="pan,wheel_zoom,box_zoom,reset,save",
                            )
                            fig.scatter(
                                x="x",
                                y="y",
                                source=fonte,
                                size=7,
                                color="#818cf8",
                                alpha=0.7,
                            )
                            hover = HoverTool(
                                tooltips=[
                                    (x_col, "@x"),
                                    (y_col, "@y{0,0}"),
                                    ("Zona", "@zona"),
                                    ("Bairro", "@bairro"),
                                ]
                            )
                            fig.add_tools(hover)
                            fig.xaxis.axis_label = x_col
                            fig.yaxis.axis_label = y_col

                elif cfg.tipo_grafico in ("barras", "bar"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if x_col and x_col in df_item.columns:
                        if y_col and y_col in df_item.columns and cfg.agregacao != "contagem":
                            if cfg.agregacao == "media":
                                agrupado = df_item.groupby(x_col)[y_col].mean().reset_index()
                            else:
                                agrupado = df_item.groupby(x_col)[y_col].median().reset_index()
                        else:
                            agrupado = df_item.groupby(x_col).size().reset_index(name="contagem")
                            y_col = "contagem"

                        agrupado_ordenado = agrupado.sort_values(by=y_col, ascending=False).head(20)
                        col_cat = agrupado_ordenado[x_col]
                        categorias: list[str] = [str(c) for c in col_cat.tolist()] if isinstance(col_cat, pd.Series) else []
                        
                        col_val = agrupado_ordenado[y_col]
                        valores: list[float] = []
                        if isinstance(col_val, pd.Series):
                            valores = [float(v) for v in pd.to_numeric(col_val, errors="coerce").fillna(0).tolist()]

                        fig = figure(
                            x_range=FactorRange(factors=categorias),
                            title=cfg.titulo,
                            width=850,
                            height=320,
                            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                        )
                        fig.vbar(x=categorias, top=valores, width=0.55, color="#34d399", alpha=0.85)
                        fig.xaxis.major_label_orientation = 0.5
                        fig.yaxis.axis_label = f"{cfg.agregacao.title()} de {y_col}"

                elif cfg.tipo_grafico in ("correlacao", "heatmap"):
                    colunas_numericas = [str(c) for c in df_item.select_dtypes(include=["number"]).columns]
                    if len(colunas_numericas) >= 2:
                        matriz_corr = df_item[colunas_numericas].corr(numeric_only=True)
                        matriz_melt = matriz_corr.reset_index().melt(id_vars="index")
                        matriz_melt.columns = pd.Index(["v1", "v2", "corr"])
                        matriz_melt["v1"] = matriz_melt["v1"].astype(str)
                        matriz_melt["v2"] = matriz_melt["v2"].astype(str)

                        color_mapper = LinearColorMapper(palette="Viridis256", low=-1, high=1)
                        fonte_corr = ColumnDataSource(matriz_melt)

                        nomes_cols = [str(c) for c in matriz_corr.columns]
                        fig = figure(
                            title=cfg.titulo,
                            x_range=FactorRange(factors=nomes_cols),
                            y_range=FactorRange(factors=list(reversed(nomes_cols))),
                            width=550,
                            height=450,
                            tools="hover,save",
                            tooltips=[("Variáveis", "@v1 vs @v2"), ("Correlação", "@corr{0.00}")],
                        )
                        fig.rect(
                            x="v1",
                            y="v2",
                            width=1,
                            height=1,
                            source=fonte_corr,
                            fill_color={"field": "corr", "transform": color_mapper},
                            line_color="#1e293b",
                        )
                        fig.xaxis.major_label_orientation = 0.5

                if fig is not None:
                    plots.append(fig)

            if not plots:
                fig_vazio = figure(title="Aviso", width=800, height=200)
                fig_vazio.text(x=[0], y=[0], text=["Nenhum gráfico gerado para as configurações fornecidas."])
                plots.append(fig_vazio)

            layout = column(*plots, sizing_mode="scale_width")
            tema = built_in_themes.get("dark_minimal")
            conteudo_html = file_html(layout, CDN, title="EDA Imóveis Ribeirão Preto - Bokeh", theme=tema)

            with open(caminho_saida, "w", encoding="utf-8") as arquivo:
                arquivo.write(conteudo_html)

            return caminho_saida
        except Exception as erro:
            raise ErroGeradorEda(f"Falha ao gerar relatório Bokeh em '{caminho_saida}': {erro}") from erro
