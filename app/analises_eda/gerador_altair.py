from __future__ import annotations
import os
from typing import final, override
import altair as alt
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
class GeradorAltair(GeradorEda):
    """Implementação de GeradorEda gerando gráficos interativos via Altair (Vega-Lite)."""

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
            raise ErroGeradorEda("O DataFrame fornecido para geração do EDA via Altair está vazio.")

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

            colunas_num = dados.select_dtypes(include=["number"]).columns.tolist()
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

            graficos: list[alt.Chart] = []

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

                chart: alt.Chart | None = None

                if cfg.tipo_grafico in ("histograma", "histogram"):
                    col = cfg.coluna or cfg.coluna_x or coluna_alvo
                    if col in df_item.columns:
                        chart = (
                            alt.Chart(df_item)
                            .mark_bar(color="#38bdf8")
                            .encode(
                                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=cfg.nbins), title=col),
                                y=alt.Y("count():Q", title="Frequência"),
                                tooltip=[
                                    alt.Tooltip(f"{col}:Q", bin=True, title="Faixa"),
                                    alt.Tooltip("count():Q", title="Volume"),
                                ],
                            )
                            .properties(title=cfg.titulo, width=700, height=280)
                        )

                elif cfg.tipo_grafico in ("boxplot", "box"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if y_col in df_item.columns:
                        encode_args = {
                            "y": alt.Y(f"{y_col}:Q", title=y_col),
                            "color": alt.value("#818cf8"),
                        }
                        if x_col and x_col in df_item.columns:
                            encode_args["x"] = alt.X(f"{x_col}:N", title=x_col)
                        chart = (
                            alt.Chart(df_item)
                            .mark_boxplot(extent="min-max")
                            .encode(**encode_args)
                            .properties(title=cfg.titulo, width=700, height=280)
                        )

                elif cfg.tipo_grafico in ("dispersao", "scatter"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if x_col and y_col and x_col in df_item.columns and y_col in df_item.columns:
                        encode_kwargs: dict[str, object] = {
                            "x": alt.X(f"{x_col}:Q", title=x_col),
                            "y": alt.Y(f"{y_col}:Q", title=y_col),
                            "tooltip": [alt.Tooltip(f"{x_col}:Q"), alt.Tooltip(f"{y_col}:Q")],
                        }
                        if cfg.cor and cfg.cor in df_item.columns:
                            encode_kwargs["color"] = alt.Color(f"{cfg.cor}:N", title=cfg.cor)
                        else:
                            encode_kwargs["color"] = alt.value("#38bdf8")

                        chart = (
                            alt.Chart(df_item)
                            .mark_circle(size=60, opacity=0.7)
                            .encode(**encode_kwargs)
                            .properties(title=cfg.titulo, width=700, height=280)
                            .interactive()
                        )

                elif cfg.tipo_grafico in ("barras", "bar"):
                    x_col = cfg.coluna_x
                    y_col = cfg.coluna_y or coluna_alvo
                    if x_col and x_col in df_item.columns:
                        agregacao = "median" if cfg.agregacao == "mediana" else ("mean" if cfg.agregacao == "media" else "count")
                        chart = (
                            alt.Chart(df_item)
                            .mark_bar(color="#34d399")
                            .encode(
                                x=alt.X(f"{x_col}:N", sort="-y", title=x_col),
                                y=alt.Y(f"{agregacao}({y_col}):Q" if y_col in df_item.columns and agregacao != "count" else "count():Q", title=f"{cfg.agregacao.title()} de {y_col or 'Registros'}"),
                                tooltip=[alt.Tooltip(f"{x_col}:N")],
                            )
                            .properties(title=cfg.titulo, width=700, height=280)
                        )

                elif cfg.tipo_grafico in ("correlacao", "heatmap"):
                    colunas_numericas = df_item.select_dtypes(include=["number"]).columns.tolist()
                    if len(colunas_numericas) >= 2:
                        matriz = df_item[colunas_numericas].corr().reset_index().melt(id_vars="index")
                        matriz.columns = pd.Index(["Variavel_1", "Variavel_2", "Correlacao"])
                        chart = (
                            alt.Chart(matriz)
                            .mark_rect()
                            .encode(
                                x=alt.X("Variavel_1:N", title=None),
                                y=alt.Y("Variavel_2:N", title=None),
                                color=alt.Color("Correlacao:Q", scale=alt.Scale(scheme="viridis")),
                                tooltip=[
                                    alt.Tooltip("Variavel_1:N"),
                                    alt.Tooltip("Variavel_2:N"),
                                    alt.Tooltip("Correlacao:Q", format=".2f"),
                                ],
                            )
                            .properties(title=cfg.titulo, width=500, height=400)
                        )

                if chart is not None:
                    graficos.append(chart)

            if not graficos:
                chart_vazio = (
                    alt.Chart(pd.DataFrame({"mensagem": ["Nenhuma visualização gerada"]}))
                    .mark_text(size=16, color="#f8fafc")
                    .encode(text="mensagem:N")
                    .properties(title="Aviso", width=600, height=200)
                )
                graficos.append(chart_vazio)

            layout_final = (
                alt.vconcat(*graficos)
                .configure(background="#0f172a")
                .configure_axis(gridColor="#1e293b", labelColor="#94a3b8", titleColor="#f8fafc")
                .configure_title(color="#f8fafc", fontSize=14)
                .configure_legend(labelColor="#94a3b8", titleColor="#f8fafc")
            )

            layout_final.save(caminho_saida)
            return caminho_saida
        except Exception as erro:
            raise ErroGeradorEda(f"Falha ao gerar relatório Altair em '{caminho_saida}': {erro}") from erro
