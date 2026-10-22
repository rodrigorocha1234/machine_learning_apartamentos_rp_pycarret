from __future__ import annotations
from typing import final
import pandas as pd
import plotly.graph_objects as go
from app.contratos_dados.gerador_eda import ConfiguracaoAnaliseGrafica


@final
class ConstrutorTabela:
    """Constrói e formata componentes interativos de tabela (go.Table) para relatórios de EDA."""

    def construir_tabela(
        self,
        dados: pd.DataFrame,
        configuracao: ConfiguracaoAnaliseGrafica,
        coluna_alvo: str
    ) -> go.Figure:
        """Constrói uma figura Plotly com go.Table baseada na configuração declarada no YAML.

        Args:
            dados: DataFrame filtrado e preparado para a análise.
            configuracao: Metadados da análise (agrupamentos, métricas, colunas, ordenação).
            coluna_alvo: Nome da variável dependente principal.

        Returns:
            Instância de go.Figure com go.Table estilizada para Dark Mode.
        """
        if configuracao.agrupar_por and configuracao.agrupar_por in dados.columns:
            df_tabela = self._construir_tabela_agrupada(dados, configuracao, coluna_alvo)
        else:
            df_tabela = self._construir_tabela_registros(dados, configuracao)

        df_exibicao = self._formatar_celulas(df_tabela)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[f"<b>{c}</b>" for c in df_exibicao.columns],
                        fill_color="#1e293b",
                        align="left",
                        font=dict(color="#38bdf8", size=13),
                        height=34,
                    ),
                    cells=dict(
                        values=[df_exibicao[c].tolist() for c in df_exibicao.columns],
                        fill_color="#0f172a",
                        align="left",
                        font=dict(color="#f8fafc", size=12),
                        height=28,
                    ),
                )
            ]
        )
        fig.update_layout(
            title=configuracao.titulo,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    def _construir_tabela_agrupada(
        self,
        dados: pd.DataFrame,
        configuracao: ConfiguracaoAnaliseGrafica,
        coluna_alvo: str
    ) -> pd.DataFrame:
        col_grupo = str(configuracao.agrupar_por)
        col_alvo_analise = configuracao.coluna or configuracao.coluna_y or coluna_alvo
        metricas = configuracao.metricas or ["contagem", "media", "mediana", "min", "max"]

        aggs: dict[str, tuple[str, str]] = {}
        if "contagem" in metricas:
            aggs["Imóveis"] = (col_alvo_analise, "count")
        if col_alvo_analise in dados.columns and pd.api.types.is_numeric_dtype(dados[col_alvo_analise]):
            if "media" in metricas:
                aggs["Média (R$)"] = (col_alvo_analise, "mean")
            if "mediana" in metricas:
                aggs["Mediana (R$)"] = (col_alvo_analise, "median")
            if "min" in metricas:
                aggs["Mínimo (R$)"] = (col_alvo_analise, "min")
            if "max" in metricas:
                aggs["Máximo (R$)"] = (col_alvo_analise, "max")

        if not aggs:
            aggs["Imóveis"] = (col_alvo_analise, "count")

        df_tab = dados.groupby(col_grupo, observed=True).agg(**aggs).reset_index()

        col_ord = configuracao.ordenar_por
        asc = configuracao.direcao_ordenacao == "asc"
        if col_ord:
            mapa_ord = {
                "contagem": "Imóveis",
                "imoveis": "Imóveis",
                "media": "Média (R$)",
                "mediana": "Mediana (R$)",
                "min": "Mínimo (R$)",
                "max": "Máximo (R$)",
            }
            col_ord_efetiva = mapa_ord.get(col_ord.lower(), col_ord)
            if col_ord_efetiva in df_tab.columns:
                df_tab = df_tab.sort_values(by=col_ord_efetiva, ascending=asc)
        else:
            if "Imóveis" in df_tab.columns:
                df_tab = df_tab.sort_values(by="Imóveis", ascending=False)

        if configuracao.limite and configuracao.limite > 0:
            df_tab = df_tab.head(configuracao.limite)

        return df_tab

    def _construir_tabela_registros(
        self,
        dados: pd.DataFrame,
        configuracao: ConfiguracaoAnaliseGrafica
    ) -> pd.DataFrame:
        cols_selecionadas = [
            c for c in (configuracao.colunas or dados.columns.tolist())
            if c in dados.columns
        ]
        if not cols_selecionadas:
            cols_selecionadas = dados.columns.tolist()

        df_tab = dados[cols_selecionadas].copy()
        if configuracao.ordenar_por and configuracao.ordenar_por in df_tab.columns:
            df_tab = df_tab.sort_values(
                by=configuracao.ordenar_por,
                ascending=(configuracao.direcao_ordenacao == "asc")
            )
        if configuracao.limite and configuracao.limite > 0:
            df_tab = df_tab.head(configuracao.limite)

        return df_tab

    def _formatar_celulas(self, df_tabela: pd.DataFrame) -> pd.DataFrame:
        df_exibicao = df_tabela.copy()
        for col_nome in df_exibicao.columns:
            if pd.api.types.is_float_dtype(df_exibicao[col_nome]):
                if any(termo in col_nome.lower() for termo in ["r$", "preço", "preco", "valor", "venda"]):
                    df_exibicao[col_nome] = df_exibicao[col_nome].map(
                        lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        if pd.notnull(v) else "-"
                    )
                else:
                    df_exibicao[col_nome] = df_exibicao[col_nome].map(
                        lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        if pd.notnull(v) else "-"
                    )
            elif pd.api.types.is_integer_dtype(df_exibicao[col_nome]):
                df_exibicao[col_nome] = df_exibicao[col_nome].map(
                    lambda v: f"{v:,}".replace(",", ".") if pd.notnull(v) else "-"
                )
        return df_exibicao
