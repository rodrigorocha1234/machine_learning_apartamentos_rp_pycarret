"""Serviço desacoplado para cálculos estatísticos univariados e bivariados da EDA."""

from __future__ import annotations

import math
import statistics
from typing import cast, final
import pandas as pd

from app.contratos_visualizacao.adaptador_visualizacao import ResultadoAnaliseEda
from app.esquema_configuracao import ItemAnaliseEda
from app.motores_tabela.motor_pandas import MotorPandas


class ErroCalculoEstatistico(Exception):
    """Exceção levantada em falhas durante cálculos analíticos da EDA."""


@final
class CalculadorEstatistico:
    """Motor matemático e estatístico responsável por gerar sumários e métricas para a EDA."""

    def __init__(self, motor_expressao: MotorPandas | None = None) -> None:
        self._motor_expressao = motor_expressao if motor_expressao is not None else MotorPandas()

    def calcular_analise(
        self, dados: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        """Executa o cálculo estatístico de acordo com a especificação do item."""
        if dados.empty:
            raise ErroCalculoEstatistico(
                f"DataFrame vazio ao calcular análise '{item.nome}'."
            )

        df_trabalho = dados.copy()

        # 1. Avaliação segura de expressão prévia (se configurada)
        if item.expressao:
            coluna_destino = item.coluna if item.coluna else f"_derivada_{item.nome}"
            try:
                df_trabalho = self._motor_expressao.transformar_dados(
                    dados=df_trabalho, expressao=item.expressao, destino=coluna_destino
                )
            except Exception as erro_expr:
                raise ErroCalculoEstatistico(
                    f"Falha ao avaliar expressão segura na análise '{item.nome}': {erro_expr}"
                ) from erro_expr

        # 2. Despacho por tipo de gráfico
        if item.tipo_grafico == "histograma":
            return self._calcular_histograma(df_trabalho, item)
        if item.tipo_grafico == "boxplot":
            return self._calcular_boxplot(df_trabalho, item)
        if item.tipo_grafico == "dispersao":
            return self._calcular_dispersao(df_trabalho, item)
        if item.tipo_grafico == "correlacao":
            return self._calcular_correlacao(df_trabalho, item)
        if item.tipo_grafico == "tabela":
            return self._calcular_tabela_resumo(df_trabalho, item)
        if item.tipo_grafico == "barras":
            return self._calcular_barras(df_trabalho, item)

        raise ErroCalculoEstatistico(f"Tipo de gráfico não suportado: '{item.tipo_grafico}'.")

    def _extrair_lista_float(self, df: pd.DataFrame, coluna: str) -> list[float]:
        if coluna not in df.columns:
            raise ErroCalculoEstatistico(f"Coluna '{coluna}' não encontrada no DataFrame.")
        coluna_dados = df[coluna]  # pyright: ignore[reportUnknownVariableType]
        valores: list[float] = []
        if isinstance(coluna_dados, pd.Series):
            for i in range(len(coluna_dados)):
                val = coluna_dados.iloc[i]  # pyright: ignore[reportUnknownVariableType]
                try:
                    fval = float(cast(float, val))
                    if not math.isnan(fval):
                        valores.append(fval)
                except (ValueError, TypeError):
                    continue
        if not valores:
            raise ErroCalculoEstatistico(f"Coluna '{coluna}' não contém valores numéricos válidos.")
        return valores

    def _extrair_lista_str(self, df: pd.DataFrame, coluna: str) -> list[str]:
        if coluna not in df.columns:
            raise ErroCalculoEstatistico(f"Coluna '{coluna}' não encontrada no DataFrame.")
        coluna_dados = df[coluna]  # pyright: ignore[reportUnknownVariableType]
        valores: list[str] = []
        if isinstance(coluna_dados, pd.Series):
            for i in range(len(coluna_dados)):
                val = coluna_dados.iloc[i]  # pyright: ignore[reportUnknownVariableType]
                if val is not None:
                    if isinstance(val, float) and math.isnan(val):
                        continue
                    valores.append(str(cast(object, val)))
        return valores

    def _extrair_pares_alinhados(
        self, df: pd.DataFrame, col_x: str, col_y: str, col_cor: str | None = None
    ) -> tuple[list[float], list[float], list[str]]:
        if col_x not in df.columns or col_y not in df.columns:
            raise ErroCalculoEstatistico(f"Colunas '{col_x}' ou '{col_y}' inexistentes.")
        dados_x = df[col_x]  # pyright: ignore[reportUnknownVariableType]
        dados_y = df[col_y]  # pyright: ignore[reportUnknownVariableType]
        dados_cor = df[col_cor] if col_cor and col_cor in df.columns else None  # pyright: ignore[reportUnknownVariableType]

        res_x: list[float] = []
        res_y: list[float] = []
        res_cor: list[str] = []

        if isinstance(dados_x, pd.Series) and isinstance(dados_y, pd.Series):
            for i in range(len(dados_x)):
                val_x = dados_x.iloc[i]  # pyright: ignore[reportUnknownVariableType]
                val_y = dados_y.iloc[i]  # pyright: ignore[reportUnknownVariableType]
                try:
                    fx = float(cast(float, val_x))
                    fy = float(cast(float, val_y))
                    if not math.isnan(fx) and not math.isnan(fy):
                        res_x.append(fx)
                        res_y.append(fy)
                        if dados_cor is not None and isinstance(dados_cor, pd.Series):
                            val_cor = dados_cor.iloc[i]  # pyright: ignore[reportUnknownVariableType]
                            res_cor.append(str(cast(object, val_cor)))
                        else:
                            res_cor.append("")
                except (ValueError, TypeError):
                    continue

        return res_x, res_y, res_cor

    def _calcular_pearson(self, x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        media_x = sum(x) / n
        media_y = sum(y) / n
        var_x = sum((xi - media_x) ** 2 for xi in x)
        var_y = sum((yi - media_y) ** 2 for yi in y)
        if var_x == 0.0 or var_y == 0.0:
            return 0.0
        cov_xy = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
        return cov_xy / math.sqrt(var_x * var_y)

    def _calcular_histograma(
        self, df: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        if not item.coluna:
            raise ErroCalculoEstatistico(f"Coluna focal obrigatória para histograma '{item.nome}'.")

        valores = sorted(self._extrair_lista_float(df, item.coluna))
        n_total = len(valores)
        n_bins = item.nbins if item.nbins is not None and item.nbins > 0 else 20

        min_val = valores[0]
        max_val = valores[-1]
        amplitude = max_val - min_val
        passo = (amplitude / n_bins) if amplitude > 0 else 1.0

        bordas: list[float] = [round(min_val + i * passo, 2) for i in range(n_bins + 1)]
        frequencias: list[int] = [0] * n_bins

        for v in valores:
            idx = int((v - min_val) / passo) if passo > 0 else 0
            if idx >= n_bins:
                idx = n_bins - 1
            frequencias[idx] += 1

        centros_bins = [
            round((bordas[i] + bordas[i + 1]) / 2.0, 2) for i in range(n_bins)
        ]

        media = statistics.mean(valores)
        mediana = statistics.median(valores)
        desvio = statistics.stdev(valores) if n_total > 1 else 0.0
        quartis = statistics.quantiles(valores, n=4) if n_total >= 4 else [min_val, mediana, max_val]

        # Assimetria amostral (skewness)
        diferencas = [v - media for v in valores]
        m2 = sum(d**2 for d in diferencas) / n_total
        m3 = sum(d**3 for d in diferencas) / n_total
        assimetria = (m3 / (m2**1.5)) if m2 > 0 else 0.0

        metricas: dict[str, float] = {
            "contagem_total": float(n_total),
            "media": round(media, 2),
            "mediana": round(mediana, 2),
            "desvio_padrao": round(desvio, 2),
            "minimo": round(min_val, 2),
            "maximo": round(max_val, 2),
            "quartil_25": round(quartis[0], 2),
            "quartil_75": round(quartis[-1], 2),
            "assimetria": round(assimetria, 4),
        }

        dados_estruturados: dict[str, object] = {
            "coluna": item.coluna,
            "nbins": n_bins,
            "bordas_bins": bordas,
            "centros_bins": centros_bins,
            "frequencias": frequencias,
        }

        return ResultadoAnaliseEda(
            nome=item.nome,
            titulo=item.titulo or f"Distribuição de {item.coluna}",
            tipo_grafico="histograma",
            descricao=item.descricao,
            coluna_principal=item.coluna,
            dados_estruturados=dados_estruturados,
            metricas_resumo=metricas,
        )

    def _calcular_boxplot(
        self, df: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        col_x = item.coluna_x
        col_y = item.coluna_y or item.coluna
        if not col_y:
            raise ErroCalculoEstatistico(f"Coluna Y necessária para o boxplot '{item.nome}'.")

        if col_x is None or col_x not in df.columns:
            valores = sorted(self._extrair_lista_float(df, col_y))
            q1, mediana, q3 = (
                statistics.quantiles(valores, n=4)
                if len(valores) >= 4
                else [min(valores), statistics.median(valores), max(valores)]
            )
            iqr = q3 - q1
            lim_inf = max(min(valores), q1 - 1.5 * iqr)
            lim_sup = min(max(valores), q3 + 1.5 * iqr)
            outliers = [v for v in valores if v < lim_inf or v > lim_sup]

            categorias_dados: dict[str, object] = {
                "Geral": {
                    "contagem": len(valores),
                    "q1": round(q1, 2),
                    "mediana": round(mediana, 2),
                    "q3": round(q3, 2),
                    "iqr": round(iqr, 2),
                    "limite_inferior": round(lim_inf, 2),
                    "limite_superior": round(lim_sup, 2),
                    "quantidade_outliers": len(outliers),
                }
            }
            metricas: dict[str, float] = {
                "mediana_geral": round(mediana, 2),
                "iqr_geral": round(iqr, 2),
                "total_outliers": float(len(outliers)),
            }
        else:
            cats = self._extrair_lista_str(df, col_x)
            vals = self._extrair_lista_float(df, col_y)
            limite_len = min(len(cats), len(vals))

            grupos: dict[str, list[float]] = {}
            for i in range(limite_len):
                grupos.setdefault(cats[i], []).append(vals[i])

            categorias_dados: dict[str, object] = {}
            total_outliers = 0

            for cat, valores_cat in grupos.items():
                if not valores_cat:
                    continue
                v_ord = sorted(valores_cat)
                q1, mediana, q3 = (
                    statistics.quantiles(v_ord, n=4)
                    if len(v_ord) >= 4
                    else [min(v_ord), statistics.median(v_ord), max(v_ord)]
                )
                iqr = q3 - q1
                lim_inf = max(min(v_ord), q1 - 1.5 * iqr)
                lim_sup = min(max(v_ord), q3 + 1.5 * iqr)
                outliers = [v for v in v_ord if v < lim_inf or v > lim_sup]
                total_outliers += len(outliers)

                categorias_dados[cat] = {
                    "contagem": len(v_ord),
                    "q1": round(q1, 2),
                    "mediana": round(mediana, 2),
                    "q3": round(q3, 2),
                    "iqr": round(iqr, 2),
                    "limite_inferior": round(lim_inf, 2),
                    "limite_superior": round(lim_sup, 2),
                    "quantidade_outliers": len(outliers),
                }

            metricas = {
                "total_categorias": float(len(categorias_dados)),
                "total_outliers": float(total_outliers),
            }

        dados_estruturados: dict[str, object] = {
            "coluna_x": col_x,
            "coluna_y": col_y,
            "categorias": list(categorias_dados.keys()),
            "estatisticas_por_categoria": categorias_dados,
        }

        return ResultadoAnaliseEda(
            nome=item.nome,
            titulo=item.titulo or f"Dispersão de {col_y} por {col_x}",
            tipo_grafico="boxplot",
            descricao=item.descricao,
            coluna_principal=col_y,
            dados_estruturados=dados_estruturados,
            metricas_resumo=metricas,
        )

    def _calcular_dispersao(
        self, df: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        col_x = item.coluna_x
        col_y = item.coluna_y
        if not col_x or not col_y:
            raise ErroCalculoEstatistico(
                f"Colunas X e Y são obrigatórias para gráfico de dispersão '{item.nome}'."
            )

        xs, ys, cores = self._extrair_pares_alinhados(df, col_x, col_y, item.cor)
        correlacao = self._calcular_pearson(xs, ys)

        dados_estruturados: dict[str, object] = {
            "coluna_x": col_x,
            "coluna_y": col_y,
            "cor_coluna": item.cor,
            "pontos_x": [round(v, 2) for v in xs],
            "pontos_y": [round(v, 2) for v in ys],
            "cor_labels": cores,
        }

        metricas: dict[str, float] = {
            "total_pontos": float(len(xs)),
            "correlacao_pearson": round(correlacao, 4),
        }

        return ResultadoAnaliseEda(
            nome=item.nome,
            titulo=item.titulo or f"Relação entre {col_x} e {col_y}",
            tipo_grafico="dispersao",
            descricao=item.descricao,
            coluna_principal=col_y,
            dados_estruturados=dados_estruturados,
            metricas_resumo=metricas,
        )

    def _calcular_correlacao(
        self, df: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        cols_numericas: list[str] = []
        dados_colunas: dict[str, list[float]] = {}

        for col in cast(list[object], list(df.columns)):
            nome_str = str(col)
            try:
                valores = self._extrair_lista_float(df, nome_str)
                if len(valores) >= 2:
                    cols_numericas.append(nome_str)
                    dados_colunas[nome_str] = valores
            except Exception:
                continue

        if len(cols_numericas) < 2:
            raise ErroCalculoEstatistico("Necessário ao menos 2 colunas numéricas para correlação.")

        # Alinha tamanhos
        tam_minimo = min(len(dados_colunas[c]) for c in cols_numericas)
        for c in cols_numericas:
            dados_colunas[c] = dados_colunas[c][:tam_minimo]

        matriz_lista: list[list[float]] = []
        pares_fortes: list[dict[str, object]] = []

        for i, c1 in enumerate(cols_numericas):
            linha: list[float] = []
            for j, c2 in enumerate(cols_numericas):
                if i == j:
                    linha.append(1.0)
                else:
                    corr = self._calcular_pearson(dados_colunas[c1], dados_colunas[c2])
                    linha.append(round(corr, 4))
                    if j > i and abs(corr) >= 0.5:
                        pares_fortes.append({"var1": c1, "var2": c2, "correlacao": round(corr, 4)})
            matriz_lista.append(linha)

        dados_estruturados: dict[str, object] = {
            "colunas": cols_numericas,
            "matriz": matriz_lista,
            "pares_fortemente_correlacionados": pares_fortes,
        }

        metricas: dict[str, float] = {
            "total_variaveis_analisadas": float(len(cols_numericas)),
            "quantidade_pares_fortes": float(len(pares_fortes)),
        }

        return ResultadoAnaliseEda(
            nome=item.nome,
            titulo=item.titulo or "Matriz de Correlação Linear",
            tipo_grafico="correlacao",
            descricao=item.descricao,
            coluna_principal=None,
            dados_estruturados=dados_estruturados,
            metricas_resumo=metricas,
        )

    def _calcular_tabela_resumo(
        self, df: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        col_agrupamento = item.agrupar_por
        col_metrica = item.coluna
        if not col_agrupamento or not col_metrica:
            raise ErroCalculoEstatistico(
                f"Coluna de agrupamento e de métrica são necessárias para tabela '{item.nome}'."
            )

        cats = self._extrair_lista_str(df, col_agrupamento)
        vals = self._extrair_lista_float(df, col_metrica)
        tam = min(len(cats), len(vals))

        grupos: dict[str, list[float]] = {}
        for i in range(tam):
            grupos.setdefault(cats[i], []).append(vals[i])

        linhas_tabela: list[dict[str, object]] = []
        for cat, valores_cat in grupos.items():
            if not valores_cat:
                continue
            linhas_tabela.append(
                {
                    "categoria": cat,
                    "contagem": len(valores_cat),
                    "media": round(statistics.mean(valores_cat), 2),
                    "mediana": round(statistics.median(valores_cat), 2),
                    "min": round(min(valores_cat), 2),
                    "max": round(max(valores_cat), 2),
                }
            )

        coluna_ordem = item.ordenar_por if item.ordenar_por else "contagem"
        ordem_desc = item.direcao_ordenacao == "desc"

        def _chave_ordenacao(item_linha: dict[str, object]) -> float:
            val_ord = item_linha.get(coluna_ordem, 0.0)
            if isinstance(val_ord, (int, float)):
                return float(val_ord)
            return 0.0

        linhas_tabela.sort(key=_chave_ordenacao, reverse=ordem_desc)

        if item.limite and item.limite > 0:
            linhas_tabela = linhas_tabela[: item.limite]

        dados_estruturados: dict[str, object] = {
            "agrupar_por": col_agrupamento,
            "coluna_metrica": col_metrica,
            "linhas": linhas_tabela,
        }

        metricas: dict[str, float] = {
            "total_categorias_listadas": float(len(linhas_tabela)),
        }

        return ResultadoAnaliseEda(
            nome=item.nome,
            titulo=item.titulo or f"Resumo Estatístico por {col_agrupamento}",
            tipo_grafico="tabela",
            descricao=item.descricao,
            coluna_principal=col_metrica,
            dados_estruturados=dados_estruturados,
            metricas_resumo=metricas,
        )

    def _calcular_barras(
        self, df: pd.DataFrame, item: ItemAnaliseEda
    ) -> ResultadoAnaliseEda:
        col_cat = item.coluna_x or item.coluna
        if not col_cat:
            raise ErroCalculoEstatistico(f"Coluna de categorias necessária para barras '{item.nome}'.")

        cats = self._extrair_lista_str(df, col_cat)
        contagens: dict[str, int] = {}
        for c in cats:
            contagens[c] = contagens.get(c, 0) + 1

        pares_ordenados = sorted(contagens.items(), key=lambda p: p[1], reverse=True)
        if item.limite and item.limite > 0:
            pares_ordenados = pares_ordenados[: item.limite]

        categorias = [p[0] for p in pares_ordenados]
        frequencias = [p[1] for p in pares_ordenados]

        dados_estruturados: dict[str, object] = {
            "coluna": col_cat,
            "categorias": categorias,
            "frequencias": frequencias,
        }

        metricas: dict[str, float] = {
            "total_categorias": float(len(categorias)),
        }

        return ResultadoAnaliseEda(
            nome=item.nome,
            titulo=item.titulo or f"Frequência por {col_cat}",
            tipo_grafico="barras",
            descricao=item.descricao,
            coluna_principal=col_cat,
            dados_estruturados=dados_estruturados,
            metricas_resumo=metricas,
        )
