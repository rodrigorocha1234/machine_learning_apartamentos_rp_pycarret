from __future__ import annotations
import json
import os
from typing import final, override
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
class GeradorNativo(GeradorEda):
    """Implementação de GeradorEda nativa utilizando agregações estatísticas do Pandas."""

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
            raise ErroGeradorEda("O DataFrame fornecido para geração do EDA nativo está vazio.")

        try:
            diretorio_destino = os.path.dirname(caminho_saida)
            if diretorio_destino and not os.path.exists(diretorio_destino):
                os.makedirs(diretorio_destino, exist_ok=True)

            resumo = dados.describe(include="all").to_dict()
            nulos = dados.isna().sum().to_dict()

            colunas_num = dados.select_dtypes(include=["number"]).columns.tolist()
            correlacoes: dict[str, dict[str, float]] = {}
            if len(colunas_num) > 1:
                correlacoes = dados[colunas_num].corr().round(4).to_dict()

            estatisticas_alvo: dict[str, float] = {}
            if coluna_alvo in dados.columns and pd.api.types.is_numeric_dtype(dados[coluna_alvo]):
                serie = dados[coluna_alvo].dropna()
                estatisticas_alvo = {
                    "media": float(serie.mean()),
                    "mediana": float(serie.median()),
                    "desvio_padrao": float(serie.std()),
                    "minimo": float(serie.min()),
                    "maximo": float(serie.max()),
                    "p25": float(serie.quantile(0.25)),
                    "p75": float(serie.quantile(0.75)),
                }

            # Processamento das análises configuradas com suporte a expressões e tabelas
            sumario_analises: list[dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]] = []
            if analises_configuradas:
                for item in analises_configuradas:
                    cfg = ConfiguracaoAnaliseGrafica.de_dicionario(item)
                    if not cfg.ativo:
                        continue

                    motor_item = (
                        FabricaMotores.criar_motor(cfg.motor_tabular)
                        if cfg.motor_tabular
                        else self._motor_tabela
                    )

                    df_item = dados.copy()
                    if cfg.expressao_calculo and cfg.coluna_destino:
                        df_item = motor_item.transformar_dados(df_item, cfg.expressao_calculo, cfg.coluna_destino)
                    if cfg.expressao_filtro:
                        df_item = motor_item.filtrar_dados(df_item, cfg.expressao_filtro)

                    info_analise: dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]] = {
                        "nome": cfg.nome,
                        "tipo_grafico": cfg.tipo_grafico,
                        "ativo": cfg.ativo,
                        "total_amostras": int(len(df_item)),
                        "expressao_filtro": cfg.expressao_filtro,
                        "expressao_calculo": cfg.expressao_calculo,
                    }

                    if cfg.tipo_grafico in ("tabela", "table") and cfg.agrupar_por and cfg.agrupar_por in df_item.columns:
                        col_alvo_analise = cfg.coluna or cfg.coluna_y or coluna_alvo
                        agrupado = df_item.groupby(cfg.agrupar_por, observed=True)[col_alvo_analise].agg(["count", "mean", "median"]).reset_index()
                        if cfg.limite and cfg.limite > 0:
                            agrupado = agrupado.head(cfg.limite)
                        info_analise["tabela_agrupada"] = [
                            {str(k): (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in row.items()}
                            for row in agrupado.to_dict(orient="records")
                        ]

                    sumario_analises.append(info_analise)

            conteudo_json = {
                "total_linhas": int(len(dados)),
                "total_colunas": int(len(dados.columns)),
                "coluna_alvo": coluna_alvo,
                "estatisticas_alvo": estatisticas_alvo,
                "resumo": resumo,
                "nulos": nulos,
                "correlacoes": correlacoes,
                "analises_cadastradas": sumario_analises,
            }

            if caminho_saida.endswith(".json"):
                with open(caminho_saida, "w", encoding="utf-8") as f:
                    json.dump(conteudo_json, f, ensure_ascii=False, indent=2)
            else:
                linhas_resumo_html = "".join(
                    f"<tr><td><strong>{col}</strong></td><td>{dados[col].dtype}</td><td>{dados[col].isna().sum()}</td></tr>"
                    for col in dados.columns
                )
                html_doc = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Relatório EDA Nativo</title>
  <style>
    body {{ background: #0b0f19; color: #f3f4f6; font-family: sans-serif; padding: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #1f2937; padding: 0.5rem 1rem; text-align: left; }}
    th {{ background: #111827; color: #38bdf8; }}
  </style>
</head>
<body>
  <h1>Relatório EDA Nativo</h1>
  <p>Total de Linhas: {len(dados)} | Colunas: {len(dados.columns)} | Alvo: {coluna_alvo}</p>
  <h3>Visão Geral das Variáveis</h3>
  <table>
    <thead><tr><th>Coluna</th><th>Tipo</th><th>Nulos</th></tr></thead>
    <tbody>{linhas_resumo_html}</tbody>
  </table>
</body>
</html>"""
                with open(caminho_saida, "w", encoding="utf-8") as f:
                    f.write(html_doc)

            return caminho_saida
        except Exception as erro:
            raise ErroGeradorEda(f"Falha ao gerar relatório EDA nativo em '{caminho_saida}': {erro}") from erro
