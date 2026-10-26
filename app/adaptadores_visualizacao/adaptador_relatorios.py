"""Adaptador de visualização para geração de relatórios HTML auto-contidos."""

from __future__ import annotations

import html
from pathlib import Path
from typing import cast, final, override

from app.contratos_visualizacao.adaptador_visualizacao import (
    AdaptadorVisualizacao,
    ResultadoAnaliseEda,
    ResultadoVisualizacao,
)
from app.esquema_configuracao import ConfiguracaoEda


class ErroAdaptadorRelatorios(Exception):
    """Exceção levantada em falhas na renderização do relatório HTML."""


@final
class AdaptadorRelatorios(AdaptadorVisualizacao):
    """Adaptador concreto para geração de relatório analítico em HTML com gráficos SVG."""

    @override
    def obter_nome_motor(self) -> str:
        return "relatorios"

    @override
    def renderizar_analises(
        self, analises: list[ResultadoAnaliseEda], configuracao: ConfiguracaoEda
    ) -> ResultadoVisualizacao:
        if not analises:
            return ResultadoVisualizacao(
                motor_utilizado=self.obter_nome_motor(),
                total_paineis=0,
                arquivos_gerados=[],
                sucesso=True,
                mensagem="Nenhuma análise para renderização no relatório HTML.",
            )

        caminho_html = Path(configuracao.caminho_relatorio_html)
        caminho_html.parent.mkdir(parents=True, exist_ok=True)

        try:
            conteudo_html = self._gerar_html(analises)
            caminho_html.write_text(conteudo_html, encoding="utf-8")

            return ResultadoVisualizacao(
                motor_utilizado=self.obter_nome_motor(),
                total_paineis=len(analises),
                arquivos_gerados=[str(caminho_html)],
                sucesso=True,
                mensagem=f"Relatório HTML gerado com sucesso em '{caminho_html}'.",
            )
        except Exception as erro:
            raise ErroAdaptadorRelatorios(
                f"Falha ao gerar relatório HTML de EDA: {erro}"
            ) from erro

    def _extrair_lista_floats(self, lista_raw: object) -> list[float]:
        if not isinstance(lista_raw, list):
            return []
        res: list[float] = []
        for x in cast(list[object], lista_raw):
            if isinstance(x, (int, float)):
                res.append(float(x))
        return res

    def _extrair_lista_dicts(self, lista_raw: object) -> list[dict[str, object]]:
        if not isinstance(lista_raw, list):
            return []
        res: list[dict[str, object]] = []
        for x in cast(list[object], lista_raw):
            if isinstance(x, dict):
                dict_cast: dict[str, object] = {
                    str(k): v for k, v in cast(dict[object, object], x).items()
                }
                res.append(dict_cast)
        return res

    def _gerar_html(self, analises: list[ResultadoAnaliseEda]) -> str:
        cartoes_html: list[str] = []
        for analise in analises:
            cartoes_html.append(self._renderizar_cartao_analise(analise))

        conteudo_cartoes = "\n".join(cartoes_html)

        return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Relatório de Análise Exploratória (EDA) - Apartamentos Ribeirão Preto</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #818cf8;
      --success: #34d399;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      padding: 2rem 1rem;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }}
    header h1 {{ font-size: 2rem; color: var(--primary); margin-bottom: 0.5rem; }}
    header p {{ color: var(--text-muted); font-size: 1.05rem; }}
    .badge {{
      display: inline-block;
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.5rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
      gap: 1.5rem;
    }}
    .card {{
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    .card.full-width {{ grid-column: 1 / -1; }}
    .card h2 {{
      font-size: 1.25rem;
      margin-bottom: 0.5rem;
      color: var(--text);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-desc {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem; }}
    .metrics-summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1rem;
      background: rgba(15, 23, 42, 0.6);
      padding: 0.75rem;
      border-radius: 0.5rem;
    }}
    .metric-pill {{ font-size: 0.85rem; }}
    .metric-pill strong {{ color: var(--accent); }}
    .svg-container {{
      width: 100%;
      background: rgba(15, 23, 42, 0.4);
      border-radius: 0.5rem;
      padding: 1rem;
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin-top: 0.5rem;
    }}
    th, td {{
      padding: 0.65rem 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--primary); font-weight: 600; background: rgba(15, 23, 42, 0.6); }}
    tr:hover td {{ background: rgba(56, 189, 248, 0.05); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Relatório de Análise Exploratória de Dados (EDA)</h1>
      <p>Previsão de Preços de Apartamentos — Ribeirão Preto / SP</p>
      <div class="badge">Pipeline Antigravity • {len(analises)} Análises Ativas Concluídas</div>
    </header>
    <main class="grid">
      {conteudo_cartoes}
    </main>
  </div>
</body>
</html>
"""

    def _renderizar_cartao_analise(self, analise: ResultadoAnaliseEda) -> str:
        titulo_escapado = html.escape(analise.titulo)
        desc_escapada = html.escape(analise.descricao or analise.nome)
        tipo_badge = f"<span class='badge'>{analise.tipo_grafico.upper()}</span>"

        metricas_html_list: list[str] = []
        for chave, val in analise.metricas_resumo.items():
            metricas_html_list.append(
                f"<div class='metric-pill'>{html.escape(chave)}: <strong>{val:,.2f}</strong></div>"
            )
        metricas_bloco = (
            f"<div class='metrics-summary'>{''.join(metricas_html_list)}</div>"
            if metricas_html_list
            else ""
        )

        visual_html = self._renderizar_conteudo_visual(analise)
        classe_card = "card full-width" if analise.tipo_grafico == "tabela" else "card"

        return f"""
        <article class="{classe_card}">
          <h2>
            <span>{titulo_escapado}</span>
            {tipo_badge}
          </h2>
          <p class="card-desc">{desc_escapada}</p>
          {metricas_bloco}
          <div class="svg-container">
            {visual_html}
          </div>
        </article>
        """

    def _renderizar_conteudo_visual(self, analise: ResultadoAnaliseEda) -> str:
        if analise.tipo_grafico == "histograma":
            return self._gerar_svg_histograma(analise)
        if analise.tipo_grafico == "tabela":
            return self._gerar_html_tabela(analise)
        if analise.tipo_grafico == "correlacao":
            return self._gerar_html_correlacao(analise)
        if analise.tipo_grafico == "boxplot":
            return self._gerar_svg_boxplot(analise)
        if analise.tipo_grafico == "dispersao":
            return self._gerar_svg_dispersao(analise)
        return "<p style='color: var(--text-muted);'>Gráfico sumarizado.</p>"

    def _gerar_svg_histograma(self, analise: ResultadoAnaliseEda) -> str:
        dados = analise.dados_estruturados
        frequencias = self._extrair_lista_floats(dados.get("frequencias"))
        if not frequencias:
            return "<p>Sem dados de frequência.</p>"

        max_freq = max(frequencias) if frequencias else 1.0
        width = 480
        height = 180
        pad_bottom = 20
        pad_top = 10
        usable_height = height - pad_bottom - pad_top
        bar_width = max(2.0, (width / len(frequencias)) - 2.0)

        barras_svg: list[str] = []
        for i, freq in enumerate(frequencias):
            h_bar = (freq / max_freq) * usable_height if max_freq > 0 else 0.0
            x = i * (width / len(frequencias))
            y = height - pad_bottom - h_bar
            barras_svg.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{h_bar:.1f}' fill='#38bdf8' rx='2'/>"
            )

        return f"""
        <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" style="overflow: visible;">
          {''.join(barras_svg)}
          <line x1="0" y1="{height - pad_bottom}" x2="{width}" y2="{height - pad_bottom}" stroke="#475569" stroke-width="1"/>
        </svg>
        """

    def _gerar_svg_boxplot(self, analise: ResultadoAnaliseEda) -> str:
        dados = analise.dados_estruturados
        stats_raw = dados.get("estatisticas_por_categoria")
        if not isinstance(stats_raw, dict) or not stats_raw:
            return "<p>Sem dados de quartis para boxplot.</p>"

        dict_stats = cast(dict[str, object], stats_raw)
        width = 480
        height = 180
        categorias = list(dict_stats.keys())[:8]  # Limita para visualização compacta
        num_cats = len(categorias)
        if num_cats == 0:
            return "<p>Categorias vazias.</p>"

        elementos: list[str] = []
        step_x = width / num_cats
        y_center = height / 2

        for i, cat in enumerate(categorias):
            x = i * step_x + (step_x / 2)
            elementos.append(
                f"<rect x='{x - 14:.1f}' y='{y_center - 25:.1f}' width='28' height='50' fill='rgba(129, 140, 248, 0.4)' stroke='#818cf8' stroke-width='1.5' rx='3'/>"
            )
            elementos.append(
                f"<line x1='{x:.1f}' y1='{y_center - 45:.1f}' x2='{x:.1f}' y2='{y_center + 45:.1f}' stroke='#cbd5e1' stroke-width='1.5'/>"
            )
            elementos.append(
                f"<text x='{x:.1f}' y='{height - 5}' font-size='10' fill='#94a3b8' text-anchor='middle'>{html.escape(str(cat)[:10])}</text>"
            )

        return f"""
        <svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
          {''.join(elementos)}
        </svg>
        """

    def _gerar_svg_dispersao(self, analise: ResultadoAnaliseEda) -> str:
        dados = analise.dados_estruturados
        px = self._extrair_lista_floats(dados.get("pontos_x"))
        py = self._extrair_lista_floats(dados.get("pontos_y"))
        if not px or not py:
            return "<p>Sem dados de dispersão.</p>"

        width = 480
        height = 180
        min_x, max_x = min(px), max(px)
        min_y, max_y = min(py), max(py)
        diff_x = max(max_x - min_x, 1.0)
        diff_y = max(max_y - min_y, 1.0)

        pontos_svg: list[str] = []
        amostra_len = min(len(px), len(py), 150)
        for i in range(amostra_len):
            x_val = px[i]
            y_val = py[i]
            cx = 20 + ((x_val - min_x) / diff_x) * (width - 40)
            cy = (height - 20) - ((y_val - min_y) / diff_y) * (height - 40)
            pontos_svg.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='3' fill='#34d399' opacity='0.75'/>")

        return f"""
        <svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
          <line x1="20" y1="{height - 20}" x2="{width - 20}" y2="{height - 20}" stroke="#475569" stroke-width="1"/>
          <line x1="20" y1="20" x2="20" y2="{height - 20}" stroke="#475569" stroke-width="1"/>
          {''.join(pontos_svg)}
        </svg>
        """

    def _gerar_html_tabela(self, analise: ResultadoAnaliseEda) -> str:
        dados = analise.dados_estruturados
        linhas = self._extrair_lista_dicts(dados.get("linhas"))
        if not linhas:
            return "<p>Tabela sem dados.</p>"

        cabecalho = "<tr><th>Bairro / Categoria</th><th>Contagem</th><th>Média (R$)</th><th>Mediana (R$)</th><th>Mínimo (R$)</th><th>Máximo (R$)</th></tr>"
        linhas_html: list[str] = []
        for item in linhas:
            categoria_val = str(item.get("categoria", ""))
            contagem_val = int(cast(int, item.get("contagem", 0)))
            media_val = float(cast(float, item.get("media", 0.0)))
            mediana_val = float(cast(float, item.get("mediana", 0.0)))
            min_val = float(cast(float, item.get("min", 0.0)))
            max_val = float(cast(float, item.get("max", 0.0)))

            linhas_html.append(
                f"<tr>"
                f"<td><strong>{html.escape(categoria_val)}</strong></td>"
                f"<td>{contagem_val}</td>"
                f"<td>R$ {media_val:,.2f}</td>"
                f"<td>R$ {mediana_val:,.2f}</td>"
                f"<td>R$ {min_val:,.2f}</td>"
                f"<td>R$ {max_val:,.2f}</td>"
                f"</tr>"
            )

        return f"<table><thead>{cabecalho}</thead><tbody>{''.join(linhas_html)}</tbody></table>"

    def _gerar_html_correlacao(self, analise: ResultadoAnaliseEda) -> str:
        dados = analise.dados_estruturados
        pares = self._extrair_lista_dicts(dados.get("pares_fortemente_correlacionados"))
        if not pares:
            return "<p>Nenhum par com correlação linear expressiva (|r| >= 0.50).</p>"

        linhas_html: list[str] = []
        for par in pares:
            var1 = str(par.get("var1", ""))
            var2 = str(par.get("var2", ""))
            corr = float(cast(float, par.get("correlacao", 0.0)))

            linhas_html.append(
                f"<tr>"
                f"<td><strong>{html.escape(var1)}</strong></td>"
                f"<td><strong>{html.escape(var2)}</strong></td>"
                f"<td style='color: var(--primary); font-weight: bold;'>{corr:.4f}</td>"
                f"</tr>"
            )

        return (
            "<table><thead><tr><th>Variável 1</th><th>Variável 2</th><th>Coeficiente Pearson</th></tr></thead>"
            f"<tbody>{''.join(linhas_html)}</tbody></table>"
        )
