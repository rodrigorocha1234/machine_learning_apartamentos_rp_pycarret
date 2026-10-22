from __future__ import annotations
from typing import final


@final
class ConstrutorLayout:
    """Monta o documento HTML completo de relatórios EDA com temas e estilos responsivos."""

    def construir_documento(
        self,
        secoes_graficos: list[str],
        total_amostras: int,
        coluna_alvo: str,
        total_analises: int,
        titulo: str = "Relatório de Análise Exploratória (Plotly)"
    ) -> str:
        """Gera o documento HTML completo estilizado em Dark Mode.

        Args:
            secoes_graficos: Lista de fragmentos HTML contendo os gráficos e tabelas.
            total_amostras: Número total de linhas da base de dados analisada.
            coluna_alvo: Nome da variável dependente principal.
            total_analises: Quantidade de análises ativas executadas.
            titulo: Título principal da página.

        Returns:
            Documento HTML estruturado e completo.
        """
        itens_cards = "".join(f'<div class="grafico-card">{s}</div>' for s in secoes_graficos)

        return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>{titulo}</title>
  <style>
    body {{
      background-color: #0b0f19;
      color: #f3f4f6;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      padding: 2rem;
    }}
    .header {{
      margin-bottom: 2rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 1rem;
    }}
    .header h1 {{
      font-size: 1.8rem;
      color: #38bdf8;
      margin-bottom: 0.5rem;
    }}
    .header p {{
      color: #94a3b8;
      font-size: 0.95rem;
      margin: 0;
    }}
    .grafico-card {{
      background: #111827;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      margin-bottom: 2rem;
      padding: 1.25rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{titulo}</h1>
    <p>Amostras: {total_amostras} | Variável Alvo: {coluna_alvo} | Gráficos Cadastrados no YAML: {total_analises}</p>
  </div>
  {itens_cards}
</body>
</html>"""
