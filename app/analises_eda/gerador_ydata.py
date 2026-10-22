from __future__ import annotations
import os
import re
from typing import final, override
import pandas as pd
from ydata_profiling import ProfileReport
from app.contratos_dados.gerador_eda import (
    ErroGeradorEda,
    GeradorEda,
    ValorConfiguracaoEda,
)


@final
class GeradorYData(GeradorEda):
    """Implementação de GeradorEda gerando relatório estatístico e auditoria via YData Profiling."""

    @override
    def gerar_relatorio(
        self,
        dados: pd.DataFrame,
        coluna_alvo: str,
        caminho_saida: str,
        analises_configuradas: list[dict[str, ValorConfiguracaoEda]] | None = None,
    ) -> str:
        if dados.empty:
            raise ErroGeradorEda("O DataFrame fornecido para geração do EDA via YData Profiling está vazio.")

        try:
            diretorio_destino = os.path.dirname(caminho_saida)
            if diretorio_destino and not os.path.exists(diretorio_destino):
                os.makedirs(diretorio_destino, exist_ok=True)

            titulo = f"Auditoria e Análise Exploratória - YData Profiling (Alvo: {coluna_alvo})"
            relatorio = ProfileReport(
                dados,
                title=titulo,
                explorative=True,
                minimal=True if len(dados) > 5000 else False,
            )

            html_bruto = relatorio.to_html()
            html_ajustado = self._corrigir_layout_tabelas(html_bruto)

            with open(caminho_saida, "w", encoding="utf-8") as f:
                f.write(html_ajustado)

            return caminho_saida
        except Exception as erro:
            raise ErroGeradorEda(f"Falha ao gerar relatório YData Profiling em '{caminho_saida}': {erro}") from erro

    def _corrigir_layout_tabelas(self, html: str) -> str:
        """Ajusta o layout e alinhamento dos cabeçalhos das tabelas (Sample e DataFrame) no HTML do YData Profiling."""
        css_correcao = (
            "<style>\n"
            "/* Correção do layout e alinhamento dos cabeçalhos das tabelas do YData Profiling */\n"
            "table.dataframe th, table.sample th, .table th, .table thead th, .table-responsive table th {\n"
            "  text-align: left !important;\n"
            "  vertical-align: middle !important;\n"
            "  padding: 0.6rem 0.85rem !important;\n"
            "  font-weight: 600 !important;\n"
            "  color: #212529 !important;\n"
            "  border-bottom: 2px solid #dee2e6 !important;\n"
            "}\n"
            "table.dataframe td, table.sample td, .table td, .table-responsive table td {\n"
            "  text-align: left !important;\n"
            "  vertical-align: middle !important;\n"
            "  padding: 0.55rem 0.85rem !important;\n"
            "}\n"
            "table.dataframe thead th:first-child, table.sample thead th:first-child {\n"
            "  color: #6c757d !important;\n"
            "  width: 60px !important;\n"
            "  min-width: 60px !important;\n"
            "  text-align: center !important;\n"
            "}\n"
            "table.dataframe tbody th, table.sample tbody th {\n"
            "  color: #6c757d !important;\n"
            "  font-weight: 600 !important;\n"
            "  width: 60px !important;\n"
            "  min-width: 60px !important;\n"
            "  text-align: center !important;\n"
            "  background-color: rgba(0, 0, 0, 0.02) !important;\n"
            "}\n"
            "table.dataframe tr:hover, table.sample tr:hover {\n"
            "  background-color: rgba(13, 110, 253, 0.04) !important;\n"
            "}\n"
            "</style>\n"
        )

        # 1. Remove style="text-align: right;" gerado pelo pandas no thead
        html_ajustado = re.sub(r'<tr style="text-align:\s*right;?">', "<tr>", html, flags=re.IGNORECASE)

        # 2. Preenche primeiro <th> vazio de índice com indicador semântico '#'
        html_ajustado = re.sub(
            r"<thead>\s*<tr>\s*<th>(?=<th>)", "<thead><tr><th>#</th>", html_ajustado, flags=re.IGNORECASE
        )

        # 3. Injeta o CSS corretivo antes de </head>
        if "</head>" in html_ajustado:
            html_ajustado = html_ajustado.replace("</head>", f"{css_correcao}</head>")
        else:
            html_ajustado = f"{css_correcao}{html_ajustado}"

        return html_ajustado
