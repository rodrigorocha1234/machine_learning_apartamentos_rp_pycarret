from __future__ import annotations
import os
from typing import final, override
import pandas as pd
import sweetviz as sv
from app.contratos_dados.gerador_eda import ErroGeradorEda, GeradorEda


@final
class GeradorSweetviz(GeradorEda):
    """Implementação concreta de GeradorEda utilizando a biblioteca Sweetviz."""

    @override
    def gerar_relatorio(
        self,
        dados: pd.DataFrame,
        coluna_alvo: str,
        caminho_saida: str,
        analises_configuradas: list[dict[str, ValorConfiguracaoEda]] | None = None,
    ) -> str:
        if dados.empty:
            raise ErroGeradorEda("O DataFrame fornecido para geração do EDA via Sweetviz está vazio.")

        try:
            diretorio_destino = os.path.dirname(caminho_saida)
            if diretorio_destino and not os.path.exists(diretorio_destino):
                os.makedirs(diretorio_destino, exist_ok=True)

            target_valido = bool(coluna_alvo and coluna_alvo in dados.columns)
            feat_cfg: sv.FeatureConfig | None = None

            if target_valido and pd.api.types.is_numeric_dtype(dados[coluna_alvo]):
                feat_cfg = sv.FeatureConfig(force_num=[coluna_alvo])

            relatorio = sv.analyze(
                source=dados,
                target_feat=coluna_alvo if target_valido else None,
                feat_cfg=feat_cfg,
            )

            relatorio.show_html(filepath=caminho_saida, open_browser=False)
            return caminho_saida
        except Exception as erro:
            raise ErroGeradorEda(f"Falha ao gerar relatório Sweetviz em '{caminho_saida}': {erro}") from erro
