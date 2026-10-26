"""Avaliador de integridade, completude e conformidade de domínio dos dados."""

from __future__ import annotations

from typing import cast, final
import numpy as np
import pandas as pd

from app.contratos_drift.detector_drift import MetricasQualidadeDados


@final
class AvaliadorQualidade:
    """Calcula indicadores de qualidade, valores nulos e categorias anômalas."""

    def avaliar_qualidade(
        self,
        dados_referencia: pd.DataFrame,
        dados_producao: pd.DataFrame,
    ) -> MetricasQualidadeDados:
        """Compara o lote de produção contra as regras de integridade da referência."""
        if dados_producao.empty:
            return MetricasQualidadeDados(
                total_amostras=0,
                taxa_nulos_geral=0.0,
                contagem_nulos_por_coluna={},
                novas_categorias_detectadas={},
                valores_fora_limites={},
            )

        total_amostras = len(dados_producao)
        total_celulas = dados_producao.size
        nulos_totais = int(dados_producao.isna().to_numpy().sum())
        taxa_nulos_geral = float(nulos_totais / total_celulas) if total_celulas > 0 else 0.0

        contagem_nulos: dict[str, int] = {}
        for col_obj in cast(list[object], list(dados_producao.columns)):
            col = str(col_obj)
            arr_nulos = np.asarray(dados_producao[col].isna(), dtype=np.bool_)
            qtd_nulos = int(np.sum(arr_nulos))
            if qtd_nulos > 0:
                contagem_nulos[col] = qtd_nulos

        novas_categorias: dict[str, list[str]] = {}
        valores_fora_limites: dict[str, int] = {}

        for col_obj in cast(list[object], list(dados_referencia.columns)):
            col = str(col_obj)
            if col not in dados_producao.columns:
                continue

            s_ref = cast(pd.Series, dados_referencia[col].dropna())
            s_prod = cast(pd.Series, dados_producao[col].dropna())

            if s_prod.empty:
                continue

            if pd.api.types.is_numeric_dtype(s_ref):
                # Verificação de limites mínimo e máximo históricos
                val_min = cast(object, s_ref.min())
                val_max = cast(object, s_ref.max())
                min_ref = float(str(val_min))
                max_ref = float(str(val_max))

                arr_prod = np.asarray(s_prod, dtype=np.float64)
                fora_limite = int(np.sum((arr_prod < min_ref) | (arr_prod > max_ref)))
                if fora_limite > 0:
                    valores_fora_limites[col] = fora_limite
            else:
                # Verificação de categorias novas não observadas no treino
                lista_ref: list[str] = [str(x) for x in cast(list[object], s_ref.to_list())]
                lista_prod: list[str] = [str(x) for x in cast(list[object], s_prod.to_list())]
                cats_ref: set[str] = set(lista_ref)
                cats_prod: set[str] = set(lista_prod)
                diferenca = sorted(list(cats_prod - cats_ref))
                if diferenca:
                    novas_categorias[col] = diferenca

        return MetricasQualidadeDados(
            total_amostras=total_amostras,
            taxa_nulos_geral=round(taxa_nulos_geral, 4),
            contagem_nulos_por_coluna=contagem_nulos,
            novas_categorias_detectadas=novas_categorias,
            valores_fora_limites=valores_fora_limites,
        )
