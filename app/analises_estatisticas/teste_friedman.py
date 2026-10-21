from __future__ import annotations
from itertools import combinations
import math
from typing import final, override

import pandas as pd
from scipy import stats
from app.contratos_dados.avaliador_estatistico import AvaliadorEstatistico, ResultadoEstatistico

class ErroEstatistico(Exception):
    """Exceção levantada em falhas de cálculos estatísticos de hipótese."""
    pass

@final
class TesteFriedman(AvaliadorEstatistico):
    """Avaliador estatístico não-paramétrico independente usando teste de Friedman e pós-teste pareado."""

    __test__: bool = False


    @override
    def testar_diferencas(self, perdas_pareadas: pd.DataFrame, alpha: float = 0.05) -> ResultadoEstatistico:
        if perdas_pareadas.shape[1] < 2:
            raise ErroEstatistico("São necessários ao menos 2 modelos competidores para o teste comparativo.")
        if perdas_pareadas.shape[0] < 2:
            raise ErroEstatistico("São necessários ao menos 2 blocos/folds pareados para o teste de Friedman.")

        colunas = list(perdas_pareadas.columns)
        amostras = [perdas_pareadas[col].to_numpy() for col in colunas]

        try:
            if len(amostras) >= 3:
                resultado_friedman = stats.friedmanchisquare(*amostras)
                estatistica = float(resultado_friedman.statistic)
                valor_p = float(resultado_friedman.pvalue)
            else:
                resultado_wilcoxon = stats.wilcoxon(amostras[0], amostras[1])
                estatistica = float(resultado_wilcoxon.statistic)
                valor_p = float(resultado_wilcoxon.pvalue)
        except Exception as erro:
            raise ErroEstatistico(f"Erro ao calcular teste comparativo pareado: {erro}") from erro


        rejeitada = valor_p < alpha

        # Postos médios por linha (menor perda = melhor posto = 1)
        postos = perdas_pareadas.rank(axis=1, ascending=True)
        postos_medios: pd.Series[float] = postos.mean(axis=0)

        # Modelo campeão é o que obtiver menor posto médio (menor erro)
        modelo_campeao = str(postos_medios.idxmin())

        # Comparações pareadas pós-hoc
        n_modelos = len(colunas)
        n_blocos = len(perdas_pareadas)
        linhas_comparacao: list[dict[str, str | float | bool]] = []

        # Diferença crítica de Nemenyi aproximada: CD = q_alpha * sqrt(k*(k+1)/(6*N))
        # Para referência analítica nas comparações
        fator_cd = math.sqrt((n_modelos * (n_modelos + 1)) / (6.0 * n_blocos))

        for mod_a, mod_b in combinations(colunas, 2):
            diff_posto = abs(float(postos_medios[mod_a] - postos_medios[mod_b]))
            linhas_comparacao.append({
                "modelo_a": mod_a,
                "modelo_b": mod_b,
                "diff_postos": round(diff_posto, 4),
                "diferenca_significativa": diff_posto > (fator_cd * 1.96) if rejeitada else False
            })

        df_pos_hoc = pd.DataFrame(linhas_comparacao)

        return ResultadoEstatistico(
            estatistica_friedman=round(estatistica, 4),
            valor_p_friedman=round(valor_p, 6),
            hipotese_nula_rejeitada=rejeitada,
            postos_medios=postos_medios.round(4),
            comparacao_pos_hoc=df_pos_hoc,
            modelo_campeao=modelo_campeao
        )
