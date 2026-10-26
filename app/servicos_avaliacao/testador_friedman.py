"""Implementação do teste não-paramétrico de Friedman e pós-teste de Nemenyi."""

from __future__ import annotations

import math
from typing import cast, final, override
import numpy as np
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

from app.contratos_avaliacao.avaliador_estatistico import (
    AvaliadorEstatistico,
    ComparacaoPar,
    ResultadoComparacaoEstatistica,
    ResultadoFriedman,
    ResultadoNemenyi,
)


class ErroTestadorFriedman(Exception):
    """Exceção levantada em falhas durante a execução dos testes estatísticos."""


@final
class TestadorFriedman(AvaliadorEstatistico):
    """Avaliador estatístico não-paramétrico de Friedman e pós-teste de Nemenyi."""

    @override
    def comparar_modelos(
        self,
        matriz_cv: dict[str, list[float]],
        metrica: str = "RMSE",
        alpha: float = 0.05,
    ) -> ResultadoComparacaoEstatistica:
        """Executa os testes de Friedman e Nemenyi sobre as medições pareadas de CV."""
        nomes_modelos = list(matriz_cv.keys())
        k = len(nomes_modelos)
        if k < 2:
            raise ErroTestadorFriedman("São necessários pelo menos 2 modelos para teste comparativo.")

        tamanhos = [len(v) for v in matriz_cv.values()]
        if len(set(tamanhos)) > 1:
            raise ErroTestadorFriedman("Todas as séries de validação cruzada devem possuir o mesmo número de dobras pareadas.")

        n_amostras = tamanhos[0]
        if n_amostras < 2:
            raise ErroTestadorFriedman("São necessárias pelo menos 2 dobras pareadas para teste estatístico.")

        # Constrói matriz (N x k): linhas = folds, colunas = modelos
        matriz_valores = np.zeros((n_amostras, k), dtype=np.float64)
        for idx_col, nome in enumerate(nomes_modelos):
            matriz_valores[:, idx_col] = np.asarray(matriz_cv[nome], dtype=np.float64)

        # 1. Cálculo dos ranks médios por linha (menor RMSE = melhor rank 1)
        ranks = np.zeros_like(matriz_valores)
        for i in range(n_amostras):
            linha = matriz_valores[i, :]
            # scipy.stats.rankdata atribui rank 1 ao menor valor
            ranks_linha = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.float64]],
                stats.rankdata(linha, method="average"),
            )
            ranks[i, :] = ranks_linha

        vetor_ranks_medios = cast(
            np.ndarray[tuple[int, ...], np.dtype[np.float64]],
            np.mean(ranks, axis=0),
        )
        ranks_dict: dict[str, float] = {
            nome: round(float(vetor_ranks_medios[idx]), 4)
            for idx, nome in enumerate(nomes_modelos)
        }

        # 2. Teste Global de Friedman
        listas_colunas = [matriz_valores[:, col] for col in range(k)]
        if k >= 3:
            try:
                res_friedman_scipy = stats.friedmanchisquare(*listas_colunas)
                stat_chi2 = float(str(cast(object, res_friedman_scipy.statistic)))
                p_valor_friedman = float(str(cast(object, res_friedman_scipy.pvalue)))
            except Exception as erro_scipy:
                raise ErroTestadorFriedman(f"Falha ao executar friedmanchisquare: {erro_scipy}") from erro_scipy
        else:
            soma_r2 = float(np.sum(vetor_ranks_medios**2))
            termo_sub = (k * ((k + 1) ** 2)) / 4.0
            stat_chi2 = float(((12.0 * n_amostras) / (k * (k + 1))) * (soma_r2 - termo_sub))
            stat_chi2 = max(stat_chi2, 0.0)
            p_valor_friedman = float(str(cast(object, stats.chi2.sf(stat_chi2, df=1))))

        graus_liberdade = k - 1
        diferenca_significativa = p_valor_friedman < alpha

        resultado_friedman = ResultadoFriedman(
            estatistica_chi2=round(stat_chi2, 4),
            p_valor=round(p_valor_friedman, 6),
            graus_liberdade=graus_liberdade,
            diferenca_significativa=diferenca_significativa,
            hipotese_nula_rejeitada=diferenca_significativa,
            ranks_medios=ranks_dict,
        )

        # 3. Pós-teste de Nemenyi se houver diferença significativa
        resultado_nemenyi: ResultadoNemenyi | None = None
        # Encontra o modelo com o menor rank médio
        campeao_nome = min(ranks_dict, key=lambda m: ranks_dict[m])
        equivalentes: list[str] = [campeao_nome]
        comparacoes: list[ComparacaoPar] = []

        # Fator de escala da variância de ranks
        # SE = sqrt(k * (k + 1) / (6 * N))
        se_rank = math.sqrt((k * (k + 1)) / (6.0 * n_amostras))

        # Quantil crítico da distribuição Studentized Range: q_alpha / sqrt(2)
        try:
            # studentized_range.ppf(1 - alpha, k, df=inf)
            q_critico_sr = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf))  # pyright: ignore[reportUnknownMemberType]
            q_alpha = q_critico_sr / math.sqrt(2.0)
            diferenca_critica = float(q_alpha * se_rank)
        except Exception:
            # Fallback assintótico padrão de Nemenyi
            diferenca_critica = float(2.576 * se_rank)

        for i in range(k):
            for j in range(i + 1, k):
                m_a = nomes_modelos[i]
                m_b = nomes_modelos[j]
                diff_r = abs(ranks_dict[m_a] - ranks_dict[m_b])
                # estatística q de Studentized Range
                q_est = (diff_r / se_rank) * math.sqrt(2.0)
                try:
                    p_val_par = float(stats.studentized_range.sf(q_est, k, np.inf))  # pyright: ignore[reportUnknownMemberType]
                except Exception:
                    p_val_par = 1.0 if diff_r < diferenca_critica else 0.01

                eh_significativo = diff_r >= diferenca_critica
                comparacoes.append(
                    ComparacaoPar(
                        modelo_a=m_a,
                        modelo_b=m_b,
                        diferenca_rank=round(diff_r, 4),
                        estatistica_q=round(q_est, 4),
                        p_valor=round(p_val_par, 6),
                        significativo=eh_significativo,
                    )
                )

        # Identifica modelos estatisticamente indistinguíveis do campeão
        for nome, rank_val in ranks_dict.items():
            if nome != campeao_nome:
                diff_campeao = abs(rank_val - ranks_dict[campeao_nome])
                if diff_campeao < diferenca_critica:
                    equivalentes.append(nome)

        resultado_nemenyi = ResultadoNemenyi(
            diferenca_critica=round(diferenca_critica, 4),
            comparacoes=comparacoes,
            modelos_equivalentes_ao_campeao=equivalentes,
        )

        # 4. Redação da justificativa fundamentada
        if diferenca_significativa:
            justificativa = (
                f"O teste de Friedman rejeitou H0 (chi2={stat_chi2:.2f}, p={p_valor_friedman:.4e} < {alpha}), "
                f"confirmando diferença estatisticamente significativa entre os modelos. "
                f"O modelo '{campeao_nome}' obteve o melhor rank médio ({ranks_dict[campeao_nome]:.2f}). "
                f"Com Diferença Crítica (CD) de {diferenca_critica:.2f}, os seguintes modelos são estatisticamente "
                f"equivalentes ao campeão: {', '.join(equivalentes)}."
            )
            candidatos = equivalentes
        else:
            justificativa = (
                f"O teste de Friedman NÃO rejeitou H0 (chi2={stat_chi2:.2f}, p={p_valor_friedman:.4f} >= {alpha}). "
                f"Portanto, os dados não fornecem evidência estatística suficiente para distinguir a superioridade "
                f"de qualquer modelo sobre os demais ao nível de significância de {alpha:.2f}. "
                f"O modelo '{campeao_nome}' foi mantido com base no menor erro empírico nominal."
            )
            candidatos = nomes_modelos

        return ResultadoComparacaoEstatistica(
            metrica_avaliada=metrica,
            total_amostras_folds=n_amostras,
            friedman=resultado_friedman,
            nemenyi=resultado_nemenyi,
            modelo_campeao_estatistico=campeao_nome,
            candidatos_selecionados=candidatos,
            justificativa_decisao=justificativa,
        )
