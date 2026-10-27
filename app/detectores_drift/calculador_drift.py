"""Implementação dos testes estatísticos de detecção de drift (KS, PSI, Degradação)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, cast, final, override
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # pyright: ignore[reportUnknownVariableType]

from app.contratos_drift.detector_drift import (
    DetectorDrift,
    ResultadoDriftVariavel,
    ResultadoRelatorioDrift,
)
from app.detectores_drift.avaliador_qualidade import AvaliadorQualidade
from app.enums_dominio import ClassificacaoDrift, MetodoEstatisticoDrift
from app.esquema_configuracao import ConfiguracaoDrift


class RetornoKs(Protocol):
    """Protocolo estrutural para a tupla de retorno do teste Kolmogorov-Smirnov."""

    statistic: float
    pvalue: float


@final
class CalculadorDrift(DetectorDrift):
    """Calculador desacoplado para testes de hipóteses de drift e degradação de modelo."""

    def __init__(self, config_drift: ConfiguracaoDrift) -> None:
        self._config = config_drift
        self._avaliador_qualidade = AvaliadorQualidade()

    @override
    def analisar_drift(
        self,
        dados_referencia: pd.DataFrame,
        dados_producao: pd.DataFrame,
        y_referencia: pd.Series | None = None,
        y_producao_real: pd.Series | None = None,
        predicoes_referencia: pd.Series | None = None,
        predicoes_producao: pd.Series | None = None,
        rmse_baseline: float | None = None,
    ) -> ResultadoRelatorioDrift:
        timestamp_atual = datetime.now(timezone.utc).isoformat()

        # 1. Avaliação de qualidade e completude
        qualidade = self._avaliador_qualidade.avaliar_qualidade(dados_referencia, dados_producao)

        limiar_pvalor = self._config.limiares.get("covariate_drift_pvalor", 0.05)
        limiar_ks_pred = self._config.limiares.get("prediction_drift_ks", 0.10)
        limiar_degrau_rmse = self._config.limiares.get("rmse_degradacao_maxima", 0.15)

        detalhes: list[ResultadoDriftVariavel] = []
        variaveis_com_drift: list[str] = []

        # 2. Avaliação de Data / Covariate Drift em cada feature preditora
        for col_obj in cast(list[object], list(dados_referencia.columns)):
            col = str(col_obj)
            if col not in dados_producao.columns:
                continue

            s_ref = cast(pd.Series, dados_referencia[col].dropna())
            s_prod = cast(pd.Series, dados_producao[col].dropna())

            if len(s_ref) < 2 or len(s_prod) < 2:
                continue

            if pd.api.types.is_numeric_dtype(s_ref):
                # Teste Kolmogorov-Smirnov para variáveis numéricas contínuas
                res_ks = cast(RetornoKs, cast(object, ks_2samp(s_ref, s_prod)))
                stat_ks = float(res_ks.statistic)
                p_val = float(res_ks.pvalue)
                tem_drift = p_val < limiar_pvalor

                if tem_drift:
                    variaveis_com_drift.append(col)

                detalhes.append(
                    ResultadoDriftVariavel(
                        nome_variavel=col,
                        tipo_variavel="numerica",
                        metodo_estatistico=MetodoEstatisticoDrift.KOLMOGOROV_SMIRNOV,
                        estatistica_teste=round(stat_ks, 4),
                        p_valor=round(p_val, 6),
                        limiar_alerta=limiar_pvalor,
                        drift_detectado=tem_drift,
                        distribuicao_referencia={
                            "media": round(float(s_ref.mean()), 2),
                            "desvio": round(float(s_ref.std()), 2),
                            "mediana": round(float(s_ref.median()), 2),
                        },
                        distribuicao_producao={
                            "media": round(float(s_prod.mean()), 2),
                            "desvio": round(float(s_prod.std()), 2),
                            "mediana": round(float(s_prod.median()), 2),
                        },
                    )
                )
            else:
                # Cálculo de PSI (Population Stability Index) para variáveis categóricas
                psi_valor = self._calcular_psi_categorico(s_ref, s_prod)
                tem_drift = psi_valor > 0.20
                if tem_drift:
                    variaveis_com_drift.append(col)

                lista_ref: list[str] = [str(x) for x in cast(list[object], s_ref.to_list())]
                lista_prod: list[str] = [str(x) for x in cast(list[object], s_prod.to_list())]
                n_cats_ref = float(len(set(lista_ref)))
                n_cats_prod = float(len(set(lista_prod)))

                detalhes.append(
                    ResultadoDriftVariavel(
                        nome_variavel=col,
                        tipo_variavel="categorica",
                        metodo_estatistico=MetodoEstatisticoDrift.PSI,
                        estatistica_teste=round(psi_valor, 4),
                        p_valor=None,
                        limiar_alerta=0.20,
                        drift_detectado=tem_drift,
                        distribuicao_referencia={"total_categorias": n_cats_ref},
                        distribuicao_producao={"total_categorias": n_cats_prod},
                    )
                )

        covariate_drift_detectado = len(variaveis_com_drift) > 0

        # 3. Prediction Drift P(Y_hat)
        prediction_drift_detectado = False
        if predicoes_referencia is not None and predicoes_producao is not None:
            if len(predicoes_referencia) >= 2 and len(predicoes_producao) >= 2:
                res_pred = cast(RetornoKs, cast(object, ks_2samp(predicoes_referencia, predicoes_producao)))
                stat_pred = float(res_pred.statistic)
                p_pred = float(res_pred.pvalue)
                prediction_drift_detectado = stat_pred > limiar_ks_pred and p_pred < limiar_pvalor

                detalhes.append(
                    ResultadoDriftVariavel(
                        nome_variavel="preco_predito",
                        tipo_variavel="predicao",
                        metodo_estatistico=MetodoEstatisticoDrift.KOLMOGOROV_SMIRNOV,
                        estatistica_teste=round(stat_pred, 4),
                        p_valor=round(p_pred, 6),
                        limiar_alerta=limiar_ks_pred,
                        drift_detectado=prediction_drift_detectado,
                        distribuicao_referencia={
                            "media": round(float(predicoes_referencia.mean()), 2),
                            "mediana": round(float(predicoes_referencia.median()), 2),
                        },
                        distribuicao_producao={
                            "media": round(float(predicoes_producao.mean()), 2),
                            "mediana": round(float(predicoes_producao.median()), 2),
                        },
                    )
                )

        # 4. Target / Label Drift P(Y)
        target_drift_detectado: bool | None = None
        if y_referencia is not None and y_producao_real is not None:
            if len(y_referencia) >= 2 and len(y_producao_real) >= 2:
                res_y = cast(RetornoKs, cast(object, ks_2samp(y_referencia, y_producao_real)))
                stat_y = float(res_y.statistic)
                p_y = float(res_y.pvalue)
                target_drift_detectado = p_y < limiar_pvalor

                detalhes.append(
                    ResultadoDriftVariavel(
                        nome_variavel="Valor_da_Venda",
                        tipo_variavel="alvo",
                        metodo_estatistico=MetodoEstatisticoDrift.KOLMOGOROV_SMIRNOV,
                        estatistica_teste=round(stat_y, 4),
                        p_valor=round(p_y, 6),
                        limiar_alerta=limiar_pvalor,
                        drift_detectado=target_drift_detectado,
                        distribuicao_referencia={
                            "media": round(float(y_referencia.mean()), 2),
                            "mediana": round(float(y_referencia.median()), 2),
                        },
                        distribuicao_producao={
                            "media": round(float(y_producao_real.mean()), 2),
                            "mediana": round(float(y_producao_real.median()), 2),
                        },
                    )
                )

        # 5. Concept Drift Real P(Y|X) (Monitoramento de Degradação de Desempenho)
        concept_drift_detectado: bool | None = None
        degradacao_rmse_pct: float | None = None

        if (
            y_producao_real is not None
            and predicoes_producao is not None
            and rmse_baseline is not None
            and rmse_baseline > 0.0
        ):
            erros_prod = np.asarray(y_producao_real, dtype=np.float64) - np.asarray(
                predicoes_producao, dtype=np.float64
            )
            rmse_producao = float(np.sqrt(np.mean(erros_prod**2)))
            degradacao_rmse_pct = float((rmse_producao - rmse_baseline) / rmse_baseline)
            concept_drift_detectado = degradacao_rmse_pct > limiar_degrau_rmse

        # 6. Classificação taxonômica rigorosa
        classificacao_drift = self._classificar_taxonomia(
            covariate_drift=covariate_drift_detectado,
            prediction_drift=prediction_drift_detectado,
            target_drift=target_drift_detectado,
            concept_drift=concept_drift_detectado,
        )

        recomendacao = self._gerar_recomendacao(
            classificacao=classificacao_drift,
            variaveis_com_drift=variaveis_com_drift,
            degradacao_pct=degradacao_rmse_pct,
        )

        return ResultadoRelatorioDrift(
            timestamp_analise=timestamp_atual,
            total_amostras_referencia=len(dados_referencia),
            total_amostras_producao=len(dados_producao),
            qualidade_dados=qualidade,
            covariate_drift_detectado=covariate_drift_detectado,
            prediction_drift_detectado=prediction_drift_detectado,
            target_drift_detectado=target_drift_detectado,
            concept_drift_detectado=concept_drift_detectado,
            classificacao_drift=classificacao_drift,
            degradacao_rmse_percentual=(
                round(degradacao_rmse_pct * 100.0, 2) if degradacao_rmse_pct is not None else None
            ),
            variaveis_com_drift=variaveis_com_drift,
            detalhes_variaveis=detalhes,
            recomendacao_acao=recomendacao,
        )

    def _calcular_psi_categorico(self, s_ref: pd.Series, s_prod: pd.Series) -> float:
        """Calcula o Population Stability Index para duas distribuições categóricas."""
        dict_ref = cast(dict[object, object], s_ref.value_counts(normalize=True).to_dict())
        dict_prod = cast(dict[object, object], s_prod.value_counts(normalize=True).to_dict())

        freq_ref: dict[str, float] = {str(k): float(str(v)) for k, v in dict_ref.items()}
        freq_prod: dict[str, float] = {str(k): float(str(v)) for k, v in dict_prod.items()}

        todas_categorias = set(freq_ref.keys()).union(set(freq_prod.keys()))
        epsilon = 1e-5
        psi = 0.0

        for cat in todas_categorias:
            p_ref = freq_ref.get(cat, 0.0) + epsilon
            p_prod = freq_prod.get(cat, 0.0) + epsilon
            psi += (p_prod - p_ref) * float(np.log(p_prod / p_ref))

        return float(psi)

    def _classificar_taxonomia(
        self,
        covariate_drift: bool,
        prediction_drift: bool,
        target_drift: bool | None,
        concept_drift: bool | None,
    ) -> ClassificacaoDrift:
        """Classifica rigorosamente o tipo de desvio sem assumir degradação apressada."""
        if concept_drift is True:
            return ClassificacaoDrift.REAL_CONCEPT_DRIFT

        if covariate_drift or prediction_drift:
            return ClassificacaoDrift.VIRTUAL_DRIFT

        if target_drift is True:
            return ClassificacaoDrift.LABEL_DRIFT

        return ClassificacaoDrift.SEM_DRIFT

    def _gerar_recomendacao(
        self,
        classificacao: ClassificacaoDrift | str,
        variaveis_com_drift: list[str],
        degradacao_pct: float | None,
    ) -> str:
        """Produz a orientação operacional prescritiva de engenharia."""
        if classificacao in (ClassificacaoDrift.REAL_CONCEPT_DRIFT, ClassificacaoDrift.REAL_CONCEPT_DRIFT.value):
            pct_txt = f"{degradacao_pct * 100.0:.1f}%" if degradacao_pct is not None else "> 15%"
            return (
                f"ALERTA CRÍTICO: Concept Drift Real detectado com degradação de erro RMSE de {pct_txt}. "
                "Recomenda-se acionar pipeline de retreinamento do modelo com dados recentes e reavaliação de hipóteses."
            )
        if classificacao in (ClassificacaoDrift.VIRTUAL_DRIFT, ClassificacaoDrift.VIRTUAL_DRIFT.value):
            vars_txt = ", ".join(variaveis_com_drift[:3])
            return (
                f"AVISO: Virtual Drift detectado em features ({vars_txt}) ou predições, porém sem degradação "
                "de performance comprovada. Monitorar de perto a evolução temporal sem necessidade imediata de retreino."
            )
        if classificacao in (ClassificacaoDrift.LABEL_DRIFT, ClassificacaoDrift.LABEL_DRIFT.value):
            return (
                "AVISO: Label Drift detectado na distribuição dos preços reais. "
                "Verificar mudanças macroeconômicas ou sazonalidade imobiliária em Ribeirão Preto."
            )
        return "STATUS VERDE: Nenhuma anomalia de distribuição ou degradação detectada. Modelo operando em regime nominal."
