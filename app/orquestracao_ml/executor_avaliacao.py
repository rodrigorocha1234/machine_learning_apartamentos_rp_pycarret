from __future__ import annotations
from dataclasses import dataclass
from typing import final
import pandas as pd
from app.analises_estatisticas.teste_friedman import TesteFriedman
from app.configuracao_projeto.leitor_configuracao import ValorConfiguracao
from app.contratos_dados.avaliador_estatistico import ResultadoEstatistico
from app.contratos_dados.motor_treino import MotorTreino


@dataclass(frozen=True)
class ResultadoAvaliacao:
    """Encapsula o resultado da validação cruzada pareada e inferência estatística."""

    perdas_pareadas: pd.DataFrame
    resultado_estatistico: ResultadoEstatistico
    modelo_campeao: str
    resumo_estatistico: dict[str, str | int | float | bool | dict[str, float]]


@final
class ExecutorAvaliacao:
    """Executa a validação cruzada pareada e a análise estatística independente (Friedman / Nemenyi)."""

    def __init__(self, avaliador_estatistico: TesteFriedman | None = None) -> None:
        self._avaliador_estatistico = avaliador_estatistico or TesteFriedman()

    def executar_avaliacao(
        self,
        motor_treino: MotorTreino,
        df_dev: pd.DataFrame,
        alvo: str,
        config_estatistica: dict[str, ValorConfiguracao] | None = None,
    ) -> ResultadoAvaliacao:
        """Executa a validação cruzada pareada de todos os modelos ativos e testa a significância estatística.

        Args:
            motor_treino: Instância do motor de treinamento (Scikit-Learn ou PyCaret).
            df_dev: DataFrame de desenvolvimento (treino/validação) segregado do holdout.
            alvo: Nome da variável dependente alvo.
            config_estatistica: Dicionário com parâmetros estatísticos (como alpha de significância).

        Returns:
            Instância de ResultadoAvaliacao com métricas por fold e campeão eleito.
        """
        # 1. Validação Cruzada Pareada de Todos os Modelos Ativos
        perdas_pareadas = motor_treino.avaliar_modelos(df_dev, alvo=alvo)

        # 2. Camada Estatística Independente (Friedman e Pós-Hoc Pareado)
        cfg_est = config_estatistica or {}
        alpha_val = cfg_est.get("alpha", 0.05)
        alpha = float(alpha_val) if isinstance(alpha_val, (int, float, str)) else 0.05

        resultado_estatistico = self._avaliador_estatistico.testar_diferencas(perdas_pareadas, alpha=alpha)
        modelo_campeao_nome = resultado_estatistico.modelo_campeao

        resumo_dict: dict[str, str | int | float | bool | dict[str, float]] = {
            "estatistica_friedman": resultado_estatistico.estatistica_friedman,
            "valor_p_friedman": resultado_estatistico.valor_p_friedman,
            "rejeitou_nula": resultado_estatistico.hipotese_nula_rejeitada,
            "postos_medios": resultado_estatistico.postos_medios.to_dict(),
            "modelo_campeao": modelo_campeao_nome,
        }

        return ResultadoAvaliacao(
            perdas_pareadas=perdas_pareadas,
            resultado_estatistico=resultado_estatistico,
            modelo_campeao=modelo_campeao_nome,
            resumo_estatistico=resumo_dict,
        )
