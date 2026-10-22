from __future__ import annotations
from dataclasses import dataclass
from typing import final
import pandas as pd
from app.configuracao_projeto.leitor_configuracao import ValorConfiguracao
from app.contratos_dados.avaliador_estatistico import ResultadoEstatistico
from app.contratos_dados.motor_treino import ModeloPreditor, MotorTreino, ValorMetricaParametro
from app.modelos_votacao.fabrica_votacao import FabricaVotacao


@dataclass(frozen=True)
class ResultadoOtimizacao:
    """Encapsula o modelo final treinado, nome de identificação e metadados de tuning/ensemble."""

    modelo_final: ModeloPreditor
    modelo_campeao_nome: str
    resultado_tuning: dict[str, ValorMetricaParametro]


@final
class OtimizadorCampeao:
    """Conduz a otimização de hiperparâmetros ou a montagem de ensemble ponderado/simples para o campeão."""

    def otimizar_ou_combinar(
        self,
        motor_treino: MotorTreino,
        df_dev: pd.DataFrame,
        dados_staging: pd.DataFrame,
        alvo: str,
        modelo_campeao_nome: str,
        config_tuning: dict[str, ValorConfiguracao],
        config_votacao: dict[str, ValorConfiguracao],
        resultado_estatistico: ResultadoEstatistico,
        perdas_pareadas: pd.DataFrame,
        cv_folds: int = 3,
    ) -> ResultadoOtimizacao:
        """Executa ensemble de votação ou tuning de hiperparâmetros conforme o plano de configuração YAML.

        Args:
            motor_treino: Motor de modelagem (Scikit-Learn ou PyCaret).
            df_dev: Conjunto de desenvolvimento para ajuste e busca de parâmetros.
            dados_staging: Base de dados consolidada para retreinamento final.
            alvo: Nome da variável dependente.
            modelo_campeao_nome: Identificador do modelo campeão eleito na avaliação.
            config_tuning: Parâmetros de tuning (iterações, método de busca).
            config_votacao: Parâmetros de ensemble (técnica, número de estimadores).
            resultado_estatistico: Resultado da inferência de Friedman com postos médios.
            perdas_pareadas: Tabela de perdas por fold para ponderação no ensemble.
            cv_folds: Número de folds da validação cruzada.

        Returns:
            Instância de ResultadoOtimizacao com estimador final pronto e metadados.
        """
        usar_votacao = bool(config_votacao.get("ativo", False))

        if usar_votacao:
            tecnica_votacao = str(config_votacao.get("tecnica", "media_simples"))
            melhores_n_val = config_votacao.get("melhores_n", 3)
            melhores_n = int(melhores_n_val) if isinstance(melhores_n_val, (int, float, str)) else 3

            # Ordena os modelos pelos postos médios de Friedman (menor posto = melhor desempenho)
            modelos_ordenados = list(resultado_estatistico.postos_medios.index)
            top_modelos_nomes = modelos_ordenados[:melhores_n]

            # Treina os N melhores modelos na base completa
            estimadores_top: list[tuple[str, ModeloPreditor]] = []
            for nome_m in top_modelos_nomes:
                est = motor_treino.treinar_final(dados_staging, alvo=alvo, modelo_nome=nome_m)
                estimadores_top.append((nome_m, est))

            metricas_rmse = perdas_pareadas.mean().to_dict()
            postos_friedman = resultado_estatistico.postos_medios.to_dict()

            modelo_final = FabricaVotacao.criar_ensemble_votacao(
                estimadores=estimadores_top,
                tecnica=tecnica_votacao,
                metricas_rmse=metricas_rmse,
                postos_friedman=postos_friedman,
                cv_folds=cv_folds,
            )
            nome_efetivo = f"ensemble_{tecnica_votacao}"
            resultado_tuning: dict[str, ValorMetricaParametro] = {
                "ensemble": tecnica_votacao,
                "modelos_componentes": top_modelos_nomes,
            }
            return ResultadoOtimizacao(
                modelo_final=modelo_final,
                modelo_campeao_nome=nome_efetivo,
                resultado_tuning=resultado_tuning,
            )

        # Caso ensemble esteja inativo: Tuning de Hiperparâmetros do Modelo Campeão Individual
        iteracoes_val = config_tuning.get("iteracoes", 5)
        iteracoes = int(iteracoes_val) if isinstance(iteracoes_val, (int, float, str)) else 5
        metodo_tuning = str(config_tuning.get("metodo", "random"))

        resultado_tuning = motor_treino.ajustar_parametros(
            df_dev,
            alvo=alvo,
            iteracoes=iteracoes,
            modelo_nome=modelo_campeao_nome,
            metodo_tuning=metodo_tuning,
        )

        # Retreino final na base completa para produção
        modelo_final = motor_treino.treinar_final(
            dados_staging, alvo=alvo, modelo_nome=modelo_campeao_nome
        )

        return ResultadoOtimizacao(
            modelo_final=modelo_final,
            modelo_campeao_nome=modelo_campeao_nome,
            resultado_tuning=resultado_tuning,
        )
