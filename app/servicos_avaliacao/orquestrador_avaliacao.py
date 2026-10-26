"""Orquestrador da Etapa 5: Avaliação Externa, Estatística (Friedman/Nemenyi) e Seleção."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import cast, final
import pandas as pd
from pydantic import BaseModel, Field

from app.comites_votacao.comite_sklearn import ComiteSklearn
from app.conf_yaml import ConfYAML
from app.contratos_avaliacao.avaliador_estatistico import ResultadoComparacaoEstatistica
from app.contratos_avaliacao.construtor_comite import ResultadoComite
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.contratos_modelagem.motor_treinador import ResultadoTreinoModelo, ResultadoTuning
from app.esquema_configuracao import ConfiguracaoAvaliacao, ConfiguracaoModelagem
from app.repositorios_dados.fabrica_repositorios import FabricaRepositorios
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_avaliacao.avaliador_holdout import AvaliadorHoldout, MetricasHoldout
from app.servicos_avaliacao.testador_friedman import TestadorFriedman
from app.servicos_avaliacao.validador_cruzado import ValidadorCruzado
from app.servicos_modelagem.rastreador_mlflow import RastreadorMlflow


class ResultadoEtapaAvaliacao(BaseModel):
    """Resultados completos e auditáveis da Etapa 5."""

    total_amostras_treino: int = Field(description="Registros utilizados no treino e CV pareada")
    total_amostras_holdout: int = Field(description="Registros preservados para teste cego final")
    comparacao_estatistica: ResultadoComparacaoEstatistica = Field(
        description="Evidência dos testes de Friedman e Nemenyi"
    )
    metricas_cv_pareada: dict[str, list[float]] = Field(
        description="RMSE por dobra de validação cruzada para cada modelo"
    )
    metricas_holdout: dict[str, MetricasHoldout] = Field(
        description="Métricas individuais no conjunto holdout isolado"
    )
    resultado_comite: ResultadoComite = Field(
        description="Resultado da composição do ensemble/votação"
    )
    modelo_final_recomendado: str = Field(
        description="Modelo campeão ou ensemble selecionado para implantação"
    )
    duracao_total_segundos: float = Field(ge=0.0, description="Tempo decorrido na etapa")


class ErroOrquestradorAvaliacao(Exception):
    """Exceção levantada em falhas na orquestração da avaliação e seleção."""


@final
class OrquestradorAvaliacao:
    """Coordenador da avaliação estatística, testes não-paramétricos e ensembles."""

    def __init__(
        self,
        config_avaliacao: ConfiguracaoAvaliacao,
        config_modelagem: ConfiguracaoModelagem,
        coluna_alvo: str = "Valor_da_Venda",
        repositorio_staging: RepositorioDados | None = None,
        rastreador_mlflow: RastreadorMlflow | None = None,
    ) -> None:
        self._config_av = config_avaliacao
        self._config_mod = config_modelagem
        self._coluna_alvo = coluna_alvo
        self._repositorio = repositorio_staging
        self._rastreador_mlflow = rastreador_mlflow
        self._validador_cruzado = ValidadorCruzado(
            config_avaliacao,
            config_modelagem,
            coluna_alvo,
            rastreador_mlflow=rastreador_mlflow,
        )
        self._testador_estatistico = TestadorFriedman()
        self._construtor_comite = ComiteSklearn()
        self._avaliador_holdout = AvaliadorHoldout(coluna_alvo)

    @classmethod
    def da_configuracao(cls, conf: ConfYAML) -> OrquestradorAvaliacao:
        """Instancia o OrquestradorAvaliacao a partir do ConfYAML unificado."""
        config_av = conf.obter_avaliacao()
        config_mod = conf.obter_modelagem()
        repositorio = FabricaRepositorios.criar_repositorio(conf.obter_staging())
        alvo = conf.obter_projeto().alvo
        rastreador = RastreadorMlflow(config_mod.mlflow)
        return cls(
            config_avaliacao=config_av,
            config_modelagem=config_mod,
            coluna_alvo=alvo,
            repositorio_staging=repositorio,
            rastreador_mlflow=rastreador,
        )

    def executar_avaliacao_e_selecao(
        self,
        dados: pd.DataFrame | None = None,
        resultado_tuning: ResultadoTuning | None = None,
    ) -> ResultadoEtapaAvaliacao:
        """Executa CV pareada, testes de Friedman/Nemenyi, ensemble e avaliação no holdout."""
        inicio = time.perf_counter()

        # 1. Carregamento dos dados
        df_base = self._obter_dados_entrada(dados)
        if df_base.empty:
            raise ErroOrquestradorAvaliacao("Dados de entrada para avaliação estão vazios.")
        if self._coluna_alvo not in df_base.columns:
            raise ErroOrquestradorAvaliacao(f"Coluna alvo '{self._coluna_alvo}' ausente na base.")

        # 2. Separação idêntica de treino e holdout preservado (80% / 20%)
        frac_holdout = self._config_av.validacao.holdout
        n_total = len(df_base)
        n_holdout = int(n_total * frac_holdout)

        df_embaralhado = df_base.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_holdout = cast(pd.DataFrame, df_embaralhado.iloc[:n_holdout]).copy().reset_index(drop=True)
        df_treino = cast(pd.DataFrame, df_embaralhado.iloc[n_holdout:]).copy().reset_index(drop=True)

        # 3. Obtenção dos modelos ajustados na Etapa 4
        modelos_tuning = self._obter_modelos_tuning(resultado_tuning)

        # 4. Validação Cruzada Pareada nos mesmos splits de treino
        matriz_cv, estimadores_treinados = self._validador_cruzado.executar_cv_pareada(
            dados_treino=df_treino,
            modelos_tuning=modelos_tuning,
        )

        # 5. Aplicação dos Testes Estatísticos Não-Paramétricos (Friedman e Nemenyi)
        cfg_estat = self._config_av.estatistica
        res_estatistica = self._testador_estatistico.comparar_modelos(
            matriz_cv=matriz_cv,
            metrica=self._config_av.validacao.metrica,
            alpha=cfg_estat.alpha,
        )

        # 6. Avaliação de todos os modelos no holdout isolado
        metricas_holdout = self._avaliador_holdout.avaliar_modelos_holdout(
            modelos_treinados=estimadores_treinados,
            dados_holdout=df_holdout,
        )

        # 7. Construção e Avaliação de Comitê de Votação (Ensemble)
        # Filtra estimadores dos candidatos selecionados pela significância estatística
        estimadores_candidatos: dict[str, object] = {
            m: estimadores_treinados[m]
            for m in res_estatistica.candidatos_selecionados
            if m in estimadores_treinados
        }
        res_comite = self._construtor_comite.construir_e_avaliar(
            modelos_treinados=estimadores_candidatos,
            dados_treino=df_treino,
            dados_holdout=df_holdout,
            coluna_alvo=self._coluna_alvo,
            config_votacao=self._config_av.votacao,
        )

        # 8. Definição do modelo final recomendado
        if res_comite.ativo:
            modelo_final = f"ensemble_{res_comite.tecnica_utilizada}"
        else:
            modelo_final = res_estatistica.modelo_campeao_estatistico

        duracao = time.perf_counter() - inicio

        resultado_etapa = ResultadoEtapaAvaliacao(
            total_amostras_treino=len(df_treino),
            total_amostras_holdout=len(df_holdout),
            comparacao_estatistica=res_estatistica,
            metricas_cv_pareada=matriz_cv,
            metricas_holdout=metricas_holdout,
            resultado_comite=res_comite,
            modelo_final_recomendado=modelo_final,
            duracao_total_segundos=round(duracao, 4),
        )

        # Persistência de evidências
        self._persistir_evidencias(resultado_etapa)

        return resultado_etapa

    def _obter_modelos_tuning(
        self, resultado_tuning: ResultadoTuning | None
    ) -> list[ResultadoTreinoModelo]:
        if resultado_tuning is not None and resultado_tuning.modelos_treinados:
            return resultado_tuning.modelos_treinados

        caminho_json = Path("storage-data/relatorio_modelagem.json")
        if caminho_json.exists():
            try:
                with open(caminho_json, "r", encoding="utf-8") as f:
                    conteudo = json.load(f)
                dict_tuning = cast(dict[str, object], conteudo.get("res_tuning", {}))
                res_obj = ResultadoTuning.model_validate(dict_tuning)
                if res_obj.modelos_treinados:
                    return res_obj.modelos_treinados
            except Exception as erro_json:
                raise ErroOrquestradorAvaliacao(
                    f"Falha ao ler relatorio_modelagem.json: {erro_json}"
                ) from erro_json

        raise ErroOrquestradorAvaliacao(
            "Nenhum resultado de tuning da Etapa 4 foi fornecido e 'storage-data/relatorio_modelagem.json' não foi encontrado."
        )

    def _obter_dados_entrada(self, dados: pd.DataFrame | None) -> pd.DataFrame:
        if dados is not None:
            return dados
        if self._repositorio is None:
            raise ErroOrquestradorAvaliacao("Nenhum repositório de staging configurado.")

        if isinstance(self._repositorio, RepositorioSqlite):
            repo_limpos = RepositorioSqlite(
                caminho=self._repositorio.caminho,
                tabela="imoveis_limpos",
            )
            if repo_limpos.verificar_existencia():
                dados_limpos = repo_limpos.ler_dados()
                if not dados_limpos.empty:
                    return dados_limpos

        return self._repositorio.ler_dados()

    def _persistir_evidencias(self, resultado: ResultadoEtapaAvaliacao) -> None:
        caminho_saida = Path("storage-data/relatorio_avaliacao.json")
        try:
            with open(caminho_saida, "w", encoding="utf-8") as f:
                json.dump(resultado.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
