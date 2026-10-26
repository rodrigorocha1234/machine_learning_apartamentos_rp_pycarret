"""Serviço orquestrador para a Etapa 4: Modelagem de Regressão e Tuning."""

from __future__ import annotations

import time
from typing import cast, final
import pandas as pd

from app.conf_yaml import ConfYAML
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.contratos_modelagem.motor_treinador import (
    MotorTreinador,
    ResultadoModelagem,
    ResultadoTuning,
)
from app.esquema_configuracao import ConfiguracaoModelagem
from app.motores_modelagem.fabrica_treinadores import FabricaTreinadores
from app.repositorios_dados.fabrica_repositorios import FabricaRepositorios
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_modelagem.rastreador_mlflow import RastreadorMlflow


class ErroOrquestradorModelagem(Exception):
    """Exceção levantada em falhas na orquestração da modelagem."""


@final
class OrquestradorModelagem:
    """Coordenador desacoplado da modelagem, tuning e governança no MLflow."""

    def __init__(
        self,
        treinador: MotorTreinador,
        rastreador: RastreadorMlflow,
        config_modelagem: ConfiguracaoModelagem,
        coluna_alvo: str = "Valor_da_Venda",
        fracao_holdout: float = 0.2,
        repositorio_staging: RepositorioDados | None = None,
    ) -> None:
        self._treinador = treinador
        self._rastreador = rastreador
        self._config = config_modelagem
        self._coluna_alvo = coluna_alvo
        self._fracao_holdout = fracao_holdout
        self._repositorio = repositorio_staging

    @classmethod
    def da_configuracao(
        cls,
        conf: ConfYAML,
        motor_sobrescrito: MotorTreinador | None = None,
    ) -> OrquestradorModelagem:
        """Instancia o OrquestradorModelagem a partir do ConfYAML unificado."""
        config_modelagem = conf.obter_modelagem()
        treinador = (
            motor_sobrescrito
            if motor_sobrescrito is not None
            else FabricaTreinadores.criar_treinador(config_modelagem)
        )
        rastreador = RastreadorMlflow(config_modelagem.mlflow)
        repositorio = FabricaRepositorios.criar_repositorio(conf.obter_staging())
        fracao_holdout = conf.obter_avaliacao().validacao.holdout

        return cls(
            treinador=treinador,
            rastreador=rastreador,
            config_modelagem=config_modelagem,
            coluna_alvo=conf.obter_projeto().alvo,
            fracao_holdout=fracao_holdout,
            repositorio_staging=repositorio,
        )

    def executar_modelagem(
        self, dados: pd.DataFrame | None = None
    ) -> tuple[ResultadoTuning, ResultadoModelagem]:
        """Executa particionamento anti-leakage, tuning de modelos e registro hierárquico no MLflow."""
        inicio = time.perf_counter()

        # 1. Carregamento dos dados tabulares preparados
        df_base = self._obter_dados_entrada(dados)
        if df_base.empty:
            raise ErroOrquestradorModelagem("Dados de entrada para modelagem estão vazios.")
        if self._coluna_alvo not in df_base.columns:
            raise ErroOrquestradorModelagem(f"Coluna alvo '{self._coluna_alvo}' ausente na base.")

        # 2. Separação de treino e holdout estrito para blindagem contra vazamento
        n_total = len(df_base)
        n_holdout = int(n_total * self._fracao_holdout)
        df_embaralhado = df_base.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_treino_cast = cast(pd.DataFrame, df_embaralhado.iloc[n_holdout:]).copy().reset_index(drop=True)

        # 3. Inicialização do run pai no MLflow: tuning_de_parametros_{metodo}
        metodo_tuning = self._config.tuning.metodo
        nome_motor = self._treinador.obter_nome_motor()
        id_run_pai = self._rastreador.iniciar_run_pai(metodo_tuning=metodo_tuning, motor=nome_motor)

        # 4. Execução do treinamento e tuning de hiperparâmetros
        try:
            resultado_tuning = self._treinador.treinar_e_ajustar(
                dados_treino=df_treino_cast,
                coluna_alvo=self._coluna_alvo,
                config=self._config,
            )
        except Exception as erro_treino:
            raise ErroOrquestradorModelagem(
                f"Falha no motor de modelagem '{nome_motor}': {erro_treino}"
            ) from erro_treino

        # 5. Registro dos runs filhos aninhados no MLflow
        tecnica_prep = (
            self._config.preprocessamento.metodo_transformacao_alvo
            if self._config.preprocessamento.transformar_alvo
            else "padrao"
        )
        runs_filhos: dict[str, str] = {}

        for modelo_res in resultado_tuning.modelos_treinados:
            try:
                id_filho = self._rastreador.registrar_run_filho(
                    resultado_modelo=modelo_res,
                    tecnica_preprocessamento=tecnica_prep,
                    id_run_pai=id_run_pai,
                )
                runs_filhos[modelo_res.nome_modelo] = id_filho
            except Exception as erro_filho:
                raise ErroOrquestradorModelagem(
                    f"Falha ao registrar run filho no MLflow para '{modelo_res.nome_modelo}': {erro_filho}"
                ) from erro_filho

        # 6. Finalização do run pai
        self._rastreador.finalizar_run_pai(resultado_tuning, id_run_pai)

        duracao = time.perf_counter() - inicio

        resultado_modelagem = ResultadoModelagem(
            motor_utilizado=nome_motor,
            total_modelos=len(resultado_tuning.modelos_treinados),
            experimento_mlflow=self._config.mlflow.experimento,
            run_id_pai=id_run_pai,
            runs_filhos=runs_filhos,
            modelo_campeao=resultado_tuning.modelo_campeao,
            duracao_total_segundos=round(duracao, 4),
        )

        return resultado_tuning, resultado_modelagem

    def _obter_dados_entrada(self, dados: pd.DataFrame | None) -> pd.DataFrame:
        if dados is not None:
            return dados

        if self._repositorio is None:
            raise ErroOrquestradorModelagem(
                "Nenhum repositório de staging configurado e nenhum DataFrame fornecido."
            )

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
