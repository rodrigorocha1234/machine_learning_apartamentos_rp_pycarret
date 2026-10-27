"""Orquestrador do pipeline de ML utilizando o padrão de projeto Cadeia de Responsabilidade (Chain of Responsibility)."""

from __future__ import annotations

import argparse
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

from app.conf_yaml import ConfYAML
from app.enums_dominio import EtapaPipeline
from app.repositorios_dados.fabrica_repositorios import FabricaRepositorios
from app.servicos_avaliacao.orquestrador_avaliacao import OrquestradorAvaliacao, ResultadoEtapaAvaliacao
from app.servicos_coleta.coletor_dados import ColetorDados, ResultadoColeta
from app.servicos_eda.analisador_eda import AnalisadorEda, ResultadoEda
from app.servicos_implantacao.orquestrador_implantacao import OrquestradorImplantacao, ResultadoImplantacao
from app.servicos_modelagem.orquestrador_modelagem import OrquestradorModelagem, ResultadoModelagem
from app.servicos_observabilidade.orquestrador_drift import OrquestradorDrift, ResultadoRelatorioDrift
from app.servicos_preparacao.preparador_dados import PreparadorDados, ResultadoPreparacao


@dataclass
class ContextoPipeline:
    """Contexto compartilhado e transitado entre os elos da Cadeia de Responsabilidade."""

    conf: ConfYAML
    etapa_alvo: EtapaPipeline | str = EtapaPipeline.TODAS
    etapas_executadas: list[str] = field(default_factory=lambda: list[str]())
    tempos_etapas: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    resultado_coleta: ResultadoColeta | None = None
    resultado_preparacao: ResultadoPreparacao | None = None
    resultado_eda: ResultadoEda | None = None
    resultado_modelagem: ResultadoModelagem | None = None
    resultado_avaliacao: ResultadoEtapaAvaliacao | None = None
    resultado_implantacao: ResultadoImplantacao | None = None
    resultado_drift: ResultadoRelatorioDrift | None = None


class EloCadeiaPipeline(ABC):
    """Classe base abstrata para os manipuladores (elos) da Cadeia de Responsabilidade do pipeline."""

    def __init__(self) -> None:
        self._proximo: EloCadeiaPipeline | None = None

    def definir_proximo(self, proximo: EloCadeiaPipeline) -> EloCadeiaPipeline:
        """Configura o próximo elo na cadeia de responsabilidade e o retorna para encadeamento fluente."""
        self._proximo = proximo
        return proximo

    @property
    @abstractmethod
    def numero_etapa(self) -> EtapaPipeline | str:
        """Identificador numérico da etapa (1 a 7)."""
        ...

    @property
    @abstractmethod
    def titulo_etapa(self) -> str:
        """Título descritivo da etapa para exibição nos relatórios de console."""
        ...

    def processar(self, contexto: ContextoPipeline) -> ContextoPipeline:
        """Executa a etapa caso corresponda ao alvo ou à execução global e propaga na cadeia."""
        identificador_etapa = str(self.numero_etapa)
        alvo = str(contexto.etapa_alvo)
        deve_executar = alvo in (EtapaPipeline.TODAS, "todas", identificador_etapa)

        if deve_executar:
            self._imprimir_cabecalho()
            inicio = time.perf_counter()
            self._executar_etapa(contexto)
            duracao = time.perf_counter() - inicio
            contexto.tempos_etapas[identificador_etapa] = duracao
            contexto.etapas_executadas.append(identificador_etapa)

        # Se foi solicitada apenas uma etapa específica e ela já foi concluída, encerra a cadeia
        if alvo == identificador_etapa:
            return contexto

        # Propaga a execução para o próximo elo da cadeia
        if self._proximo is not None:
            return self._proximo.processar(contexto)

        return contexto

    def _imprimir_cabecalho(self) -> None:
        largura = 70
        print("\n" + "=" * largura)
        print(f" {self.titulo_etapa.center(largura - 2)} ")
        print("=" * largura)

    @abstractmethod
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        """Lógica de negócio específica de execução do elo."""
        ...


class EloColetaDados(EloCadeiaPipeline):
    """Elo responsável pela Etapa 1: Coleta de dados e persistência no banco de staging."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.COLETA

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 1: Coleta de Dados e Persistência no Staging"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        coletor = ColetorDados.da_configuracao(contexto.conf)
        resultado = coletor.executar_coleta()
        contexto.resultado_coleta = resultado
        print(f"✓ Registros coletados: {resultado.total_registros}")
        print(f"✓ Colunas: {resultado.total_colunas}")
        print(f"✓ Tabela destino: {resultado.tabela_destino}")
        print(f"✓ Duração: {resultado.tempo_execucao_segundos:.2f}s")


class EloPreparacaoDados(EloCadeiaPipeline):
    """Elo responsável pela Etapa 2: Preparação de dados e engenharia de features."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.PREPARACAO

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 2: Preparação de Dados e Engenharia de Features"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        preparador = PreparadorDados.da_configuracao(contexto.conf)
        repo = FabricaRepositorios.criar_repositorio(contexto.conf.obter_staging())
        dados_brutos = repo.ler_dados()
        df_limpo, resultado = preparador.preparar(dados_brutos)
        _ = df_limpo
        contexto.resultado_preparacao = resultado
        print(f"✓ Linhas removidas por filtros: {resultado.linhas_removidas_filtros}")
        print(f"✓ Colunas finais: {resultado.total_colunas_finais}")
        print(f"✓ Novos atributos criados: {len(resultado.atributos_novos_criados)}")
        print(f"✓ Duração: {resultado.tempo_execucao_segundos:.2f}s")


class EloAnaliseEda(EloCadeiaPipeline):
    """Elo responsável pela Etapa 3: Análise Exploratória de Dados (EDA)."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.EDA

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 3: Análise Exploratória de Dados (EDA)"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        analisador = AnalisadorEda.da_configuracao(contexto.conf)
        analises, res_vis, res_eda = analisador.executar_eda()
        _ = analises
        _ = res_vis
        contexto.resultado_eda = res_eda
        print(f"✓ Análises processadas: {res_eda.total_analises_processadas}")
        print(f"✓ Motor visual: {res_eda.motor_visualizacao}")
        print(f"✓ Artefatos gerados: {len(res_eda.arquivos_gerados)}")


class EloModelagemRegressao(EloCadeiaPipeline):
    """Elo responsável pela Etapa 4: Modelagem de regressão e tuning de hiperparâmetros."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.MODELAGEM

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 4: Modelagem de Regressão e Tuning de Hiperparâmetros"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        orquestrador = OrquestradorModelagem.da_configuracao(contexto.conf)
        res_tuning, res_modelagem = orquestrador.executar_modelagem()
        contexto.resultado_modelagem = res_modelagem
        print(f"✓ Motor: {res_modelagem.motor_utilizado}")
        print(f"✓ Total modelos treinados: {res_modelagem.total_modelos}")
        print(f"✓ Modelo campeão inicial: {res_modelagem.modelo_campeao}")
        print(f"✓ Melhor RMSE CV: R$ {res_tuning.melhor_rmse:,.2f}")
        print(f"✓ Run Pai MLflow: {res_modelagem.run_id_pai}")


class EloAvaliacaoSelecao(EloCadeiaPipeline):
    """Elo responsável pela Etapa 5: Validação cruzada externa e testes estatísticos."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.AVALIACAO

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 5: Validação Cruzada Externa e Testes Estatísticos"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        orquestrador = OrquestradorAvaliacao.da_configuracao(contexto.conf)
        resultado = orquestrador.executar_avaliacao_e_selecao()
        contexto.resultado_avaliacao = resultado
        print(f"✓ Amostras Treino / Holdout: {resultado.total_amostras_treino} / {resultado.total_amostras_holdout}")
        print(f"✓ Teste de Friedman p-valor: {resultado.comparacao_estatistica.friedman.p_valor:.6f}")
        print(f"✓ Diferença significativa: {resultado.comparacao_estatistica.friedman.diferenca_significativa}")
        print(f"✓ Modelo recomendado: {resultado.modelo_final_recomendado}")
        print(f"✓ Duração: {resultado.duracao_total_segundos:.2f}s")


class EloImplantacaoRegras(EloCadeiaPipeline):
    """Elo responsável pela Etapa 6: Implantação, promoção e regras de negócio."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.IMPLANTACAO

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 6: Implantação, Promoção e Regras de Negócio"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        orquestrador = OrquestradorImplantacao(contexto.conf)
        resultado = orquestrador.executar_implantacao()
        contexto.resultado_implantacao = resultado
        print(f"✓ Modelo promovido: {resultado.modelo_promovido} (Versão {resultado.versao_modelo})")
        print(f"✓ RMSE Holdout cego: R$ {resultado.rmse_holdout:,.2f}")
        print(f"✓ R² Holdout: {resultado.r2_holdout:.4f}")
        print(f"✓ Total regras ativas: {resultado.total_regras_ativas}")
        print(f"✓ Exemplo preço predito: R$ {resultado.exemplo_predicao.preco_predito:,.2f}")


class EloObservabilidadeDrift(EloCadeiaPipeline):
    """Elo responsável pela Etapa 7: Observabilidade, qualidade e detecção de drift."""

    @property
    @override
    def numero_etapa(self) -> EtapaPipeline:
        return EtapaPipeline.DRIFT

    @property
    @override
    def titulo_etapa(self) -> str:
        return "Etapa 7: Observabilidade, Qualidade e Detecção de Drift"

    @override
    def _executar_etapa(self, contexto: ContextoPipeline) -> None:
        orquestrador = OrquestradorDrift.da_configuracao(contexto.conf)
        resultado = orquestrador.executar_monitoramento()
        contexto.resultado_drift = resultado
        print(f"✓ Amostras Referência / Produção: {resultado.total_amostras_referencia} / {resultado.total_amostras_producao}")
        print(f"✓ Covariate drift detectado: {resultado.covariate_drift_detectado}")
        print(f"✓ Prediction drift detectado: {resultado.prediction_drift_detectado}")
        print(f"✓ Diagnóstico taxonômico: {resultado.classificacao_drift}")


class ConstrutorCadeiaPipeline:
    """Construtor da cadeia de responsabilidade para execução encadeada das 7 etapas."""

    @staticmethod
    def construir_cadeia_completa() -> EloCadeiaPipeline:
        """Monta a cadeia sequencial de 1 a 7 e retorna o elo inicial (cabeça da cadeia)."""
        elo1 = EloColetaDados()
        elo2 = EloPreparacaoDados()
        elo3 = EloAnaliseEda()
        elo4 = EloModelagemRegressao()
        elo5 = EloAvaliacaoSelecao()
        elo6 = EloImplantacaoRegras()
        elo7 = EloObservabilidadeDrift()

        (
            elo1.definir_proximo(elo2)
            .definir_proximo(elo3)
            .definir_proximo(elo4)
            .definir_proximo(elo5)
            .definir_proximo(elo6)
            .definir_proximo(elo7)
        )
        return elo1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disparador do Pipeline de Previsão de Imóveis (Padrão GoF Cadeia de Responsabilidade)"
    )
    opcoes_etapa = [
        EtapaPipeline.COLETA.value,
        EtapaPipeline.PREPARACAO.value,
        EtapaPipeline.EDA.value,
        EtapaPipeline.MODELAGEM.value,
        EtapaPipeline.AVALIACAO.value,
        EtapaPipeline.IMPLANTACAO.value,
        EtapaPipeline.DRIFT.value,
        EtapaPipeline.TODAS.value,
    ]
    parser.add_argument(
        "--etapa",
        choices=opcoes_etapa,
        default=EtapaPipeline.TODAS.value,
        help="Etapa individual a executar ou 'todas' para pipeline completo",
    )
    args = parser.parse_args()

    inicio_global = time.perf_counter()
    conf = ConfYAML()
    contexto = ContextoPipeline(conf=conf, etapa_alvo=args.etapa)

    if args.etapa == EtapaPipeline.TODAS.value:
        print("\n🚀 INICIANDO EXECUÇÃO COMPLETA DO PIPELINE (CADEIA DE RESPONSABILIDADE: ETAPAS 1 A 7)...")
    else:
        print(f"\n🎯 DISPARANDO CADEIA DE RESPONSABILIDADE PARA A ETAPA {args.etapa}...")

    cadeia = ConstrutorCadeiaPipeline.construir_cadeia_completa()
    contexto_final = cadeia.processar(contexto)

    duracao_total = time.perf_counter() - inicio_global
    largura = 70
    print("\n" + "=" * largura)
    print(f" PIPELINE CONCLUÍDO COM SUCESSO EM {duracao_total:.2f} SEGUNDOS ".center(largura))
    print(f" Etapas executadas: {', '.join(contexto_final.etapas_executadas)} ".center(largura))
    print("=" * largura)


if __name__ == "__main__":
    main()
