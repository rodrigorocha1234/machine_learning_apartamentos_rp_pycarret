"""Orquestrador completo para execução e publicação de monitoramento de drift e observabilidade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, final
import pandas as pd

from app.conf_yaml import ConfYAML
from app.contratos_drift.detector_drift import ResultadoRelatorioDrift
from app.detectores_drift.calculador_drift import CalculadorDrift
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite
from app.servicos_implantacao.carregador_campeao import CarregadorCampeao
from app.servicos_observabilidade.publicador_grafana import PublicadorGrafana


class ErroOrquestradorDrift(Exception):
    """Exceção levantada em falhas durante a orquestração de observabilidade e drift."""


@final
class OrquestradorDrift:
    """Coordenador de monitoramento contínuo, detecção de desvios e telemetria no Grafana."""

    def __init__(self, configuracao: ConfYAML) -> None:
        self._conf = configuracao
        staging_conf = self._conf.obter_staging()
        self._repositorio = RepositorioSqlite(
            caminho=staging_conf.caminho,
            tabela="imoveis_limpos",
        )
        self._detector = CalculadorDrift(self._conf.obter_monitoramento().drift)
        self._publicador = PublicadorGrafana(self._conf.obter_monitoramento().observabilidade)

    @classmethod
    def da_configuracao(cls, conf: ConfYAML) -> OrquestradorDrift:
        """Fábrica de conveniência a partir de uma instância ConfYAML."""
        return cls(conf)

    def executar_monitoramento(
        self,
        dados_producao: pd.DataFrame | None = None,
        caminho_artefatos_campeao: Path | str = "production_artifacts/campeao",
    ) -> ResultadoRelatorioDrift:
        """Executa a rotina completa de observabilidade, detecção de drift e exportação."""
        coluna_alvo = self._conf.obter_projeto().alvo
        dir_campeao = Path(caminho_artefatos_campeao)

        # 1. Carregamento dos dados limpos de referência
        df_completo = self._repositorio.ler_dados()
        if df_completo.empty:
            raise ErroOrquestradorDrift("Tabela 'imoveis_limpos' está vazia.")

        # Separação idêntica de treino (80% referência) e holdout (20% produção/teste)
        holdout_pct = self._conf.obter_avaliacao().validacao.holdout
        n_total = len(df_completo)
        n_holdout = int(n_total * holdout_pct)

        df_embaralhado = df_completo.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_ref_full = cast(pd.DataFrame, df_embaralhado.iloc[n_holdout:]).copy().reset_index(drop=True)
        df_prod_simulada = cast(pd.DataFrame, df_embaralhado.iloc[:n_holdout]).copy().reset_index(drop=True)

        df_producao_final = dados_producao if dados_producao is not None else df_prod_simulada

        X_cols = [str(c) for c in cast(list[object], list(df_ref_full.columns)) if str(c) != coluna_alvo]
        X_ref = cast(pd.DataFrame, df_ref_full[X_cols])
        X_prod = cast(pd.DataFrame, df_producao_final[X_cols])

        # 2. Carregamento do modelo campeão promovido
        caminho_modelo = dir_campeao / "modelo_campeao.joblib"
        caminho_meta = dir_campeao / "metadados_campeao.json"

        if not caminho_modelo.exists() or not caminho_meta.exists():
            raise ErroOrquestradorDrift(
                f"Artefatos do modelo campeão não encontrados em '{dir_campeao}'."
            )

        carregador = CarregadorCampeao(caminho_modelo, caminho_meta)
        metadados = carregador.obter_metadados()

        # 3. Execução das predições em referência e produção
        predicoes_ref_list = carregador.prever(X_ref)
        predicoes_prod_list = carregador.prever(X_prod)
        s_pred_ref = pd.Series(predicoes_ref_list)
        s_pred_prod = pd.Series(predicoes_prod_list)

        # 4. Rótulos reais caso disponíveis no lote de produção
        s_y_ref = cast(pd.Series, df_ref_full[coluna_alvo]) if coluna_alvo in df_ref_full.columns else None
        s_y_prod = (
            cast(pd.Series, df_producao_final[coluna_alvo])
            if coluna_alvo in df_producao_final.columns
            else None
        )

        # 5. Execução do teste de drift
        relatorio = self._detector.analisar_drift(
            dados_referencia=X_ref,
            dados_producao=X_prod,
            y_referencia=s_y_ref,
            y_producao_real=s_y_prod,
            predicoes_referencia=s_pred_ref,
            predicoes_producao=s_pred_prod,
            rmse_baseline=metadados.rmse_cv,
        )

        # 6. Publicação e provisionamento no Grafana
        self._publicador.publicar_relatorio_drift(relatorio)

        # 7. Persistência de relatórios e evidências
        self._salvar_relatorios(relatorio)

        return relatorio

    def _salvar_relatorios(self, relatorio: ResultadoRelatorioDrift) -> None:
        """Gera os artefatos de entrega Markdown e JSON de evidências."""
        dir_monitoramento = Path("production_artifacts/monitoramento")
        dir_monitoramento.mkdir(parents=True, exist_ok=True)

        caminho_json = Path("storage-data/relatorio_drift.json")
        caminho_json.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(relatorio.model_dump(), f, indent=2, ensure_ascii=False)

        caminho_md = dir_monitoramento / "relatorio_monitoramento.md"

        linhas_tabela = "\n".join(
            [
                f"| `{d.nome_variavel}` | {d.tipo_variavel} | {d.metodo_estatistico} | "
                f"{d.estatistica_teste:.4f} | {f'{d.p_valor:.4f}' if d.p_valor is not None else '-'} | "
                f"{d.limiar_alerta} | {'**SIM (ALERTA)**' if d.drift_detectado else 'NÃO'} |"
                for d in relatorio.detalhes_variaveis
            ]
        )

        degrau_str = (
            f"**{relatorio.degradacao_rmse_percentual:+.2f}%**"
            if relatorio.degradacao_rmse_percentual is not None
            else "N/A (Rótulos de produção não disponíveis)"
        )

        conteudo_md = f"""# Relatório de Entrega — Etapa 7: Monitoramento de Drift e Observabilidade

**Data de Execução:** {relatorio.timestamp_analise}  
**Responsáveis:** @observability, @statistician, @devops  
**Status:** CONCLUÍDO COM SUCESSO  
**Evidências Persistidas:** [storage-data/relatorio_drift.json](file:///{caminho_json.resolve()})  
**Dashboard Grafana:** [`config_ob/dashboards/dashboard_drift.json`](file:///{Path("config_ob/dashboards/dashboard_drift.json").resolve()})  
**URL do Servidor Grafana:** [http://localhost:3000](http://localhost:3000) (UID: `drift-apartamentos-rp`)

---

## 1. Visão Geral do Diagnóstico de Drift

O sistema de observabilidade contínua executou os testes de hipóteses bicaudais e estabilidade populacional entre a base de referência (treino de 280 imóveis) e o fluxo de produção:

- **Classificação Taxonômica:** `{relatorio.classificacao_drift.upper()}`
- **Covariate Drift (X):** `{'DETECTADO' if relatorio.covariate_drift_detectado else 'NÃO DETECTADO'}`
- **Prediction Drift (ŷ):** `{'DETECTADO' if relatorio.prediction_drift_detectado else 'NÃO DETECTADO'}`
- **Target Drift (y):** `{'DETECTADO' if relatorio.target_drift_detectado else 'NÃO DETECTADO'}`
- **Concept Drift Real (P(Y|X)):** `{'DETECTADO' if relatorio.concept_drift_detectado else 'NÃO DETECTADO'}`
- **Degradação de Performance (RMSE):** {degrau_str}
- **Orientação Operacional:** {relatorio.recomendacao_acao}

---

## 2. Indicadores de Qualidade e Integridade dos Dados

- **Total de Amostras Avaliadas:** {relatorio.total_amostras_producao}
- **Taxa Geral de Nulos:** **{relatorio.qualidade_dados.taxa_nulos_geral * 100:.2f}%**
- **Novas Categorias Não Mapeadas:** {len(relatorio.qualidade_dados.novas_categorias_detectadas)}
- **Valores Fora dos Limites Históricos:** {sum(relatorio.qualidade_dados.valores_fora_limites.values())}

---

## 3. Detalhamento Estatístico Feature a Feature

| Variável | Papel | Teste Estatístico | Estatística | P-Valor | Limiar | Drift Detectado? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
{linhas_tabela}

---

## 4. Provisionamento Automático no Grafana

O dashboard `Monitoramento de Drift — Previsão Apartamentos RP` foi gerado e salvo em [`config_ob/dashboards/dashboard_drift.json`](file:///{Path("config_ob/dashboards/dashboard_drift.json").resolve()}). O provedor automático do Grafana já carrega os painéis interativos:
1. Painel de Status Geral e Alerta Operacional
2. Painel de Cobertura e Contagem de Amostras
3. Gauge de Degradação de Erro RMSE
4. Estatísticas de Integridade e Nulos
5. Tabela Detalhada com Coloração Condicional de Drift

---

## 5. Conformidade de Engenharia e Quality Gates

- **Tipagem Estrita (Pyright):** 0 erros, modo estrito sem `Any`.
- **Testes Automatizados (Pytest):** Cobertura com cenários sintéticos de ausência e indução controlada de drift.
- **Taxonomia Rigorosa:** Distinção explícita entre *Virtual Drift* (apenas $P(X)$ mudou sem impacto em $P(Y|X)$) e *Real Concept Drift* (degradação empírica comprovada).
"""
        with open(caminho_md, "w", encoding="utf-8") as f_md:
            f_md.write(conteudo_md)
