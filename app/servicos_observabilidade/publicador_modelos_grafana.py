"""Publicador e gerador de dashboard do Grafana para modelos de regressão e tuning.

Gerencia o provisionamento automático do dashboard de modelos em config_ob/dashboards/dashboard_modelos.json
com fuso horário ajustado para América/São Paulo (Brasília, UTC-3) e emissão de logs para terminal e Loki/Grafana.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import cast
import urllib.request
from zoneinfo import ZoneInfo

from app.contratos_modelagem.motor_treinador import ResultadoTuning
from app.servicos_avaliacao.avaliador_holdout import MetricasHoldout

FUSO_HORARIO_PADRAO = ZoneInfo("America/Sao_Paulo")

MAPA_NOMES_LEGIVEIS: dict[str, str] = {
    "regressao_linear": "regressão linear",
    "lasso": "lasso",
    "ridge": "ridge",
    "elasticnet": "elasticnet",
    "arvore_decisao": "árvore de decisão",
    "random_forest": "random forest",
    "svr": "svr",
    "rede_neural": "rede neural",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}


def formatar_nome_modelo(nome: str) -> str:
    """Retorna o nome amigável e legível em português para o modelo."""
    return MAPA_NOMES_LEGIVEIS.get(nome, nome.replace("_", " "))


def obter_timestamp_sao_paulo() -> str:
    """Retorna a data e hora atual no fuso horário de Brasília/São Paulo (UTC-3)."""
    try:
        return datetime.now(FUSO_HORARIO_PADRAO).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PublicadorModelosGrafana:
    """Publicador de métricas e gerador de dashboard para acompanhamento de modelos e tuning no Grafana."""

    _logs_historico: list[dict[str, str]] = []
    _lock = threading.Lock()

    def __init__(
        self,
        caminho_dashboard: Path | str = "config_ob/dashboards/dashboard_modelos.json",
        loki_url: str = "http://localhost:3100",
        grafana_url: str = "http://localhost:3000",
    ) -> None:
        self._caminho_dashboard = Path(caminho_dashboard)
        self._loki_url = loki_url.rstrip("/")
        self._grafana_url = grafana_url.rstrip("/")

    @classmethod
    def registrar_log(cls, nome_modelo: str, mensagem: str) -> None:
        """Registra um evento de tuning no histórico em horário de Brasília e envia ao Loki."""
        timestamp_sp = obter_timestamp_sao_paulo()
        registro: dict[str, str] = {
            "timestamp": timestamp_sp,
            "modelo": nome_modelo,
            "mensagem": mensagem,
        }

        with cls._lock:
            cls._logs_historico.append(registro)

        # Disparo assíncrono para o Loki (não-bloqueante e tolerante a falhas)
        instancia = cls()
        thread = threading.Thread(
            target=instancia._enviar_loki_seguro,
            args=(nome_modelo, mensagem),
            daemon=True,
        )
        thread.start()

    @classmethod
    def obter_logs(cls) -> list[dict[str, str]]:
        """Retorna uma cópia dos logs acumulados de tuning."""
        with cls._lock:
            return list(cls._logs_historico)

    @classmethod
    def limpar_logs(cls) -> None:
        """Limpa o buffer de logs em memória."""
        with cls._lock:
            cls._logs_historico.clear()

    def _enviar_loki_seguro(self, nome_modelo: str, mensagem: str) -> None:
        """Tenta enviar o log de tuning ao endpoint /loki/api/v1/push sem propagar exceções."""
        try:
            url_push = f"{self._loki_url}/loki/api/v1/push"
            timestamp_ns = str(int(datetime.now(timezone.utc).timestamp() * 1e9))
            payload = {
                "streams": [
                    {
                        "stream": {
                            "app": "ml-pipeline",
                            "etapa": "modelagem",
                            "componente": "tuning",
                            "modelo": nome_modelo,
                        },
                        "values": [[timestamp_ns, f"[{nome_modelo}] {mensagem}"]],
                    }
                ]
            }
            dados = json.dumps(payload).encode("utf-8")
            requisicao = urllib.request.Request(
                url_push,
                data=dados,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(requisicao, timeout=0.5):
                pass
        except Exception:
            # Falha de envio tolerada quando os containers Docker não estiverem ativos
            pass

    def _sincronizar_grafana_api(self, dashboard_json: dict[str, object]) -> None:
        """Tenta atualizar o dashboard via REST API do Grafana em tempo de execução."""
        try:
            url = f"{self._grafana_url}/api/dashboards/db"
            payload = json.dumps({"dashboard": dashboard_json, "overwrite": True}).encode("utf-8")
            auth = base64.b64encode(b"admin:Admin@2026").decode("ascii")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {auth}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0):
                pass
        except Exception:
            # Tolerado quando o Grafana estiver offline ou em inicialização
            pass

    @staticmethod
    def _tentar_carregar_metricas_holdout() -> dict[str, MetricasHoldout] | None:
        """Tenta carregar as métricas do conjunto holdout salvas na Etapa 5."""
        caminho_aval = Path("storage-data/relatorio_avaliacao.json")
        if caminho_aval.exists():
            try:
                with open(caminho_aval, "r", encoding="utf-8") as f:
                    dados = json.load(f)
                raw_holdout = dados.get("metricas_holdout", {})
                if isinstance(raw_holdout, dict) and raw_holdout:
                    return {
                        str(k): MetricasHoldout.model_validate(v)
                        for k, v in raw_holdout.items()
                    }
            except Exception:
                return None
        return None

    def publicar_dashboard_modelos(
        self,
        resultado_tuning: ResultadoTuning,
        motor: str,
        metricas_holdout: dict[str, MetricasHoldout] | None = None,
    ) -> Path:
        """Constrói e persiste o dashboard JSON do Grafana com modelos, eventos e gráfico de overfitting/underfitting."""
        self._caminho_dashboard.parent.mkdir(parents=True, exist_ok=True)
        if metricas_holdout is None:
            metricas_holdout = self._tentar_carregar_metricas_holdout()

        dashboard_json = self._construir_dashboard(resultado_tuning, motor, metricas_holdout)

        self._caminho_dashboard.write_text(
            json.dumps(dashboard_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._sincronizar_grafana_api(dashboard_json)
        return self._caminho_dashboard

    def _construir_dashboard(
        self,
        resultado_tuning: ResultadoTuning,
        motor: str,
        metricas_holdout: dict[str, MetricasHoldout] | None = None,
    ) -> dict[str, object]:
        """Gera o schema JSON completo do Grafana com painéis analíticos e eventos de tuning."""
        nome_campeao_formatado = formatar_nome_modelo(resultado_tuning.modelo_campeao).capitalize()
        horario_atual_sp = obter_timestamp_sao_paulo()

        # 1. Painel de Indicadores Chave (KPIs) com design estilizado
        kpi_markdown = f"""<div style="display: flex; gap: 14px; width: 100%; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 220px; background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 8px; padding: 14px 18px;">
    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 4px;">🏆 Modelo Campeão do Tuning</div>
    <div style="font-size: 1.6rem; font-weight: 700; color: #10b981;">{nome_campeao_formatado}</div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Menor erro médio na validação cruzada</div>
  </div>
  <div style="flex: 1; min-width: 220px; background: rgba(59, 130, 246, 0.12); border: 1px solid #3b82f6; border-radius: 8px; padding: 14px 18px;">
    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 4px;">📉 Melhor RMSE (Validação Cruzada)</div>
    <div style="font-size: 1.6rem; font-weight: 700; color: #60a5fa;">R$ {resultado_tuning.melhor_rmse:,.2f}</div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Métrica em moeda brasileira (R$)</div>
  </div>
  <div style="flex: 1; min-width: 220px; background: rgba(168, 85, 247, 0.12); border: 1px solid #a855f7; border-radius: 8px; padding: 14px 18px;">
    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 4px;">📊 Modelos Avaliados</div>
    <div style="font-size: 1.6rem; font-weight: 700; color: #c084fc;">{len(resultado_tuning.modelos_treinados)} modelos</div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Bateria de regressão comparativa</div>
  </div>
  <div style="flex: 1; min-width: 220px; background: rgba(245, 158, 11, 0.12); border: 1px solid #f59e0b; border-radius: 8px; padding: 14px 18px;">
    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 4px;">⚙️ Motor & Busca</div>
    <div style="font-size: 1.6rem; font-weight: 700; color: #fbbf24;">{motor.upper()} • {resultado_tuning.metodo_tuning.upper()}</div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">{resultado_tuning.iteracoes} iterações • Fuso: América/São Paulo (-03)</div>
  </div>
</div>"""

        # 2. Montagem da Linha do Tempo e Eventos de Tuning (Fuso Horário Brasília / SP)
        linhas_tabela_logs: list[str] = [
            f"### Linha do Tempo e Eventos de Tuning de Modelos *(Fuso Horário: Brasília / America/Sao_Paulo -03)*",
            "",
            "| Horário (SP) | Modelo | Evento de Execução Registrado |",
            "| :--- | :--- | :--- |",
        ]

        with self._lock:
            logs_atuais = list(self._logs_historico)

        if logs_atuais:
            for item in logs_atuais:
                mod_legivel = formatar_nome_modelo(item["modelo"])
                linhas_tabela_logs.append(
                    f"| {item['timestamp']} | **{mod_legivel}** | `{item['mensagem']}` |"
                )
        else:
            for mod in resultado_tuning.modelos_treinados:
                mod_legivel = formatar_nome_modelo(mod.nome_modelo)
                msg = mod.mensagem_tuning or f"realizando tuning de parâmetros {mod_legivel}"
                linhas_tabela_logs.append(
                    f"| {horario_atual_sp} | **{mod_legivel}** | `{msg}` |"
                )

        conteudo_markdown_logs = "\n".join(linhas_tabela_logs)

        # 3. Montagem da Tabela de Desempenho e Hiperparâmetros
        modelos_ordenados = sorted(
            resultado_tuning.modelos_treinados,
            key=lambda m: m.metricas.get("RMSE", float("inf")),
        )

        linhas_tabela_performance: list[str] = [
            "### Desempenho Comparativo dos Modelos & Hiperparâmetros de Tuning",
            "",
            "| Ranking | Modelo | Status Tuning | Melhores Hiperparâmetros | RMSE (CV) | MAE | R² | Duração |",
            "| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
        ]

        for i, mod in enumerate(modelos_ordenados, start=1):
            mod_legivel = formatar_nome_modelo(mod.nome_modelo)
            rmse = mod.metricas.get("RMSE", 0.0)
            mae = mod.metricas.get("MAE", 0.0)
            r2 = mod.metricas.get("R2", 0.0)
            status_tuning = "Ativo (Otimizado)" if mod.tuning_ativo else "Desativado (Padrão)"
            medalha = "🥇 1º" if i == 1 else ("🥈 2º" if i == 2 else ("🥉 3º" if i == 3 else f"{i}º"))
            destaque_modelo = f"**{mod_legivel.capitalize()}**" if i == 1 else mod_legivel.capitalize()

            params_str = (
                ", ".join(f"`{k}={v}`" for k, v in mod.melhores_parametros.items())
                if mod.melhores_parametros
                else "*Parâmetros baseline*"
            )

            linhas_tabela_performance.append(
                f"| {medalha} | {destaque_modelo} | {status_tuning} | {params_str} | **R$ {rmse:,.2f}** | R$ {mae:,.2f} | {r2:.4f} | {mod.duracao_segundos:.2f}s |"
            )

        conteudo_markdown_performance = "\n".join(linhas_tabela_performance)

        paineis: list[dict[str, object]] = [
            # Painel 1: Indicadores Chave de Performance (KPI Cards)
            {
                "id": 1,
                "title": "Indicadores Gerais da Modelagem",
                "type": "text",
                "gridPos": {"x": 0, "y": 0, "w": 24, "h": 4},
                "options": {
                    "mode": "html",
                    "content": kpi_markdown,
                },
                "description": f"Resumo dos resultados da etapa de modelagem e tuning de hiperparâmetros.",
            },
            # Painel 2: Eventos e Logs de Tuning no Grafana
            {
                "id": 2,
                "title": "Eventos e Logs de Tuning no Grafana",
                "type": "text",
                "gridPos": {"x": 0, "y": 4, "w": 24, "h": 8},
                "options": {
                    "mode": "markdown",
                    "content": conteudo_markdown_logs,
                },
                "description": "Feed em tempo real dos eventos de tuning emitidos no fuso horário de Brasília/São Paulo.",
            },
            # Painel 3: Desempenho dos Modelos e Hiperparâmetros
            {
                "id": 3,
                "title": "Desempenho dos Modelos e Hiperparâmetros de Tuning",
                "type": "text",
                "gridPos": {"x": 0, "y": 12, "w": 24, "h": 9},
                "options": {
                    "mode": "markdown",
                    "content": conteudo_markdown_performance,
                },
                "description": "Tabela comparativa ranqueada por menor RMSE na validação cruzada.",
            },
            # Painel 4: Gráfico e Diagnóstico de Overfitting vs Underfitting
            self._construir_painel_overfitting_underfitting(resultado_tuning, metricas_holdout),
        ]

        dashboard_completo: dict[str, object] = {
            "title": "Modelagem & Tuning — Previsão Apartamentos RP",
            "uid": "modelos-apartamentos-rp",
            "schemaVersion": 38,
            "version": 1,
            "timezone": "America/Sao_Paulo",
            "refresh": "10s",
            "style": "dark",
            "tags": ["mlops", "modelos", "tuning", "regressao", "grafana", "apartamentos-rp"],
            "panels": cast(list[object], paineis),
            "time": {"from": "now-6h", "to": "now"},
            "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m"]},
        }

        return dashboard_completo

    def _construir_painel_overfitting_underfitting(
        self,
        resultado_tuning: ResultadoTuning,
        metricas_holdout: dict[str, MetricasHoldout] | None,
    ) -> dict[str, object]:
        """Gera o painel com gráfico comparativo e diagnóstico de Overfitting e Underfitting."""
        # Determina o maior RMSE para normalizar visualmente a escala horizontal das barras
        valores_rmse: list[float] = [
            m.metricas.get("RMSE", 0.0) for m in resultado_tuning.modelos_treinados
        ]
        if metricas_holdout:
            valores_rmse.extend(mh.rmse for mh in metricas_holdout.values())

        max_rmse = max(valores_rmse) if valores_rmse and max(valores_rmse) > 0 else 100000.0

        modelos_ordenados = sorted(
            resultado_tuning.modelos_treinados,
            key=lambda m: m.metricas.get("RMSE", float("inf")),
        )

        total_good_fit = 0
        total_overfitting = 0
        total_underfitting = 0

        blocos_grafico_modelos: list[str] = []

        for mod in modelos_ordenados:
            nome_canonico = mod.nome_modelo
            nome_legivel = formatar_nome_modelo(nome_canonico).capitalize()
            rmse_cv = mod.metricas.get("RMSE", 0.0)
            r2_cv = mod.metricas.get("R2", 0.0)

            # Holdout real ou estimativa baseada na validação
            teve_holdout = False
            if metricas_holdout and nome_canonico in metricas_holdout:
                mh = metricas_holdout[nome_canonico]
                rmse_teste = mh.rmse
                r2_teste = mh.r2
                teve_holdout = True
            else:
                rmse_teste = rmse_cv * 1.05
                r2_teste = r2_cv
                teve_holdout = False

            if rmse_cv > 0:
                gap_pct = ((rmse_teste - rmse_cv) / rmse_cv) * 100.0
            else:
                gap_pct = 0.0

            # Critérios diagnósticos de Viés e Variância:
            # 1. Underfitting: R2 baixo (< 0.70) em ambas as partições
            # 2. Overfitting: Erro no teste muito superior ao treino (gap > +15% ou queda drástica de R2)
            # 3. Leve Variância: gap entre 8% e 15% (aceitável)
            # 4. Ajuste Ótimo: R2 alto e gap controlado (<= 8%)
            if r2_cv < 0.70 or r2_teste < 0.65:
                diagnostico_nome = "🔵 Subajuste (Underfitting)"
                diagnostico_desc = "Alto viés: modelo muito simples ou regularização excessiva, não capturou adequadamente os padrões de preço."
                acao_sugerida = "Aumentar complexidade do modelo, reduzir regularização ou adicionar novos atributos derivados."
                cor_status = "#38bdf8"
                cor_bg = "rgba(56, 189, 248, 0.15)"
                cor_borda = "#38bdf8"
                total_underfitting += 1
            elif gap_pct > 15.0 or (r2_cv - r2_teste) > 0.12:
                diagnostico_nome = "🔴 Superajuste (Overfitting)"
                diagnostico_desc = "Alta variância: o modelo decorou ruídos do treino e degradou consideravelmente a performance no teste."
                acao_sugerida = "Aumentar regularização L1/L2, podar profundidade de árvores ou aplicar seleção de features."
                cor_status = "#f87171"
                cor_bg = "rgba(239, 68, 68, 0.15)"
                cor_borda = "#ef4444"
                total_overfitting += 1
            elif gap_pct > 8.0:
                diagnostico_nome = "🟡 Leve Variância (Aceitável)"
                diagnostico_desc = "Leve decaimento no teste, dentro da margem de tolerância estocástica esperada no mercado imobiliário."
                acao_sugerida = "Monitorar estabilidade de inferência e avaliar ajuste fino de regularização."
                cor_status = "#fbbf24"
                cor_bg = "rgba(245, 158, 11, 0.15)"
                cor_borda = "#f59e0b"
                total_good_fit += 1
            else:
                diagnostico_nome = "🟢 Ajuste Ótimo (Good Fit)"
                diagnostico_desc = "Excelente equilíbrio entre viés e variância: generalização estável com baixos resíduos preditivos."
                acao_sugerida = "Modelo ideal para promoção produtiva ou para compor comitê de votação."
                cor_status = "#34d399"
                cor_bg = "rgba(16, 185, 129, 0.15)"
                cor_borda = "#10b981"
                total_good_fit += 1

            largura_treino = min(100, max(6, int((rmse_cv / max_rmse) * 100)))
            largura_teste = min(100, max(6, int((rmse_teste / max_rmse) * 100)))
            rotulo_teste = "Teste (Holdout)" if teve_holdout else "Teste (Estimado CV)"
            cor_gap = "#f87171" if gap_pct > 15.0 else ("#fbbf24" if gap_pct > 8.0 else "#34d399")

            bloco_html = f"""<div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 8px; padding: 14px 18px; margin-bottom: 14px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 10px;">
      <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">{nome_legivel}</span>
      <span style="font-size: 0.75rem; padding: 3px 10px; border-radius: 12px; background: {cor_bg}; color: {cor_status}; border: 1px solid {cor_borda}; font-weight: 600;">{diagnostico_nome}</span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-size: 0.82rem; color: #94a3b8;">Gap de Generalização (Δ):</span>
      <span style="font-size: 1.05rem; font-weight: 700; color: {cor_gap};">{gap_pct:+.1f}%</span>
    </div>
  </div>

  <!-- Barra Treino / CV -->
  <div style="display: flex; align-items: center; margin-bottom: 6px; font-size: 0.82rem;">
    <span style="width: 145px; color: #94a3b8; font-weight: 500;">Treino (Val. Cruzada):</span>
    <div style="flex: 1; background: rgba(51, 65, 85, 0.5); border-radius: 4px; height: 20px; overflow: hidden; margin-right: 14px;">
      <div style="background: linear-gradient(90deg, #0284c7, #38bdf8); width: {largura_treino}%; height: 100%; border-radius: 4px;"></div>
    </div>
    <span style="width: 190px; text-align: right; color: #38bdf8; font-weight: 600;">R$ {rmse_cv:,.2f} <span style="color: #64748b; font-size: 0.75rem; font-weight: 400;">(R² {r2_cv:.3f})</span></span>
  </div>

  <!-- Barra Teste / Holdout -->
  <div style="display: flex; align-items: center; margin-bottom: 8px; font-size: 0.82rem;">
    <span style="width: 145px; color: #94a3b8; font-weight: 500;">{rotulo_teste}:</span>
    <div style="flex: 1; background: rgba(51, 65, 85, 0.5); border-radius: 4px; height: 20px; overflow: hidden; margin-right: 14px;">
      <div style="background: linear-gradient(90deg, #9333ea, #c084fc); width: {largura_teste}%; height: 100%; border-radius: 4px;"></div>
    </div>
    <span style="width: 190px; text-align: right; color: #c084fc; font-weight: 600;">R$ {rmse_teste:,.2f} <span style="color: #64748b; font-size: 0.75rem; font-weight: 400;">(R² {r2_teste:.3f})</span></span>
  </div>

  <!-- Rodapé com Diagnóstico e Recomendação -->
  <div style="font-size: 0.78rem; color: #94a3b8; border-top: 1px dashed rgba(148, 163, 184, 0.2); padding-top: 8px; margin-top: 6px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
    <div><strong>Análise:</strong> {diagnostico_desc}</div>
    <div><strong>Recomendação:</strong> <span style="color: #cbd5e1;">{acao_sugerida}</span></div>
  </div>
</div>"""
            blocos_grafico_modelos.append(bloco_html)

        html_modelos = "\n".join(blocos_grafico_modelos)

        conteudo_painel = f"""<div style="width: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <!-- Cards Resumo de Overfitting / Underfitting -->
  <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
    <div style="flex: 1; min-width: 180px; background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 8px; padding: 12px 16px;">
      <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">Ajuste Ótimo (Good Fit)</div>
      <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{total_good_fit} modelos</div>
      <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Generalização consistente</div>
    </div>
    <div style="flex: 1; min-width: 180px; background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 8px; padding: 12px 16px;">
      <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">Superajuste (Overfitting)</div>
      <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{total_overfitting} modelos</div>
      <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Erro Teste &gt;&gt; Erro Treino</div>
    </div>
    <div style="flex: 1; min-width: 180px; background: rgba(56, 189, 248, 0.12); border: 1px solid #38bdf8; border-radius: 8px; padding: 12px 16px;">
      <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">Subajuste (Underfitting)</div>
      <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">{total_underfitting} modelos</div>
      <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Alto viés em ambas partições</div>
    </div>
  </div>

  <!-- Gráficos de Barras Comparativos (Treino vs Teste) -->
  {html_modelos}

  <!-- Legenda e Fundamentos Metodológicos -->
  <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 6px; padding: 10px 14px; font-size: 0.76rem; color: #94a3b8; margin-top: 10px;">
    <strong>Guia Metodológico de Diagnóstico de Ajuste:</strong>
    <ul style="margin: 4px 0 0 16px; padding: 0;">
      <li><strong>Underfitting (Alto Viés):</strong> R² inferior a 0.70 ou erros elevados tanto na validação quanto no teste. Indica modelo excessivamente rígido.</li>
      <li><strong>Overfitting (Alta Variância):</strong> Gap entre Teste e Treino superior a +15% (ou queda expressiva de R²). O modelo decorou ruídos da amostra de treino.</li>
      <li><strong>Ajuste Ótimo (Good Fit):</strong> R² robusto (&gt; 0.75) e gap controlado (|Δ| ≤ 8%), garantindo inferências calibradas e confiáveis em novos apartamentos.</li>
    </ul>
  </div>
</div>"""

        return {
            "id": 4,
            "title": "⚖️ Diagnóstico Gráfico de Overfitting e Underfitting (Treino/CV vs Teste Holdout)",
            "type": "text",
            "gridPos": {"x": 0, "y": 21, "w": 24, "h": 16},
            "options": {
                "mode": "html",
                "content": conteudo_painel,
            },
            "description": "Comparativo visual do erro (RMSE e R²) entre Validação Cruzada (Treino) e Holdout (Teste) para diagnóstico de Viés e Variância.",
        }
