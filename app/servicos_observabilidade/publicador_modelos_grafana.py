"""Publicador e gerador de dashboard do Grafana para modelos de regressão e tuning.

Gerencia o provisionamento automático do dashboard de modelos em config_ob/dashboards/dashboard_modelos.json
com fuso horário ajustado para América/São Paulo (Brasília, UTC-3) e emissão de logs para terminal e Loki/Grafana.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import cast
import urllib.request
from zoneinfo import ZoneInfo

from app.contratos_modelagem.motor_treinador import ResultadoTuning

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
    ) -> None:
        self._caminho_dashboard = Path(caminho_dashboard)
        self._loki_url = loki_url.rstrip("/")

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

    def publicar_dashboard_modelos(
        self,
        resultado_tuning: ResultadoTuning,
        motor: str,
    ) -> Path:
        """Constrói e persiste o dashboard JSON do Grafana com os modelos e eventos de tuning."""
        self._caminho_dashboard.parent.mkdir(parents=True, exist_ok=True)
        dashboard_json = self._construir_dashboard(resultado_tuning, motor)

        self._caminho_dashboard.write_text(
            json.dumps(dashboard_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self._caminho_dashboard

    def _construir_dashboard(
        self, resultado_tuning: ResultadoTuning, motor: str
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
