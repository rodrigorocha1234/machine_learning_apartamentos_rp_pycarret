"""Publicador e gerador de dashboard do Grafana para modelos de regressão e tuning.

Gerencia o provisionamento automático do dashboard de modelos em config_ob/dashboards/dashboard_modelos.json
e a emissão de logs de tuning para o terminal e integração com Loki/Grafana.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import cast
import urllib.request

from app.contratos_modelagem.motor_treinador import ResultadoTuning

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
        """Registra um evento de tuning no histórico e envia de forma resiliente ao Loki."""
        timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        registro: dict[str, str] = {
            "timestamp": timestamp_iso,
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

        # Montagem do texto markdown de eventos/logs de tuning
        linhas_tabela_logs: list[str] = [
            "### Linha do Tempo e Eventos de Tuning de Modelos",
            "",
            "| Horário | Modelo | Evento de Execução Registrado |",
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
                    f"| Concluído | **{mod_legivel}** | `{msg}` |"
                )

        conteudo_markdown_logs = "\n".join(linhas_tabela_logs)

        # Montagem das linhas estruturadas de métricas por modelo para a tabela
        dados_tabela_modelos: list[dict[str, object]] = []
        for mod in resultado_tuning.modelos_treinados:
            mod_legivel = formatar_nome_modelo(mod.nome_modelo)
            rmse = mod.metricas.get("RMSE", 0.0)
            mae = mod.metricas.get("MAE", 0.0)
            r2 = mod.metricas.get("R2", 0.0)
            status_tuning = "Ativo (Otimizado)" if mod.tuning_ativo else "Desativado (Padrão)"
            msg = mod.mensagem_tuning or f"realizando tuning de parâmetros {mod_legivel}"
            params_str = (
                ", ".join(f"{k}={v}" for k, v in mod.melhores_parametros.items())
                if mod.melhores_parametros
                else "Parâmetros baseline"
            )

            dados_tabela_modelos.append(
                {
                    "modelo": mod_legivel,
                    "status_tuning": status_tuning,
                    "evento_tuning": msg,
                    "rmse": f"R$ {rmse:,.2f}",
                    "mae": f"R$ {mae:,.2f}",
                    "r2": f"{r2:.4f}",
                    "melhores_parametros": params_str,
                    "duracao_segundos": f"{mod.duracao_segundos:.2f}s",
                }
            )

        paineis: list[dict[str, object]] = [
            # Painel 1: Stat Modelo Campeão
            {
                "id": 1,
                "title": "Modelo Campeão do Tuning",
                "type": "stat",
                "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
                "options": {
                    "colorMode": "background",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "auto",
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "fixed", "fixedColor": "green"},
                        "thresholds": {"mode": "absolute", "steps": []},
                    }
                },
                "description": f"Modelo selecionado com o menor erro de CV: {nome_campeao_formatado}",
            },
            # Painel 2: Stat Melhor RMSE
            {
                "id": 2,
                "title": "Melhor RMSE (Validação Cruzada)",
                "type": "stat",
                "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                },
                "fieldConfig": {
                    "defaults": {
                        "unit": "currencyBRL",
                        "color": {"mode": "fixed", "fixedColor": "blue"},
                        "thresholds": {"mode": "absolute", "steps": []},
                    }
                },
                "description": f"RMSE R$ {resultado_tuning.melhor_rmse:,.2f}",
            },
            # Painel 3: Stat Total de Modelos
            {
                "id": 3,
                "title": "Modelos Avaliados",
                "type": "stat",
                "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "fixed", "fixedColor": "purple"},
                        "thresholds": {"mode": "absolute", "steps": []},
                    }
                },
                "description": f"Total de {len(resultado_tuning.modelos_treinados)} modelos avaliados",
            },
            # Painel 4: Stat Motor e Método
            {
                "id": 4,
                "title": "Motor & Busca",
                "type": "stat",
                "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "auto",
                },
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "fixed", "fixedColor": "orange"},
                        "thresholds": {"mode": "absolute", "steps": []},
                    }
                },
                "description": f"Motor: {motor.upper()} | Busca: {resultado_tuning.metodo_tuning.upper()} ({resultado_tuning.iteracoes} iterações)",
            },
            # Painel 5: Painel de Logs e Eventos de Tuning (Loki / Texto formatado)
            {
                "id": 5,
                "title": "Eventos e Logs de Tuning no Grafana",
                "type": "text",
                "gridPos": {"x": 0, "y": 4, "w": 24, "h": 8},
                "options": {
                    "mode": "markdown",
                    "content": conteudo_markdown_logs,
                },
                "description": "Feed em tempo real dos eventos de tuning emitidos para cada modelo de regressão.",
            },
            # Painel 6: Tabela Detalhada de Modelos e Hiperparâmetros
            {
                "id": 6,
                "title": "Desempenho dos Modelos e Hiperparâmetros de Tuning",
                "type": "table",
                "gridPos": {"x": 0, "y": 12, "w": 24, "h": 10},
                "options": {
                    "sortBy": [{"displayName": "RMSE", "desc": False}],
                },
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "align": "left",
                            "filterable": True,
                        }
                    }
                },
                "description": "Mapeamento completo dos hiperparâmetros ajustados, status e métricas.",
                "custom": {
                    "dados_modelos": dados_tabela_modelos,
                },
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
