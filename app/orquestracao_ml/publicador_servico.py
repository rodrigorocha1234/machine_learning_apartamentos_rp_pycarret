from __future__ import annotations
from datetime import datetime, timezone
import os
from typing import final
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from app.contratos_dados.observador_mlflow import EventoPublicacaoMLflow, ObservadorMLflow
from app.orquestracao_ml.modelo_servico import ModeloServico

class ErroPublicacaoMLflow(Exception):
    """Exceção levantada quando ocorre erro durante a publicação ou consulta no MLflow."""
    pass

@final
class PublicadorServico:
    """Subject do padrão Observer responsável por publicar modelos no MLflow Registry e notificar observadores."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._observadores: list[ObservadorMLflow] = []

    def adicionar_observador(self, observador: ObservadorMLflow) -> None:
        """Registra um observador para receber notificações de ciclo de vida do MLflow."""
        if observador not in self._observadores:
            self._observadores.append(observador)

    def remover_observador(self, observador: ObservadorMLflow) -> None:
        """Remove um observador previamente registrado."""
        if observador in self._observadores:
            self._observadores.remove(observador)

    def _notificar_observadores(self, evento: EventoPublicacaoMLflow) -> None:
        """Dispara a notificação para todos os observadores inscritos."""
        for obs in self._observadores:
            try:
                obs.notificar(evento)
            except Exception:
                # Falha em um observador individual não deve abortar os demais nem o fluxo principal
                pass

    def obter_modelo_campeao_salvo(
        self,
        nome_modelo_registrado: str = "previsao_preco_imoveis",
        fallback_padrao: str = "random_forest"
    ) -> str:
        """Recupera o identificador do modelo campeão salvo no MLflow Registry ou nas runs do experimento.
        
        Caso o MLflow não contenha registros ou esteja inacessível, retorna fallback_padrao de forma segura.
        """
        try:
            client = MlflowClient(tracking_uri=self._tracking_uri)
            
            # 1. Tenta recuperar da versão mais recente do modelo registrado
            modelos_registrados = client.search_model_versions(f"name='{nome_modelo_registrado}'")
            if modelos_registrados:
                # Ordena pela versão mais recente
                versoes_ordenadas = sorted(modelos_registrados, key=lambda v: int(v.version), reverse=True)
                ultima_versao = versoes_ordenadas[0]
                run_id = ultima_versao.run_id
                if run_id:
                    run = client.get_run(run_id)
                    # Verifica tags da run
                    modelo_tag = run.data.tags.get("modelo_campeao")
                    if modelo_tag:
                        return str(modelo_tag)
                    # Verifica parâmetros da run
                    modelo_param = run.data.params.get("modelo_campeao")
                    if modelo_param:
                        return str(modelo_param)
        except Exception:
            # Em caso de indisponibilidade ou ausência de modelo registrado, usa fallback
            pass

        return fallback_padrao

    def publicar_modelo_servico(
        self,
        modelo_servico: ModeloServico,
        experimento: str = "imoveis_regressao",
        nome_modelo_registrado: str = "previsao_preco_imoveis",
        tags: dict[str, str] | None = None,
        parametros: dict[str, str | int | float] | None = None,
        metricas: dict[str, float] | None = None,
        caminho_artefato: str = "modelo_imoveis"
    ) -> dict[str, str]:
        """Registra e publica o ModeloServico no MLflow e notifica todos os observadores cadastrados."""
        timestamp_atual = datetime.now(timezone.utc).isoformat()
        tags_efetivas = dict(tags or {})
        parametros_efetivos = dict(parametros or {})
        metricas_efetivas = dict(metricas or {})
        modelo_campeao_nome = str(tags_efetivas.get("modelo_campeao", "desconhecido"))
        modo_exec = str(tags_efetivas.get("modo_execucao", "padrao"))

        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(experimento)

            with mlflow.start_run() as run:
                run_id = run.info.run_id

                if tags_efetivas:
                    mlflow.set_tags(tags_efetivas)

                if parametros_efetivos:
                    mlflow.log_params(parametros_efetivos)

                if metricas_efetivas:
                    mlflow.log_metrics(metricas_efetivas)

                # Log do modelo pyfunc customizado (contém o modelo final + regras dinâmicas de negócio)
                mlflow.pyfunc.log_model(
                    artifact_path=caminho_artefato,
                    python_model=modelo_servico,
                    registered_model_name=nome_modelo_registrado
                )

                model_uri = f"models:/{nome_modelo_registrado}/latest"
                resultado = {
                    "status": "sucesso",
                    "run_id": run_id,
                    "model_uri": model_uri,
                    "modelo_registrado": nome_modelo_registrado,
                    "experimento": experimento
                }

                # Dispara evento de sucesso aos observadores
                evento_sucesso = EventoPublicacaoMLflow(
                    tipo_evento="modelo_publicado",
                    status="sucesso",
                    modelo_campeao=modelo_campeao_nome,
                    modo_execucao=modo_exec,
                    experimento=experimento,
                    nome_modelo_registrado=nome_modelo_registrado,
                    run_id=run_id,
                    model_uri=model_uri,
                    mensagem="Modelo registrado com sucesso no MLflow Model Registry.",
                    metricas=metricas_efetivas,
                    tags=tags_efetivas,
                    timestamp_iso=timestamp_atual
                )
                self._notificar_observadores(evento_sucesso)

                return resultado
        except Exception as erro:
            # Dispara evento de falha aos observadores antes de propagar exceção
            evento_falha = EventoPublicacaoMLflow(
                tipo_evento="falha_publicacao",
                status="erro",
                modelo_campeao=modelo_campeao_nome,
                modo_execucao=modo_exec,
                experimento=experimento,
                nome_modelo_registrado=nome_modelo_registrado,
                mensagem=str(erro),
                tags=tags_efetivas,
                timestamp_iso=timestamp_atual
            )
            self._notificar_observadores(evento_falha)

            raise ErroPublicacaoMLflow(
                f"Falha ao publicar modelo no MLflow ({self._tracking_uri}): {erro}"
            ) from erro
