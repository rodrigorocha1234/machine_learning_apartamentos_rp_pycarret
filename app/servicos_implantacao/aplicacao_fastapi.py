"""Fábrica da aplicação FastAPI para serving de inferência e cálculo de regras."""

from __future__ import annotations

from typing import cast
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field

from app.contratos_implantacao.avaliador_regras import (
    RequisicaoImovel,
    RespostaPredicao,
    StatusServico,
)
from app.esquema_configuracao import ConfiguracaoImplantacao
from app.servicos_implantacao.servico_inferencia import ErroServicoInferencia, ServicoInferencia


class RequisicaoInvocationsMlflow(BaseModel):
    """Esquema de payload compatível com MLflow Serving (instances ou split)."""

    dataframe_records: list[dict[str, object]] | None = Field(
        default=None, description="Registros no formato records de DataFrame"
    )
    instances: list[dict[str, object]] | None = Field(
        default=None, description="Lista de instâncias no formato de inferência"
    )


def criar_aplicacao_fastapi(
    servico_inferencia: ServicoInferencia,
    config_implantacao: ConfiguracaoImplantacao,
) -> FastAPI:
    """Cria e configura a instância do FastAPI para serving de modelos e regras."""
    app = FastAPI(
        title="API de Previsão de Preços de Imóveis - Ribeirão Preto",
        description="Microsserviço de inferência com modelo campeão e regras dinâmicas de negócio.",
        version="1.0.0",
    )

    @app.get("/health", response_model=StatusServico, tags=["Saúde"])
    def verificar_saude() -> StatusServico:
        """Endpoint de verificação de disponibilidade operacional e versão do modelo."""
        meta = servico_inferencia.obter_provedor().obter_metadados()
        total_ativas = len([r for r in config_implantacao.regras if r.ativo])
        return StatusServico(
            status="operacional",
            versao_api="1.0.0",
            modelo_campeao=meta.nome_modelo,
            versao_modelo=meta.versao,
            total_regras_ativas=total_ativas,
            ambiente="producao",
        )

    @app.post("/predict", response_model=RespostaPredicao, tags=["Inferência"])
    def predizer(requisicao: RequisicaoImovel) -> RespostaPredicao:
        """Calcula o preço de venda e regras de negócio para uma requisição de imóvel."""
        try:
            return servico_inferencia.predizer_imovel(requisicao)
        except ErroServicoInferencia as erro:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Falha na inferência do modelo: {erro}",
            ) from erro
        except Exception as erro_inesperado:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro interno não tratado: {erro_inesperado}",
            ) from erro_inesperado

    # Rota configurada via YAML (ex: /invocations)
    endpoint_serving = config_implantacao.serving.endpoint.strip()
    if not endpoint_serving.startswith("/"):
        endpoint_serving = f"/{endpoint_serving}"

    @app.post(endpoint_serving, tags=["Serving MLflow"])
    def invocar_serving(payload: dict[str, object]) -> JSONResponse:
        """Endpoint compatível com MLflow Serving (/invocations) e chamadas em lote."""
        try:
            registros: list[dict[str, object]] = []
            if "dataframe_records" in payload and isinstance(payload["dataframe_records"], list):
                registros = cast(list[dict[str, object]], payload["dataframe_records"])
            elif "instances" in payload and isinstance(payload["instances"], list):
                registros = cast(list[dict[str, object]], payload["instances"])
            elif "Metragem" in payload:
                # Payload plano direto de um único imóvel
                registros = [payload]
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Formato de payload não reconhecido. Esperado 'dataframe_records', 'instances' ou campos do imóvel.",
                )

            df_entrada = pd.DataFrame(registros)
            df_resultado = servico_inferencia.predizer_lote(df_entrada)
            resultado_dict = cast(list[dict[str, object]], df_resultado.to_dict(orient="records"))
            return JSONResponse(content={"predictions": resultado_dict})
        except HTTPException:
            raise
        except Exception as erro:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro ao processar invocação de serving: {erro}",
            ) from erro

    return app
