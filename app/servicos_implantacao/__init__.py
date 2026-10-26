"""Pacote de serviços de implantação, serialização de modelo campeão e serving HTTP."""

from __future__ import annotations

from app.servicos_implantacao.aplicacao_fastapi import criar_aplicacao_fastapi
from app.servicos_implantacao.carregador_campeao import CarregadorCampeao
from app.servicos_implantacao.orquestrador_implantacao import OrquestradorImplantacao
from app.servicos_implantacao.servico_inferencia import ServicoInferencia

__all__ = [
    "CarregadorCampeao",
    "OrquestradorImplantacao",
    "ServicoInferencia",
    "criar_aplicacao_fastapi",
]
