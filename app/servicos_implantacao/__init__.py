"""Pacote de serviços de implantação e promoção de modelo no MLflow Registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.servicos_implantacao.modelo_mlflow import ModeloComRegrasNegocio

if TYPE_CHECKING:
    from app.servicos_implantacao.carregador_campeao import CarregadorCampeao
    from app.servicos_implantacao.orquestrador_implantacao import (
        OrquestradorImplantacao,
        ResultadoImplantacao,
    )


def __getattr__(name: str) -> Any:
    if name == "CarregadorCampeao":
        from app.servicos_implantacao.carregador_campeao import CarregadorCampeao

        return CarregadorCampeao
    if name in ("OrquestradorImplantacao", "ResultadoImplantacao"):
        from app.servicos_implantacao.orquestrador_implantacao import (
            OrquestradorImplantacao,
            ResultadoImplantacao,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CarregadorCampeao",
    "ModeloComRegrasNegocio",
    "OrquestradorImplantacao",
    "ResultadoImplantacao",
]

