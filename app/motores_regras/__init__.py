"""Pacote de motores de cálculo de regras de negócio configuradas dinamicamente."""

from __future__ import annotations

from app.motores_regras.avaliador_pandas import AvaliadorPandas
from app.motores_regras.avaliador_polars import AvaliadorPolars

__all__ = ["AvaliadorPandas", "AvaliadorPolars"]
