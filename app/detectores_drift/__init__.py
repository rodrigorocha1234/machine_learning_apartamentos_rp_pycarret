"""Pacote de componentes estatísticos de detecção de data drift, prediction drift e qualidade."""

from __future__ import annotations

from app.detectores_drift.avaliador_qualidade import AvaliadorQualidade
from app.detectores_drift.calculador_drift import CalculadorDrift

__all__ = ["AvaliadorQualidade", "CalculadorDrift"]
