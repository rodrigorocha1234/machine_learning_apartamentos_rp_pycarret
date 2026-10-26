"""Módulo de centralização de configurações do projeto.

Permite a importação direta de ConfYAML e ErroConfiguracao tanto a partir da raiz
do projeto (`from conf_yaml import ConfYAML`) quanto a partir do pacote app (`from app.conf_yaml import ConfYAML`).
"""

from __future__ import annotations

from app.conf_yaml import ConfYAML, ErroConfiguracao

__all__ = ["ConfYAML", "ErroConfiguracao"]
