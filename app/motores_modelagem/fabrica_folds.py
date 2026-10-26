"""Fábrica de estratégias de validação cruzada (folds).

Instancia divisores de cross-validation configurados via YAML, suportando
múltiplas técnicas com validação estrita de parâmetros.
"""

from typing import cast
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    ShuffleSplit,
    TimeSeriesSplit,
)

from app.esquema_configuracao import (
    ConfiguracaoFolds,
    TecnicaFolds,
    ValorParametroFolds,
)


class ErroFabricaFolds(Exception):
    """Exceção levantada quando ocorre erro na fábrica de folds."""


type GeradorFolds = KFold | RepeatedKFold | ShuffleSplit | TimeSeriesSplit


class FabricaFolds:
    """Fábrica para instanciar divisores de validação cruzada do Scikit-Learn."""

    def criar_divisor(
        self,
        config: ConfiguracaoFolds,
        folds_fallback: int = 3,
    ) -> GeradorFolds:
        """Instancia a técnica de validação cruzada conforme a configuração fornecida.

        Se `config.ativo` for False, retorna KFold padrão com o número de folds fallback.
        """
        if not config.ativo:
            return KFold(n_splits=folds_fallback, shuffle=True, random_state=42)

        tecnica: TecnicaFolds = config.tecnica
        params_tecnica: dict[str, ValorParametroFolds] = config.parametros.get(tecnica, {})

        match tecnica:
            case "kfold":
                return self._criar_kfold(params_tecnica, folds_fallback)
            case "repeated_kfold":
                return self._criar_repeated_kfold(params_tecnica, folds_fallback)
            case "shuffle_split":
                return self._criar_shuffle_split(params_tecnica, folds_fallback)
            case "time_series_split":
                return self._criar_time_series_split(params_tecnica, folds_fallback)
            case _:
                raise ErroFabricaFolds(f"Técnica de folds não suportada: '{tecnica}'")

    def _criar_kfold(
        self,
        params: dict[str, ValorParametroFolds],
        folds_fallback: int,
    ) -> KFold:
        n_splits = int(cast(int, params.get("n_splits", folds_fallback)))
        shuffle = bool(params.get("shuffle", True))
        raw_seed = params.get("random_state", 42)
        random_state = int(cast(int, raw_seed)) if raw_seed is not None else None
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _criar_repeated_kfold(
        self,
        params: dict[str, ValorParametroFolds],
        folds_fallback: int,
    ) -> RepeatedKFold:
        n_splits = int(cast(int, params.get("n_splits", folds_fallback)))
        n_repeats = int(cast(int, params.get("n_repeats", 3)))
        raw_seed = params.get("random_state", 42)
        random_state = int(cast(int, raw_seed)) if raw_seed is not None else None
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    def _criar_shuffle_split(
        self,
        params: dict[str, ValorParametroFolds],
        folds_fallback: int,
    ) -> ShuffleSplit:
        n_splits = int(cast(int, params.get("n_splits", folds_fallback)))
        raw_test_size = params.get("test_size", 0.2)
        test_size = float(cast(float, raw_test_size)) if raw_test_size is not None else 0.2
        raw_seed = params.get("random_state", 42)
        random_state = int(cast(int, raw_seed)) if raw_seed is not None else None
        return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    def _criar_time_series_split(
        self,
        params: dict[str, ValorParametroFolds],
        folds_fallback: int,
    ) -> TimeSeriesSplit:
        n_splits = int(cast(int, params.get("n_splits", folds_fallback)))
        raw_max_train = params.get("max_train_size", None)
        max_train_size = int(cast(int, raw_max_train)) if raw_max_train is not None else None
        return TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
