"""Testes unitários para a fábrica de folds e estratégias de particionamento."""

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    ShuffleSplit,
    TimeSeriesSplit,
)

from app.enums_dominio import TecnicaValidacaoCruzada
from app.esquema_configuracao import ConfiguracaoFolds
from app.motores_modelagem.fabrica_folds import FabricaFolds
from app.motores_modelagem.treinador_pycaret import TreinadorPycaret
from app.motores_modelagem.treinador_sklearn import TreinadorSklearn


def test_fabrica_folds_kfold():
    fabrica = FabricaFolds()
    config = ConfiguracaoFolds(
        ativo=True,
        tecnica=TecnicaValidacaoCruzada.KFOLD,
        parametros={"kfold": {"n_splits": 4, "shuffle": True, "random_state": 123}},
    )
    divisor = fabrica.criar_divisor(config, folds_fallback=3)
    assert isinstance(divisor, KFold)
    assert divisor.n_splits == 4
    assert divisor.shuffle is True
    assert divisor.random_state == 123


def test_fabrica_folds_repeated_kfold():
    fabrica = FabricaFolds()
    config = ConfiguracaoFolds(
        ativo=True,
        tecnica=TecnicaValidacaoCruzada.REPEATED_KFOLD,
        parametros={"repeated_kfold": {"n_splits": 5, "n_repeats": 2, "random_state": 99}},
    )
    divisor = fabrica.criar_divisor(config, folds_fallback=3)
    assert isinstance(divisor, RepeatedKFold)
    assert divisor.get_n_splits() == 10  # 5 splits * 2 repeats


def test_fabrica_folds_shuffle_split():
    fabrica = FabricaFolds()
    config = ConfiguracaoFolds(
        ativo=True,
        tecnica=TecnicaValidacaoCruzada.SHUFFLE_SPLIT,
        parametros={"shuffle_split": {"n_splits": 6, "test_size": 0.25, "random_state": 42}},
    )
    divisor = fabrica.criar_divisor(config, folds_fallback=3)
    assert isinstance(divisor, ShuffleSplit)
    assert divisor.get_n_splits() == 6
    assert divisor.test_size == 0.25


def test_fabrica_folds_time_series_split():
    fabrica = FabricaFolds()
    config = ConfiguracaoFolds(
        ativo=True,
        tecnica=TecnicaValidacaoCruzada.TIME_SERIES_SPLIT,
        parametros={"time_series_split": {"n_splits": 4, "max_train_size": 100}},
    )
    divisor = fabrica.criar_divisor(config, folds_fallback=3)
    assert isinstance(divisor, TimeSeriesSplit)
    assert divisor.n_splits == 4
    assert divisor.max_train_size == 100


def test_fabrica_folds_inativo_retorna_fallback():
    fabrica = FabricaFolds()
    config = ConfiguracaoFolds(
        ativo=False,
        tecnica=TecnicaValidacaoCruzada.TIME_SERIES_SPLIT,
        parametros={"time_series_split": {"n_splits": 10}},
    )
    divisor = fabrica.criar_divisor(config, folds_fallback=5)
    assert isinstance(divisor, KFold)
    assert divisor.n_splits == 5
    assert divisor.shuffle is True


def test_treinador_sklearn_com_fabrica_folds_injetada():
    fabrica = FabricaFolds()
    treinador = TreinadorSklearn(fabrica_folds=fabrica)
    assert treinador.obter_nome_motor() == "sklearn"


def test_treinador_pycaret_com_fabrica_folds_injetada():
    fabrica = FabricaFolds()
    treinador = TreinadorPycaret(fabrica_folds=fabrica)
    assert treinador.obter_nome_motor() == "pycaret"

