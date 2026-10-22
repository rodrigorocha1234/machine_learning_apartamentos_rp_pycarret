from __future__ import annotations
import pytest

from app.tipos_enumerados import (
    EstrategiaFold,
    EstrategiaImputacao,
    MetodoTransformacao,
    MetodoTuning,
    MotorEda,
    MotorTabular,
    MotorTreinoTipo,
    TecnicaVotacao,
    TipoCodificador,
    TipoEscalonador,
    TipoGraficoEda,
    TipoModelo,
)


def test_tipo_escalonador_valores_e_normalizacao() -> None:
    assert TipoEscalonador.STANDARD == "standard"
    assert TipoEscalonador.MINMAX == "minmax"
    assert TipoEscalonador.de_string("Standard") == TipoEscalonador.STANDARD
    assert TipoEscalonador.de_string("MINMAX") == TipoEscalonador.MINMAX
    assert TipoEscalonador.de_string("none") == TipoEscalonador.NENHUM
    assert TipoEscalonador.de_string("") == TipoEscalonador.NENHUM

    with pytest.raises(ValueError, match="Escalonador não suportado"):
        TipoEscalonador.de_string("invalido_escalonador")


def test_tipo_codificador_valores_e_normalizacao() -> None:
    assert TipoCodificador.ONE_HOT == "one_hot"
    assert TipoCodificador.ORDINAL == "ordinal"
    assert TipoCodificador.de_string("one-hot") == TipoCodificador.ONE_HOT
    assert TipoCodificador.de_string("ONE_HOT") == TipoCodificador.ONE_HOT
    assert TipoCodificador.de_string("none") == TipoCodificador.NENHUM

    with pytest.raises(ValueError, match="Codificador não suportado"):
        TipoCodificador.de_string("codificador_desconhecido")


def test_metodo_transformacao_aliases() -> None:
    assert MetodoTransformacao.de_string("log1p") == MetodoTransformacao.LOG1P
    assert MetodoTransformacao.de_string("LOG") == MetodoTransformacao.LOG
    assert MetodoTransformacao.de_string("yeojohnson") == MetodoTransformacao.YEO_JOHNSON
    assert MetodoTransformacao.de_string("yeo-johnson") == MetodoTransformacao.YEO_JOHNSON
    assert MetodoTransformacao.de_string("boxcox") == MetodoTransformacao.BOX_COX
    assert MetodoTransformacao.de_string("box-cox") == MetodoTransformacao.BOX_COX
    assert MetodoTransformacao.de_string("nenhum") == MetodoTransformacao.NENHUM

    with pytest.raises(ValueError, match="Método de transformação não suportado"):
        MetodoTransformacao.de_string("invalido")


def test_estrategia_fold_aliases() -> None:
    assert EstrategiaFold.de_string("kfold") == EstrategiaFold.KFOLD
    assert EstrategiaFold.de_string("KF") == EstrategiaFold.KFOLD
    assert EstrategiaFold.de_string("stratifiedkfold") == EstrategiaFold.STRATIFIED_KFOLD
    assert EstrategiaFold.de_string("timeseries") == EstrategiaFold.TIME_SERIES
    assert EstrategiaFold.de_string("groupkfold") == EstrategiaFold.GROUP_KFOLD

    with pytest.raises(ValueError, match="Estratégia de fold não suportada"):
        EstrategiaFold.de_string("fold_errado")


def test_metodo_tuning_aliases() -> None:
    assert MetodoTuning.de_string("random") == MetodoTuning.RANDOM
    assert MetodoTuning.de_string("random_search") == MetodoTuning.RANDOM
    assert MetodoTuning.de_string("grid") == MetodoTuning.GRID
    assert MetodoTuning.de_string("halving") == MetodoTuning.HALVING
    assert MetodoTuning.de_string("optuna") == MetodoTuning.OPTUNA
    assert MetodoTuning.de_string("tune-sklearn") == MetodoTuning.TUNE_SKLEARN

    with pytest.raises(ValueError, match="Método de tuning não suportado"):
        MetodoTuning.de_string("busca_inexistente")


def test_tipo_modelo_aliases_e_pycaret_id() -> None:
    assert TipoModelo.de_string("linearregression") == TipoModelo.REGRESSAO_LINEAR
    assert TipoModelo.de_string("lr") == TipoModelo.REGRESSAO_LINEAR
    assert TipoModelo.REGRESSAO_LINEAR.obter_pycaret_id() == "lr"

    assert TipoModelo.de_string("rf") == TipoModelo.RANDOM_FOREST
    assert TipoModelo.RANDOM_FOREST.obter_pycaret_id() == "rf"

    assert TipoModelo.de_string("svm") == TipoModelo.SVR
    assert TipoModelo.SVR.obter_pycaret_id() == "svm"

    assert TipoModelo.de_string("mlp") == TipoModelo.REDE_NEURAL
    assert TipoModelo.REDE_NEURAL.obter_pycaret_id() == "mlp"

    assert TipoModelo.de_string("en") == TipoModelo.ELASTICNET
    assert TipoModelo.ELASTICNET.obter_pycaret_id() == "en"

    assert TipoModelo.de_string("dt") == TipoModelo.ARVORE_DECISAO
    assert TipoModelo.ARVORE_DECISAO.obter_pycaret_id() == "dt"

    assert TipoModelo.de_string("xgb") == TipoModelo.XGBOOST
    assert TipoModelo.XGBOOST.obter_pycaret_id() == "xgboost"

    assert TipoModelo.de_string("lgbm") == TipoModelo.LIGHTGBM
    assert TipoModelo.LIGHTGBM.obter_pycaret_id() == "lightgbm"

    with pytest.raises(ValueError, match="Modelo não suportado"):
        TipoModelo.de_string("modelo_inexistente")


def test_estrategia_imputacao_aliases() -> None:
    assert EstrategiaImputacao.de_string("mediana") == EstrategiaImputacao.MEDIANA
    assert EstrategiaImputacao.de_string("median") == EstrategiaImputacao.MEDIANA
    assert EstrategiaImputacao.de_string("media") == EstrategiaImputacao.MEDIA
    assert EstrategiaImputacao.de_string("mean") == EstrategiaImputacao.MEDIA
    assert EstrategiaImputacao.de_string("most_frequent") == EstrategiaImputacao.MAIS_FREQUENTE
    assert EstrategiaImputacao.de_string("mode") == EstrategiaImputacao.MAIS_FREQUENTE
    assert EstrategiaImputacao.de_string("constant") == EstrategiaImputacao.CONSTANTE
    assert EstrategiaImputacao.de_string("knn") == EstrategiaImputacao.KNN
    assert EstrategiaImputacao.de_string("mice") == EstrategiaImputacao.ITERATIVA

    with pytest.raises(ValueError, match="Estratégia de imputação não suportada"):
        EstrategiaImputacao.de_string("imputacao_invalida")


def test_outros_enums_sistema() -> None:
    assert MotorTreinoTipo.de_string("scikit-learn") == MotorTreinoTipo.SKLEARN
    assert MotorTreinoTipo.de_string("pycaret") == MotorTreinoTipo.PYCARET

    assert MotorTabular.de_string("pandas") == MotorTabular.PANDAS
    assert MotorTabular.de_string("polars") == MotorTabular.POLARS

    assert MotorEda.de_string("plotly") == MotorEda.PLOTLY
    assert MotorEda.de_string("ydata_profiling") == MotorEda.YDATA

    assert TipoGraficoEda.de_string("histograma") == TipoGraficoEda.HISTOGRAMA
    assert TipoGraficoEda.de_string("boxplot") == TipoGraficoEda.BOXPLOT

    assert TecnicaVotacao.de_string("voting-regressor") == TecnicaVotacao.VOTING_REGRESSOR
    assert TecnicaVotacao.de_string("media_aparada") == TecnicaVotacao.MEDIA_APARADA
