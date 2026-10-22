from __future__ import annotations
from enum import StrEnum
from typing import final


@final
class TipoEscalonador(StrEnum):
    """Escalonadores suportados para normalização numérica."""
    STANDARD = "standard"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    NENHUM = "nenhum"

    @classmethod
    def de_string(cls, valor: str | TipoEscalonador) -> TipoEscalonador:
        texto = str(valor).lower().strip()
        for membro in cls:
            if membro.value == texto:
                return membro
        if texto in ("none", "null", "", "false"):
            return cls.NENHUM
        raise ValueError(f"Escalonador não suportado: '{valor}'")


@final
class TipoCodificador(StrEnum):
    """Codificadores para variáveis categóricas."""
    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    NENHUM = "nenhum"

    @classmethod
    def de_string(cls, valor: str | TipoCodificador) -> TipoCodificador:
        texto = str(valor).lower().strip().replace("-", "_")
        for membro in cls:
            if membro.value == texto:
                return membro
        if texto in ("none", "null", "", "false"):
            return cls.NENHUM
        raise ValueError(f"Codificador não suportado: '{valor}'")


@final
class MetodoTransformacao(StrEnum):
    """Métodos de transformação matemática para alvo ou atributos."""
    LOG1P = "log1p"
    LOG = "log"
    YEO_JOHNSON = "yeo-johnson"
    BOX_COX = "box-cox"
    NENHUM = "nenhum"

    @classmethod
    def de_string(cls, valor: str | MetodoTransformacao) -> MetodoTransformacao:
        texto = str(valor).lower().strip().replace("_", "-")
        if texto in ("yeojohnson", "yeo-johnson", "yeo_johnson"):
            return cls.YEO_JOHNSON
        if texto in ("boxcox", "box-cox", "box_cox"):
            return cls.BOX_COX
        if texto == "log1p":
            return cls.LOG1P
        if texto == "log":
            return cls.LOG
        if texto in ("none", "null", "nenhum", "", "false"):
            return cls.NENHUM
        raise ValueError(f"Método de transformação não suportado: '{valor}'")


@final
class EstrategiaFold(StrEnum):
    """Estratégias de particionamento para validação cruzada."""
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratifiedkfold"
    TIME_SERIES = "timeseries"
    GROUP_KFOLD = "groupkfold"

    @classmethod
    def de_string(cls, valor: str | EstrategiaFold) -> EstrategiaFold:
        texto = str(valor).lower().strip().replace("_", "").replace("-", "")
        if texto in ("kfold", "kf"):
            return cls.KFOLD
        if texto in ("stratifiedkfold", "stratified", "skf"):
            return cls.STRATIFIED_KFOLD
        if texto in ("timeseries", "timeseriessplit", "ts"):
            return cls.TIME_SERIES
        if texto in ("groupkfold", "group", "gkf"):
            return cls.GROUP_KFOLD
        raise ValueError(f"Estratégia de fold não suportada: '{valor}'")


@final
class MetodoTuning(StrEnum):
    """Métodos de busca e otimização de hiperparâmetros."""
    RANDOM = "random"
    GRID = "grid"
    HALVING = "halving"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"
    TUNE_SKLEARN = "tune-sklearn"

    @classmethod
    def de_string(cls, valor: str | MetodoTuning) -> MetodoTuning:
        texto = str(valor).lower().strip()
        texto_traco = texto.replace("_", "-")
        if texto in ("random", "randomized", "random_search") or texto_traco in ("random-search",):
            return cls.RANDOM
        if texto in ("grid", "grid_search") or texto_traco in ("grid-search",):
            return cls.GRID
        if texto in ("halving", "halving_random", "halving_grid") or texto_traco in ("halving-random", "halving-grid"):
            return cls.HALVING
        if texto in ("bayesian", "bayes"):
            return cls.BAYESIAN
        if texto in ("optuna",):
            return cls.OPTUNA
        if texto_traco in ("tune-sklearn", "tunesklearn"):
            return cls.TUNE_SKLEARN
        raise ValueError(f"Método de tuning não suportado: '{valor}'")


@final
class TipoModelo(StrEnum):
    """Catálogo de modelos de regressão suportados."""
    REGRESSAO_LINEAR = "regressao_linear"
    LASSO = "lasso"
    RIDGE = "ridge"
    ELASTICNET = "elasticnet"
    ARVORE_DECISAO = "arvore_decisao"
    RANDOM_FOREST = "random_forest"
    SVR = "svr"
    REDE_NEURAL = "rede_neural"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    @classmethod
    def de_string(cls, valor: str | TipoModelo) -> TipoModelo:
        texto = str(valor).lower().strip().replace("-", "_")
        if texto in ("regressao_linear", "linearregression", "lr", "linear"):
            return cls.REGRESSAO_LINEAR
        if texto in ("lasso",):
            return cls.LASSO
        if texto in ("ridge",):
            return cls.RIDGE
        if texto in ("elasticnet", "elastic_net", "en"):
            return cls.ELASTICNET
        if texto in ("arvore_decisao", "decisiontreeregressor", "decision_tree", "dt"):
            return cls.ARVORE_DECISAO
        if texto in ("random_forest", "randomforestregressor", "rf"):
            return cls.RANDOM_FOREST
        if texto in ("svr", "svm", "svm_regressor"):
            return cls.SVR
        if texto in ("rede_neural", "mlpregressor", "mlp", "neural_network"):
            return cls.REDE_NEURAL
        if texto in ("xgboost", "xgbregressor", "xgb"):
            return cls.XGBOOST
        if texto in ("lightgbm", "lgbmregressor", "lgb", "lgbm"):
            return cls.LIGHTGBM
        raise ValueError(f"Modelo não suportado: '{valor}'")

    def obter_pycaret_id(self) -> str:
        """Retorna o identificador nativo aceito pelo PyCaret create_model()."""
        mapeamento: dict[TipoModelo, str] = {
            TipoModelo.REGRESSAO_LINEAR: "lr",
            TipoModelo.LASSO: "lasso",
            TipoModelo.RIDGE: "ridge",
            TipoModelo.ELASTICNET: "en",
            TipoModelo.ARVORE_DECISAO: "dt",
            TipoModelo.RANDOM_FOREST: "rf",
            TipoModelo.SVR: "svm",
            TipoModelo.REDE_NEURAL: "mlp",
            TipoModelo.XGBOOST: "xgboost",
            TipoModelo.LIGHTGBM: "lightgbm",
        }
        return mapeamento[self]


@final
class EstrategiaImputacao(StrEnum):
    """Estratégias de imputação de valores ausentes."""
    MEDIANA = "mediana"
    MEDIA = "media"
    MAIS_FREQUENTE = "mais_frequente"
    CONSTANTE = "constante"
    KNN = "knn"
    ITERATIVA = "iterativa"

    @classmethod
    def de_string(cls, valor: str | EstrategiaImputacao) -> EstrategiaImputacao:
        texto = str(valor).lower().strip().replace("-", "_")
        if texto in ("mediana", "median"):
            return cls.MEDIANA
        if texto in ("media", "mean"):
            return cls.MEDIA
        if texto in ("mais_frequente", "most_frequent", "mode", "moda"):
            return cls.MAIS_FREQUENTE
        if texto in ("constante", "constant"):
            return cls.CONSTANTE
        if texto in ("knn", "knn_imputer"):
            return cls.KNN
        if texto in ("iterativa", "iterative", "mice"):
            return cls.ITERATIVA
        raise ValueError(f"Estratégia de imputação não suportada: '{valor}'")


@final
class MotorTreinoTipo(StrEnum):
    """Trilhas de treino de modelos."""
    SKLEARN = "sklearn"
    PYCARET = "pycaret"

    @classmethod
    def de_string(cls, valor: str | MotorTreinoTipo) -> MotorTreinoTipo:
        texto = str(valor).lower().strip()
        if texto in ("sklearn", "scikit-learn", "scikit_learn"):
            return cls.SKLEARN
        if texto in ("pycaret", "py-caret"):
            return cls.PYCARET
        raise ValueError(f"Motor de treino não suportado: '{valor}'")


@final
class MotorTabular(StrEnum):
    """Motores computacionais tabulares para processamento de expressões."""
    PANDAS = "pandas"
    POLARS = "polars"

    @classmethod
    def de_string(cls, valor: str | MotorTabular) -> MotorTabular:
        texto = str(valor).lower().strip()
        if texto == "pandas":
            return cls.PANDAS
        if texto == "polars":
            return cls.POLARS
        raise ValueError(f"Motor tabular não suportado: '{valor}'")


@final
class MotorEda(StrEnum):
    """Motores de análise exploratória de dados (EDA)."""
    PLOTLY = "plotly"
    SWEETVIZ = "sweetviz"
    YDATA = "ydata"
    ALTAIR = "altair"
    BOKEH = "bokeh"
    NATIVO = "nativo"

    @classmethod
    def de_string(cls, valor: str | MotorEda) -> MotorEda:
        texto = str(valor).lower().strip()
        for membro in cls:
            if membro.value == texto:
                return membro
        if texto in ("ydata_profiling", "ydata-profiling", "profiling"):
            return cls.YDATA
        raise ValueError(f"Motor EDA não suportado: '{valor}'")


@final
class TipoGraficoEda(StrEnum):
    """Tipos de gráficos e visualizações no catálogo de EDA."""
    HISTOGRAMA = "histograma"
    BOXPLOT = "boxplot"
    DISPERSAO = "dispersao"
    CORRELACAO = "correlacao"
    BARRAS = "barras"
    TABELA = "tabela"

    @classmethod
    def de_string(cls, valor: str | TipoGraficoEda) -> TipoGraficoEda:
        texto = str(valor).lower().strip()
        for membro in cls:
            if membro.value == texto:
                return membro
        raise ValueError(f"Tipo de gráfico não suportado: '{valor}'")


@final
class TecnicaVotacao(StrEnum):
    """Técnicas de ensemble e votação de modelos."""
    MEDIA_SIMPLES = "media_simples"
    VOTING_REGRESSOR = "voting_regressor"
    MEDIA_PONDERADA_RMSE = "media_ponderada_rmse"
    MEDIA_PONDERADA_POSTOS = "media_ponderada_postos"
    MEDIANA = "mediana"
    MEDIA_APARADA = "media_aparada"
    STACKING = "stacking"

    @classmethod
    def de_string(cls, valor: str | TecnicaVotacao) -> TecnicaVotacao:
        texto = str(valor).lower().strip().replace("-", "_")
        for membro in cls:
            if membro.value == texto:
                return membro
        raise ValueError(f"Técnica de votação não suportada: '{valor}'")
