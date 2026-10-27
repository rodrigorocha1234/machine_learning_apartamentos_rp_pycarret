"""Enumerações fortemente tipadas do domínio do sistema de previsão de preços.

Define enums baseados em StrEnum (Python 3.11+) para manter compatibilidade total
com valores textuais de arquivos YAML e modelos Pydantic, garantindo tipagem
estrita (zero Any), eliminando 'magic strings' e mantendo nomes em português.
"""

from __future__ import annotations

from enum import StrEnum


class EtapaPipeline(StrEnum):
    """Etapas numeradas e alvos executáveis da Cadeia de Responsabilidade do pipeline."""

    COLETA = "1"
    PREPARACAO = "2"
    EDA = "3"
    MODELAGEM = "4"
    AVALIACAO = "5"
    IMPLANTACAO = "6"
    DRIFT = "7"
    TODAS = "todas"
    CAMPEAO = "campeao"


class TipoFonteDados(StrEnum):
    """Formatos e tecnologias suportadas para a fonte primária de dados."""

    XLSX = "xlsx"
    CSV = "csv"
    SQL = "sql"
    API = "api"


class TipoRepositorioStaging(StrEnum):
    """Bancos relacionais e tecnologias de persistência intermediária (staging)."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"


class MotorTabularTipo(StrEnum):
    """Motores de processamento tabular de DataFrames suportados pelo pipeline."""

    PANDAS = "pandas"
    POLARS = "polars"


class TipoEstatisticoVariavel(StrEnum):
    """Classificação estatística formal das variáveis dos dados imobiliários."""

    QUANTITATIVA_CONTINUA = "quantitativa_continua"
    QUANTITATIVA_DISCRETA = "quantitativa_discreta"
    QUALITATIVA_NOMINAL = "qualitativa_nominal"
    QUALITATIVA_ORDINAL = "qualitativa_ordinal"


class PapelVariavel(StrEnum):
    """Papel funcional da coluna na modelagem e pré-processamento."""

    ALVO = "alvo"
    NUMERICA = "numerica"
    CATEGORICA = "categorica"
    IDENTIFICADOR = "identificador"
    IGNORAR = "ignorar"


class MotorGraficoTipo(StrEnum):
    """Motores de renderização e exportação de análises visuais exploratórias."""

    GRAFANA = "grafana"
    SEABORN = "seaborn"
    PLOTLY = "plotly"
    ALTAIR = "altair"


class TipoGrafico(StrEnum):
    """Tipos de representações gráficas e visuais configuráveis na EDA."""

    HISTOGRAMA = "histograma"
    BOXPLOT = "boxplot"
    DISPERSAO = "dispersao"
    CORRELACAO = "correlacao"
    BARRAS = "barras"
    TABELA = "tabela"


class MotorTreinoTipo(StrEnum):
    """Frameworks de machine learning suportados para treino e regressão."""

    PYCARET = "pycaret"
    SKLEARN = "sklearn"


class EstrategiaImputacao(StrEnum):
    """Estratégias para tratamento e preenchimento de valores ausentes (nulos)."""

    MEDIANA = "mediana"
    MEDIA = "media"
    MAIS_FREQUENTE = "mais_frequente"
    CONSTANTE = "constante"


class MetodoTransformacaoEscala(StrEnum):
    """Métodos matemáticos para transformação monotônica de escala e normalização."""

    LOG1P = "log1p"
    LOG = "log"
    YEO_JOHNSON = "yeo-johnson"
    BOX_COX = "box-cox"


class TecnicaValidacaoCruzada(StrEnum):
    """Técnicas de particionamento e amostragem para validação cruzada (CV)."""

    KFOLD = "kfold"
    REPEATED_KFOLD = "repeated_kfold"
    SHUFFLE_SPLIT = "shuffle_split"
    TIME_SERIES_SPLIT = "time_series_split"


class MetodoTuning(StrEnum):
    """Algoritmos de busca e otimização de hiperparâmetros de regressores."""

    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"


class MetricaAvaliacao(StrEnum):
    """Métricas estatísticas de avaliação de performance para modelos de regressão."""

    RMSE = "RMSE"
    MAE = "MAE"
    MSE = "MSE"
    R2 = "R2"
    MAPE = "MAPE"
    RMSLE = "RMSLE"


class TipoTesteEstatistico(StrEnum):
    """Testes estatísticos de hipóteses para comparação de modelos em validação cruzada."""

    __test__ = False
    FRIEDMAN = "friedman"


class TipoPosTeste(StrEnum):
    """Pós-testes estatísticos de comparações múltiplas pareadas."""

    NEMENYI = "nemenyi"


# Aliases para compatibilidade e flexibilidade
TesteEstatisticoTipo = TipoTesteEstatistico
PosTesteTipo = TipoPosTeste


class CriterioDesempate(StrEnum):
    """Critérios de ordenação e desempate para seleção do modelo campeão."""

    MENOR_RMSE = "menor_rmse"
    MENOR_MAE = "menor_mae"
    MAIOR_R2 = "maior_r2"


class TecnicaVotacao(StrEnum):
    """Estratégias de ensemble e combinação de predições de regressores candidatos."""

    MEDIA_SIMPLES = "media_simples"
    VOTING_REGRESSOR = "voting_regressor"
    MEDIA_PONDERADA_RMSE = "media_ponderada_rmse"
    STACKING = "stacking"


class MetaModeloStacking(StrEnum):
    """Modelos meta-estimadores suportados para regressão empilhada (stacking)."""

    RIDGE = "ridge"
    LINEAR = "linear"


class TipoDrift(StrEnum):
    """Taxonomia de drifts monitorados em produção."""

    COVARIATE_DRIFT = "covariate_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"
    PREDICTION_DRIFT = "prediction_drift"


class ClassificacaoDrift(StrEnum):
    """Diagnóstico taxonômico final emitido pela análise contínua de drift."""

    SEM_DRIFT = "sem_drift"
    VIRTUAL_DRIFT = "virtual_drift"
    LABEL_DRIFT = "label_drift"
    REAL_CONCEPT_DRIFT = "real_concept_drift"
    SUDDEN_DRIFT = "sudden_drift"
    GRADUAL_DRIFT = "gradual_drift"


class MetodoEstatisticoDrift(StrEnum):
    """Testes e métricas matemáticas para cálculo de divergência estatística e drift."""

    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    PSI = "psi"
    WASSERSTEIN = "wasserstein"
    DEGRADACAO_RMSE = "degradacao_rmse"
