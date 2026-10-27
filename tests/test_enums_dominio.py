from __future__ import annotations

from app.enums_dominio import (
    ClassificacaoDrift,
    CriterioDesempate,
    EstrategiaImputacao,
    EtapaPipeline,
    MetaModeloStacking,
    MetodoEstatisticoDrift,
    MetodoTransformacaoEscala,
    MetodoTuning,
    MetricaAvaliacao,
    MotorGraficoTipo,
    MotorTabularTipo,
    MotorTreinoTipo,
    PapelVariavel,
    PosTesteTipo,
    TecnicaValidacaoCruzada,
    TecnicaVotacao,
    TesteEstatisticoTipo,
    TipoDrift,
    TipoEstatisticoVariavel,
    TipoFonteDados,
    TipoGrafico,
    TipoRepositorioStaging,
)
from app.esquema_configuracao import (
    ColunaSelecao,
    ConfiguracaoFonte,
    ConfiguracaoRepositorioStaging,
    ConfiguracaoTabular,
)


def test_etapa_pipeline_valores() -> None:
    """Verifica se os valores de EtapaPipeline correspondem às fases e alvos válidos."""
    assert EtapaPipeline.COLETA == "1"
    assert EtapaPipeline.PREPARACAO == "2"
    assert EtapaPipeline.EDA == "3"
    assert EtapaPipeline.MODELAGEM == "4"
    assert EtapaPipeline.AVALIACAO == "5"
    assert EtapaPipeline.IMPLANTACAO == "6"
    assert EtapaPipeline.DRIFT == "7"
    assert EtapaPipeline.TODAS == "todas"
    assert EtapaPipeline.CAMPEAO == "campeao"
    assert isinstance(EtapaPipeline.COLETA, str)


def test_tipo_fonte_dados_valores() -> None:
    """Valida as tecnologias de fontes primárias de dados."""
    assert TipoFonteDados.XLSX == "xlsx"
    assert TipoFonteDados.CSV == "csv"
    assert TipoFonteDados.SQL == "sql"
    assert TipoFonteDados.API == "api"


def test_tipo_repositorio_staging_valores() -> None:
    """Valida as tecnologias de banco de staging intermediário."""
    assert TipoRepositorioStaging.SQLITE == "sqlite"
    assert TipoRepositorioStaging.POSTGRES == "postgres"
    assert TipoRepositorioStaging.MYSQL == "mysql"


def test_motor_tabular_tipo_valores() -> None:
    """Valida os motores tabulares suportados."""
    assert MotorTabularTipo.PANDAS == "pandas"
    assert MotorTabularTipo.POLARS == "polars"


def test_tipo_estatistico_variavel_valores() -> None:
    """Valida a classificação estatística formal das variáveis."""
    assert TipoEstatisticoVariavel.QUANTITATIVA_CONTINUA == "quantitativa_continua"
    assert TipoEstatisticoVariavel.QUANTITATIVA_DISCRETA == "quantitativa_discreta"
    assert TipoEstatisticoVariavel.QUALITATIVA_NOMINAL == "qualitativa_nominal"
    assert TipoEstatisticoVariavel.QUALITATIVA_ORDINAL == "qualitativa_ordinal"


def test_papel_variavel_valores() -> None:
    """Valida os papéis atribuíveis a cada feature imobiliária."""
    assert PapelVariavel.ALVO == "alvo"
    assert PapelVariavel.NUMERICA == "numerica"
    assert PapelVariavel.CATEGORICA == "categorica"
    assert PapelVariavel.IDENTIFICADOR == "identificador"
    assert PapelVariavel.IGNORAR == "ignorar"


def test_motor_grafico_tipo_valores() -> None:
    """Valida os motores gráficos da análise exploratória."""
    assert MotorGraficoTipo.GRAFANA == "grafana"
    assert MotorGraficoTipo.SEABORN == "seaborn"
    assert MotorGraficoTipo.PLOTLY == "plotly"
    assert MotorGraficoTipo.ALTAIR == "altair"


def test_tipo_grafico_valores() -> None:
    """Valida as representações visuais suportadas."""
    assert TipoGrafico.HISTOGRAMA == "histograma"
    assert TipoGrafico.BOXPLOT == "boxplot"
    assert TipoGrafico.DISPERSAO == "dispersao"
    assert TipoGrafico.CORRELACAO == "correlacao"
    assert TipoGrafico.BARRAS == "barras"
    assert TipoGrafico.TABELA == "tabela"


def test_motor_treino_tipo_valores() -> None:
    """Valida os frameworks de ML suportados."""
    assert MotorTreinoTipo.PYCARET == "pycaret"
    assert MotorTreinoTipo.SKLEARN == "sklearn"


def test_estrategia_imputacao_valores() -> None:
    """Valida as opções de imputação de valores ausentes."""
    assert EstrategiaImputacao.MEDIANA == "mediana"
    assert EstrategiaImputacao.MEDIA == "media"
    assert EstrategiaImputacao.MAIS_FREQUENTE == "mais_frequente"
    assert EstrategiaImputacao.CONSTANTE == "constante"


def test_metodo_transformacao_escala_valores() -> None:
    """Valida as opções de transformação matemática."""
    assert MetodoTransformacaoEscala.LOG1P == "log1p"
    assert MetodoTransformacaoEscala.LOG == "log"
    assert MetodoTransformacaoEscala.YEO_JOHNSON == "yeo-johnson"
    assert MetodoTransformacaoEscala.BOX_COX == "box-cox"


def test_tecnica_validacao_cruzada_valores() -> None:
    """Valida as técnicas de particionamento e folds."""
    assert TecnicaValidacaoCruzada.KFOLD == "kfold"
    assert TecnicaValidacaoCruzada.REPEATED_KFOLD == "repeated_kfold"
    assert TecnicaValidacaoCruzada.SHUFFLE_SPLIT == "shuffle_split"
    assert TecnicaValidacaoCruzada.TIME_SERIES_SPLIT == "time_series_split"


def test_metodo_tuning_valores() -> None:
    """Valida os métodos de busca de hiperparâmetros."""
    assert MetodoTuning.RANDOM == "random"
    assert MetodoTuning.GRID == "grid"
    assert MetodoTuning.BAYESIAN == "bayesian"
    assert MetodoTuning.OPTUNA == "optuna"


def test_metrica_avaliacao_valores() -> None:
    """Valida as métricas estatísticas de regressão."""
    assert MetricaAvaliacao.RMSE == "RMSE"
    assert MetricaAvaliacao.MAE == "MAE"
    assert MetricaAvaliacao.MSE == "MSE"
    assert MetricaAvaliacao.R2 == "R2"
    assert MetricaAvaliacao.MAPE == "MAPE"
    assert MetricaAvaliacao.RMSLE == "RMSLE"


def test_testes_estatisticos_e_desempate() -> None:
    """Valida testes estatísticos e critérios de ranqueamento."""
    assert TesteEstatisticoTipo.FRIEDMAN == "friedman"
    assert PosTesteTipo.NEMENYI == "nemenyi"
    assert CriterioDesempate.MENOR_RMSE == "menor_rmse"
    assert CriterioDesempate.MENOR_MAE == "menor_mae"
    assert CriterioDesempate.MAIOR_R2 == "maior_r2"


def test_tecnica_votacao_e_stacking() -> None:
    """Valida as estratégias de ensemble."""
    assert TecnicaVotacao.MEDIA_SIMPLES == "media_simples"
    assert TecnicaVotacao.VOTING_REGRESSOR == "voting_regressor"
    assert TecnicaVotacao.MEDIA_PONDERADA_RMSE == "media_ponderada_rmse"
    assert TecnicaVotacao.STACKING == "stacking"
    assert MetaModeloStacking.RIDGE == "ridge"
    assert MetaModeloStacking.LINEAR == "linear"


def test_taxonomia_e_metodos_drift() -> None:
    """Valida taxonomia e métodos de detecção de drift."""
    assert TipoDrift.COVARIATE_DRIFT == "covariate_drift"
    assert TipoDrift.CONCEPT_DRIFT == "concept_drift"
    assert TipoDrift.LABEL_DRIFT == "label_drift"
    assert TipoDrift.PREDICTION_DRIFT == "prediction_drift"

    assert ClassificacaoDrift.SEM_DRIFT == "sem_drift"
    assert ClassificacaoDrift.VIRTUAL_DRIFT == "virtual_drift"
    assert ClassificacaoDrift.LABEL_DRIFT == "label_drift"
    assert ClassificacaoDrift.REAL_CONCEPT_DRIFT == "real_concept_drift"

    assert MetodoEstatisticoDrift.KOLMOGOROV_SMIRNOV == "kolmogorov_smirnov"
    assert MetodoEstatisticoDrift.PSI == "psi"
    assert MetodoEstatisticoDrift.WASSERSTEIN == "wasserstein"
    assert MetodoEstatisticoDrift.DEGRADACAO_RMSE == "degradacao_rmse"


def test_integracao_com_pydantic() -> None:
    """Garante que dicionários brutos (como do YAML) são automaticamente convertidos para Enums nos modelos Pydantic."""
    cfg_fonte = ConfiguracaoFonte.model_validate({"tipo": "xlsx", "caminho": "dados.xlsx"})
    assert cfg_fonte.tipo == TipoFonteDados.XLSX

    cfg_staging = ConfiguracaoRepositorioStaging.model_validate({"tipo": "sqlite", "caminho": "staging.db"})
    assert cfg_staging.tipo == TipoRepositorioStaging.SQLITE

    cfg_tabular = ConfiguracaoTabular.model_validate({"motor": "pandas"})
    assert cfg_tabular.motor == MotorTabularTipo.PANDAS

    col = ColunaSelecao.model_validate({
        "nome": "preco",
        "ativo": True,
        "tipo": "quantitativa_continua",
        "papel": "alvo",
    })
    assert col.tipo == TipoEstatisticoVariavel.QUANTITATIVA_CONTINUA
    assert col.papel == PapelVariavel.ALVO
