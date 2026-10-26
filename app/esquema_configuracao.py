"""Modelos e esquemas fortemente tipados para a configuração do projeto.

Implementa os esquemas de validação de dados para todas as etapas do pipeline
sem nenhuma utilização de Any.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ConfiguracaoProjeto(BaseModel):
    """Configurações gerais de identificação do projeto."""

    nome: str = Field(min_length=1, description="Nome identificador do projeto")
    alvo: str = Field(min_length=1, description="Nome da variável alvo a ser prevista")
    semente: int = Field(default=42, description="Semente aleatória para reprodutibilidade")


class ConfiguracaoFonte(BaseModel):
    """Configurações da fonte primária de ingestão de dados."""

    tipo: Literal["xlsx", "csv", "sql", "api"] = Field(
        default="xlsx", description="Formato da fonte primária de dados"
    )
    caminho: str = Field(min_length=1, description="Caminho do arquivo ou recurso de dados")
    aba: int | str = Field(default=0, description="Índice numérico ou nome da aba da planilha")


class ConfiguracaoRepositorioStaging(BaseModel):
    """Configurações do banco intermediário de persistência (staging)."""

    tipo: Literal["sqlite", "postgres", "mysql"] = Field(
        default="sqlite", description="Tipo de tecnologia do banco de staging"
    )
    caminho: str = Field(min_length=1, description="Caminho ou DSN de conexão do staging")
    tabela: str = Field(
        default="imoveis_preparados", description="Nome da tabela de persistência"
    )


class ConfiguracaoColeta(BaseModel):
    """Configurações e validações de atributos de negócio na etapa de coleta."""

    nome_alvo: str | None = Field(
        default=None, description="Nome da variável alvo a ser validada nos dados coletados"
    )
    colunas_obrigatorias: list[str] = Field(
        default_factory=list, description="Lista de atributos de negócio obrigatórios na fonte"
    )
    validar_nulos_alvo: bool = Field(
        default=False, description="Flag para rejeitar a coleta caso haja valores ausentes no alvo"
    )
    substituir_staging: bool = Field(
        default=True, description="Indica se deve sobrescrever a tabela no staging intermediário"
    )


class ColunaSelecao(BaseModel):
    """Definição individual de uma coluna selecionada para o pipeline."""

    nome: str = Field(min_length=1, description="Nome da coluna no conjunto de dados")
    ativo: bool = Field(default=True, description="Indica se a coluna está ativa para uso")
    tipo: Literal[
        "quantitativa_continua",
        "quantitativa_discreta",
        "qualitativa_nominal",
        "qualitativa_ordinal",
    ] = Field(description="Tipo estatístico da variável")
    papel: Literal["alvo", "numerica", "categorica", "identificador", "ignorar"] = Field(
        description="Papel da coluna na modelagem"
    )


class ConfiguracaoSelecao(BaseModel):
    """Configurações da seleção de atributos e tipagem."""

    ativo: bool = Field(default=True, description="Flag geral para ativação da seleção")
    colunas: list[ColunaSelecao] = Field(
        default=[], description="Lista de colunas selecionadas"
    )


class ExpressaoFiltro(BaseModel):
    """Definição de uma regra de filtro condicional."""

    nome: str = Field(min_length=1, description="Nome do filtro")
    ativo: bool = Field(default=True, description="Flag de ativação do filtro individual")
    expressao: dict[str, str] = Field(
        description="Mapeamento de expressões por motor tabular (pandas, polars)"
    )


class ConfiguracaoFiltros(BaseModel):
    """Configurações de filtragem condicional de registros."""

    ativo: bool = Field(default=True, description="Flag geral para ativação dos filtros")
    expressoes: list[ExpressaoFiltro] = Field(
        default=[], description="Lista de regras de filtro"
    )


class ExpressaoTransformacao(BaseModel):
    """Definição de uma transformação sobre coluna existente."""

    nome: str = Field(min_length=1, description="Nome da transformação")
    ativo: bool = Field(default=True, description="Flag de ativação da transformação")
    destino: str = Field(min_length=1, description="Coluna destino modificada")
    expressao: dict[str, str] = Field(
        description="Mapeamento de expressões por motor tabular (pandas, polars)"
    )


class ConfiguracaoTransformacoes(BaseModel):
    """Configurações de transformação de atributos existentes."""

    ativo: bool = Field(default=True, description="Flag geral de transformações")
    expressoes: list[ExpressaoTransformacao] = Field(
        default=[], description="Lista de transformações"
    )


class ExpressaoAtributoNovo(BaseModel):
    """Definição de criação de um novo atributo derivado."""

    nome: str = Field(min_length=1, description="Nome do novo atributo")
    ativo: bool = Field(default=True, description="Flag de ativação")
    destino: str = Field(min_length=1, description="Nome da nova coluna gerada")
    expressao: dict[str, str] = Field(
        description="Mapeamento de expressões por motor tabular (pandas, polars)"
    )


class ConfiguracaoAtributosNovos(BaseModel):
    """Configurações de engenharia e criação de novos atributos."""

    ativo: bool = Field(default=False, description="Flag geral para criação de atributos")
    expressoes: list[ExpressaoAtributoNovo] = Field(
        default=[], description="Lista de novos atributos"
    )


class ConfiguracaoTabular(BaseModel):
    """Configurações do motor de processamento tabular."""

    motor: Literal["pandas", "polars"] = Field(
        default="pandas", description="Motor tabular utilizado na preparação de dados"
    )


class ItemAnaliseEda(BaseModel):
    """Especificação de um gráfico ou análise exploratória."""

    nome: str = Field(min_length=1, description="Identificador único da análise")
    descricao: str = Field(default="", description="Descrição textual do objetivo da análise")
    ativo: bool = Field(default=True, description="Flag de ativação da análise")
    expressao: str | None = Field(
        default=None,
        description="Expressão Python restrita via AST para geração de atributo ou cálculo",
    )
    tipo_grafico: Literal[
        "histograma", "boxplot", "dispersao", "correlacao", "barras", "tabela"
    ] = Field(description="Tipo de representação visual")
    coluna: str | None = Field(default=None, description="Coluna principal para gráficos univariados")
    coluna_x: str | None = Field(default=None, description="Variável para o eixo X")
    coluna_y: str | None = Field(default=None, description="Variável para o eixo Y")
    cor: str | None = Field(default=None, description="Variável para estratificação por cor")
    titulo: str = Field(default="", description="Título descritivo da visualização")
    nbins: int | None = Field(default=None, description="Quantidade de classes no histograma")
    escala: str | None = Field(default=None, description="Escala de visualização (ex: linear, log)")
    agrupar_por: str | None = Field(default=None, description="Coluna para agrupamento estatístico")
    metricas: list[str] = Field(
        default_factory=list, description="Métricas de agregação a serem calculadas"
    )
    ordenar_por: str | None = Field(default=None, description="Coluna base de ordenação")
    direcao_ordenacao: str | None = Field(
        default=None, description="Direção da ordenação (asc ou desc)"
    )
    limite: int | None = Field(default=None, description="Limite de linhas a exibir")


class ConfiguracaoEda(BaseModel):
    """Configurações de visualização e Análise Exploratória de Dados."""

    motor_grafico: Literal["grafana", "seaborn", "plotly", "altair"] = Field(
        default="grafana", description="Motor de renderização gráfica"
    )
    caminho_relatorio_html: str = Field(
        default="storage-data/relatorio_eda.html", description="Caminho do relatório HTML"
    )
    caminho_relatorio_json: str = Field(
        default="storage-data/relatorio_eda.json", description="Caminho do relatório em JSON"
    )
    analises: list[ItemAnaliseEda] = Field(
        default=[], description="Catálogo de análises configuradas"
    )


class ConfiguracaoImputacaoAtributo(BaseModel):
    """Estratégia de imputação customizada para um atributo específico."""

    ativo: bool = Field(default=True, description="Flag de ativação da imputação customizada")
    estrategia: Literal["mediana", "media", "mais_frequente", "constante"] = Field(
        description="Estratégia aplicada à coluna"
    )


class ConfiguracaoImputacao(BaseModel):
    """Configurações de tratamento de valores ausentes (imputação)."""

    ativo: bool = Field(default=True, description="Flag de ativação da imputação")
    estrategia_numerica: Literal["mediana", "media", "mais_frequente", "constante"] = Field(
        default="mediana", description="Estratégia padrão para atributos numéricos"
    )
    estrategia_categorica: Literal["mais_frequente", "constante"] = Field(
        default="mais_frequente", description="Estratégia padrão para atributos categóricos"
    )
    valor_constante_numerico: float = Field(
        default=0.0, description="Valor sentinela numérico caso estratégia seja constante"
    )
    valor_constante_categorico: str = Field(
        default="Ausente", description="Valor sentinela textual caso estratégia seja constante"
    )
    adicionar_indicador: bool = Field(
        default=False, description="Adiciona coluna indicadora de presença de nulos"
    )
    por_atributo: dict[str, ConfiguracaoImputacaoAtributo] = Field(
        default_factory=dict, description="Estratégias específicas por atributo"
    )


class ConfiguracaoPreprocessamentoAtributos(BaseModel):
    """Configurações de transformação matemática de atributos."""

    ativo: bool = Field(default=True, description="Flag de ativação da transformação de features")
    metodo: Literal["log1p", "log", "yeo-johnson"] = Field(
        default="log1p", description="Método matemático de transformação"
    )
    colunas: list[str] = Field(
        default_factory=list, description="Lista de atributos a transformar"
    )


class ConfiguracaoPreprocessamento(BaseModel):
    """Configurações de transformações de escala e pré-processamento."""

    transformar_alvo: bool = Field(
        default=True, description="Aplica transformação monotônica no alvo com reversão automática"
    )
    metodo_transformacao_alvo: Literal["log1p", "log", "yeo-johnson", "box-cox"] = Field(
        default="log1p", description="Método matemático aplicado à variável alvo"
    )
    transformar_atributos: ConfiguracaoPreprocessamentoAtributos = Field(
        default_factory=ConfiguracaoPreprocessamentoAtributos,
        description="Transformações aplicadas nas variáveis preditoras",
    )


class ItemModelo(BaseModel):
    """Especificação de um modelo de regressão competidor."""

    nome: str = Field(min_length=1, description="Nome canônico do modelo no projeto")
    ativo: bool = Field(default=True, description="Flag de ativação do modelo na bateria de testes")
    pycaret_id: str = Field(min_length=1, description="Identificador nativo do modelo no PyCaret")
    sklearn_class: str = Field(min_length=1, description="Nome da classe equivalente no Scikit-Learn")


type ValorParametro = int | float | str | bool | list[int]
type TecnicaFolds = Literal["kfold", "repeated_kfold", "shuffle_split", "time_series_split"]
type ValorParametroFolds = int | float | bool | str | None


class ConfiguracaoFolds(BaseModel):
    """Configurações da técnica e parâmetros de particionamento em folds."""

    ativo: bool = Field(default=True, description="Flag de ativação da técnica de folds configurada")
    tecnica: TecnicaFolds = Field(
        default="kfold", description="Técnica de particionamento em folds selecionada"
    )
    parametros: dict[str, dict[str, ValorParametroFolds]] = Field(
        default_factory=lambda: {
            "kfold": {"n_splits": 3, "shuffle": True, "random_state": 42},
            "repeated_kfold": {"n_splits": 5, "n_repeats": 3, "random_state": 42},
            "shuffle_split": {"n_splits": 5, "test_size": 0.2, "random_state": 42},
            "time_series_split": {"n_splits": 3, "max_train_size": None},
        },
        description="Mapeamento de parâmetros para cada técnica de folds",
    )


class ConfiguracaoTuning(BaseModel):
    """Configurações de otimização de hiperparâmetros (tuning)."""

    ativo: bool = Field(default=True, description="Flag geral para execução de tuning")
    metodo: Literal["random", "grid", "bayesian", "optuna"] = Field(
        default="random", description="Algoritmo de busca de hiperparâmetros"
    )
    otimizar_metrica: Literal["RMSE", "MAE", "R2", "RMSLE"] = Field(
        default="RMSE", description="Métrica estatística orientadora da otimização"
    )
    iteracoes: int = Field(
        default=10, ge=1, description="Número de iterações ou amostragens da busca"
    )
    manter_melhor: bool = Field(
        default=True, description="Retém o modelo baseline se a busca de tuning resultar em métricas inferiores"
    )
    early_stopping: bool = Field(
        default=False, description="Ativa parada antecipada caso não haja ganho na busca"
    )
    parametros: dict[str, dict[str, list[ValorParametro]]] = Field(
        default_factory=dict, description="Espaços de busca parametrizados por modelo"
    )


class ConfiguracaoMlflow(BaseModel):
    """Configurações de rastreamento de experimentos e registro no MLflow."""

    tracking_uri: str = Field(min_length=1, description="URI do servidor MLflow")
    experimento: str = Field(
        default="imoveis_regressao", description="Nome do experimento no MLflow"
    )
    modelo_registrado: str = Field(
        default="previsao_preco_imoveis", description="Nome do modelo no MLflow Registry"
    )
    modelo_campeao: str = Field(
        default="random_forest", description="Identificador padrão ou fallback do campeão"
    )
    registrar_modelo_campeao: bool = Field(
        default=True, description="Indica se o campeão deve ser versionado no Registry"
    )


class ConfiguracaoModelagem(BaseModel):
    """Configurações completas da etapa de modelagem de machine learning."""

    motor_treino: Literal["pycaret", "sklearn"] = Field(
        default="pycaret", description="Framework de machine learning ativo"
    )
    imputacao: ConfiguracaoImputacao = Field(default_factory=ConfiguracaoImputacao)
    preprocessamento: ConfiguracaoPreprocessamento = Field(
        default_factory=ConfiguracaoPreprocessamento
    )
    modelos: list[ItemModelo] = Field(default=[])
    validacao_cruzada: ConfiguracaoFolds = Field(default_factory=ConfiguracaoFolds)
    tuning: ConfiguracaoTuning = Field(default_factory=ConfiguracaoTuning)
    mlflow: ConfiguracaoMlflow = Field(
        default_factory=lambda: ConfiguracaoMlflow(tracking_uri="http://localhost:5000")
    )


class CvExterna(BaseModel):
    """Configuração da validação cruzada externa de benchmark."""

    metodo: Literal["repeated_kfold", "kfold"] = Field(
        default="repeated_kfold", description="Método da validação cruzada externa"
    )
    splits: int = Field(default=5, ge=2, description="Número de folds")
    repeticoes: int = Field(default=3, ge=1, description="Número de repetições no RepeatedKFold")


class CvInterna(BaseModel):
    """Configuração da validação cruzada interna de tuning."""

    metodo: Literal["kfold"] = Field(
        default="kfold", description="Método da validação cruzada interna"
    )
    splits: int = Field(default=3, ge=2, description="Número de folds internos")
    embaralhar: bool = Field(default=True, description="Embaralha registros antes dos folds")


class ConfiguracaoValidacao(BaseModel):
    """Configuração de particionamento e validação de desempenho."""

    holdout: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Fração retida para holdout intocado"
    )
    cv_externa: CvExterna = Field(default_factory=CvExterna)
    cv_interna: CvInterna = Field(default_factory=CvInterna)
    metrica: Literal["RMSE", "MAE", "R2"] = Field(
        default="RMSE", description="Métrica de otimização e ranqueamento"
    )


class ConfiguracaoEstatistica(BaseModel):
    """Configuração dos testes estatísticos de hipóteses pareadas."""

    teste: Literal["friedman"] = Field(default="friedman", description="Teste não-paramétrico global")
    pos_teste: Literal["nemenyi"] = Field(
        default="nemenyi", description="Pós-teste de comparações pareadas"
    )
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Nível de significância estatística")
    desempate: Literal["menor_rmse", "menor_mae", "maior_r2"] = Field(
        default="menor_rmse", description="Critério determinístico de desempate"
    )


class ConfiguracaoVotacao(BaseModel):
    """Configuração do ensemble e técnicas de votação de regressores."""

    ativo: bool = Field(default=False, description="Flag de ativação da técnica de ensemble")
    tecnica: Literal["media_simples", "voting_regressor", "media_ponderada_rmse", "stacking"] = (
        Field(default="media_simples", description="Estratégia de combinação dos modelos")
    )
    melhores_n: int = Field(default=3, ge=2, description="Quantidade de modelos ranqueados a combinar")
    meta_modelo: Literal["ridge", "linear"] = Field(
        default="ridge", description="Meta-estimador para empilhamento (stacking)"
    )


class ConfiguracaoAvaliacao(BaseModel):
    """Configurações da etapa de avaliação, validação cruzada e seleção."""

    validacao: ConfiguracaoValidacao = Field(default_factory=ConfiguracaoValidacao)
    estatistica: ConfiguracaoEstatistica = Field(default_factory=ConfiguracaoEstatistica)
    votacao: ConfiguracaoVotacao = Field(default_factory=ConfiguracaoVotacao)


class RegraNegocioItem(BaseModel):
    """Especificação individual de uma fórmula de regra de negócio."""

    nome: str = Field(min_length=1, description="Nome identificador da regra")
    ativo: bool = Field(default=True, description="Flag de ativação da regra")
    descricao: str = Field(default="", description="Descrição textual de negócio")
    destino: str = Field(min_length=1, description="Nome do campo retornado no payload")
    expressao: str = Field(min_length=1, description="Fórmula aritmética segura calculada via AST")


class ConfiguracaoServing(BaseModel):
    """Configurações do endpoint de inferência do modelo campeão."""

    host: str = Field(default="0.0.0.0", description="Host de bind do serviço")
    porta: int = Field(default=8080, ge=1024, le=65535, description="Porta TCP do serviço")
    endpoint: str = Field(default="/invocations", description="Endpoint HTTP de inferência")


class ConfiguracaoImplantacao(BaseModel):
    """Configurações da etapa de implantação e pós-processamento de regras."""

    ativo: bool = Field(default=True, description="Flag geral de ativação das regras")
    coluna_alvo_predito: str = Field(
        default="preco_predito", description="Nome da variável com a previsão do modelo"
    )
    regras: list[RegraNegocioItem] = Field(
        default=[], description="Catálogo de regras aplicadas"
    )
    serving: ConfiguracaoServing = Field(default_factory=ConfiguracaoServing)


class ConfiguracaoObservabilidade(BaseModel):
    """Configurações de integração com Grafana e telemetria."""

    grafana_url: str = Field(default="http://localhost:3000", description="URL do servidor Grafana")
    dashboards: list[str] = Field(
        default_factory=lambda: ["eda", "modelos", "estatistica", "servico"],
        description="Lista de dashboards monitorados",
    )
    metricas: list[str] = Field(
        default_factory=lambda: [
            "rmse",
            "mae",
            "r2",
            "latencia_p50",
            "latencia_p95",
            "latencia_p99",
            "taxa_erro",
            "drift",
        ],
        description="Métricas de observabilidade publicadas",
    )


class ConfiguracaoDrift(BaseModel):
    """Configurações de monitoramento de data drift e degradação estatística."""

    ativo: bool = Field(default=True, description="Flag de ativação do monitoramento de drift")
    tipos_monitorados: list[str] = Field(
        default_factory=lambda: [
            "covariate_drift",
            "concept_drift",
            "label_drift",
            "prediction_drift",
        ],
        description="Tipos de drift monitorados",
    )
    limiares: dict[str, float] = Field(
        default_factory=lambda: {
            "covariate_drift_pvalor": 0.05,
            "prediction_drift_ks": 0.10,
            "rmse_degradacao_maxima": 0.15,
        },
        description="Limiares de alerta de drift",
    )


class ConfiguracaoMonitoramento(BaseModel):
    """Configurações consolidadas de observabilidade e monitoramento contínuo."""

    observabilidade: ConfiguracaoObservabilidade = Field(
        default_factory=ConfiguracaoObservabilidade
    )
    drift: ConfiguracaoDrift = Field(default_factory=ConfiguracaoDrift)


class ConfiguracaoCompleta(BaseModel):
    """Esquema consolidado completo do projeto de previsão de preços."""

    projeto: ConfiguracaoProjeto
    fonte: ConfiguracaoFonte
    repositorio_staging: ConfiguracaoRepositorioStaging
    coleta: ConfiguracaoColeta = Field(default_factory=ConfiguracaoColeta)
    tabular: ConfiguracaoTabular = Field(default_factory=ConfiguracaoTabular)
    selecao: ConfiguracaoSelecao
    filtros: ConfiguracaoFiltros = Field(default_factory=ConfiguracaoFiltros)
    transformacoes: ConfiguracaoTransformacoes = Field(default_factory=ConfiguracaoTransformacoes)
    atributos_novos: ConfiguracaoAtributosNovos = Field(default_factory=ConfiguracaoAtributosNovos)
    eda: ConfiguracaoEda = Field(default_factory=ConfiguracaoEda)
    modelagem: ConfiguracaoModelagem
    avaliacao: ConfiguracaoAvaliacao
    implantacao: ConfiguracaoImplantacao = Field(default_factory=ConfiguracaoImplantacao)
    monitoramento: ConfiguracaoMonitoramento = Field(default_factory=ConfiguracaoMonitoramento)
