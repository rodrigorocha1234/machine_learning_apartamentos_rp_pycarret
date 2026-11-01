# Plataforma reutilizável para qualquer problema de regressão machine learning. Estudo de caso: Previsão de preço de apartamentos para a cidade de Ribeirão Preto

**caso:** Previsão de preço de apartamentos para a cidade de Ribeirão Preto

## 1-Introdução
O objetivo desse projeto é propor um plataforma de regressão de machine learning, aplicando os conceitos necessários, para ser reutilizados para qualquer problema de regressão., utilizando códigos python modulares (Padrão de projeto).

## 2 – Tecnologias usadas
- Python;
- Sklearn
- PyCaret
- Docker (Docker compose)
- Mlflow
- Grafana
- Grafana loki
- Grafana Alloy (o coletor open-telemetry do Grafana)
- Google antigravity para orquestrar os agentes

## 3 – Requisitos:
### 3.1- Requisitos Funcionais:

- **RF-01 (Ingestão de Dados):** Carga dos 5.712 registros oficiais de apartamentos de Ribeirão Preto a partir de arquivo Excel (.xlsx) com validação de colunas mandatórias. 
- **RF-02 (Staging Desacoplado):** Persistência intermediária em banco relacional SQLite via padrão Repository e Factory GoF, permitindo substituição transparente por PostgreSQL. 
- **RF-03 (Intercambialidade de Motores Tabulares):** Seleção declarativa via YAML entre motores Pandas e Polars. 
- **RF-04 (Filtros e Transformações com AST Seguro):** Criação de variáveis e filtros baseados em parser AST com lista positiva de nós, eliminando vulnerabilidades de injeção de código ou eval arbitrário. 
- **RF-05 (Análise Exploratória & Visualização):** Cálculo de distribuições estatísticas, correlações e geração de dashboards automatizados para Grafana. 
- **RF-06 (Catálogo de 10 Modelos de Regressão):** Suporte a Regressão Linear, Lasso, Ridge, ElasticNet, Árvore de Decisão, Random Forest, SVR, Rede Neural MLP, XGBoost e LightGBM. 
- **RF-07 (Tuning de Hiperparâmetros & Rastreamento MLflow):** Otimização via GridSearch/RandomSearch com registro de runs aninhados (runs filhos por estimador e fold) no MLflow. 
- **RF-08 (Validação Cruzada & Testes Não-Paramétricos):** Avaliação pareada em K-Fold com aplicação dos testes estatísticos de Friedman e pós-teste de Nemenyi (α=0.05). 
- **RF-09 (Comitê de Votação / Ensemble):** Construção de ensembles (VotingRegressor, média simples, média ponderada por RMSE e Stacking) entre os candidatos estatisticamente indistinguíveis. 
- **RF-10 (Promoção do Campeão & Regras de Negócio):** Avaliação cega em Holdout (20%) e aplicação das 4 regras pós-predição (faixas ±5%, preço/m² e taxa de corretagem de 6%). 
- **RF-11 (API REST de Inferência):** Endpoints desacoplados via FastAPI e MLflow Serving com validação contratual Pydantic na porta 8080. 
- **RF-12 (Monitoramento de Drift & Observabilidade):** Detecção contínua de desvios populacionais via testes de Kolmogorov-Smirnov e Wasserstein, com exportação para Prometheus e Grafana.

### 3-2 – Requisitos Não Funcionais (RNF) — 10 Requisitos
- **RNF-01 (Tipagem Estrita - Type Safety):** Python 3.12+ com proibição estrita de typing.Any no código próprio, atingindo 0 erros no Mypy em 108 arquivos. 
- **RNF-02 (Padrões de Projeto GoF):** Arquitetura orientada a Factory Method, Strategy, Chain of Responsibility, Adapter e Builder. 
- **RNF-03 (Configuração Centralizada & Imutabilidade):** Nenhuma constante hardcoded; 100% dos parâmetros parametrizados nos 7 arquivos YAML sob a classe ConfYAML. 
- **RNF-04 (Segurança Estática):** Avaliação de expressões com whitelist AST, bloqueando acesso a sistema de arquivos, rede ou módulos externos. 
- **RNF-05 (Desempenho & Latência):** Tempo de inferência msP95<50 ms com pipelines vetorizados. 
- **RNF-06 (Isolamento de Infraestrutura):** Orquestração conteinerizada de MLflow, PostgreSQL, Prometheus e Grafana via Docker Compose em rede isolada. 
- **RNF-07 (Testabilidade & Qualidade):** 162 testes automatizados no Pytest com 100% de aprovação. 
- **RNF-08 (Observabilidade Automática):** Atualização periódica dos dashboards do Grafana a cada 10 segundos via volume provisioning. 
- **RNF-09 (Padronização Idiomática):** Nomenclatura corporativa integral em Português do Brasil. 
- **RNF-10 (Idempotência e Resiliência):** Execuções repetidas do pipeline sem corrupção de estado ou inconsistência de tabelas.

## 4 – Uso dos arquivos yaml
Para o uso da plataforma de regressão machine learning ser reaproveitado em outros projetos, os arquivos yaml foram separados de acordo com as etapas de machine learnig.

- **01_coletas_dados.yaml:** Aqui, é configurado as fontes de dados, que podem ser, api, csv, banco de dados:
- **02_preparacao_dados.yaml:** Aqui, é configurado o motor de extração, que podem ser pandas ou polars, seleção de colunas de acordo com o tipo qualitativa ou quantitativa, criação de filtros com expressões python, transformações de atributos e criação de atributos usando expressões python
- **03_analise_eda.yaml:** Criação de gráficos e análises customizadas
- **04_modelagem_regressao.yaml:** Aqui, será responsável por configurar pré-processamento antes do treinamento, como, inputação de valores faltantes, configuração de pré-processamento, quando necessário, de acordo com o tipo de regressão, tuning de parâmetros e validação cruzada (kfold \| repeated_kfold \| shuffle_split \| time_series_split)
- **05_avaliacao_selecao.yaml:** Configuração de técnicas de votação (true e false) com os n melhores modelos, técnicas disponíveis: media_simples , voting_regressor , media_ponderada_rmse , stacking
- **06_implantacao_regras.yaml:** Aqui será configurado as regras de negócio usando o valor predito, quando cadastrar uma nova, basta executar o treinamento para o modelo campeão, a nova regra de negócio, já fica disponível na api do mlfow.
  - Regras de negócios cadastradas:
    - Faixa mínima de negociação (5% abaixo do preço previsto)
    - Faixa máxima de negociação (5% acima do preço previsto)
    - Preço por m² calculado a partir da metragem
    - Comissão estimada de corretagem imobiliária (6%)
- **07_monitoramento_drift.yaml:** Configuração do monitoramento drift usando o grafana 

## 5 – Coleta e uso de atributos
Foi coletado do site de uma imobiliária de Ribeirão Preto e para o treinamento, foi usado os atributos zona, bairro, qtd_quartos , metragem, quantidade de banheiros, quantidade de vagas de garagem

## 6 – Resultados via api do mlflow:
Após a escolha do modelo campeão, o mlflow libera a consulta via api, conforme exemplo abaixo:

```json
{
	"dataframe_records": [
		{
			"Metragem": 56,
			"Quartos": 1,
			"Banheiros": 2,
			"Vagas": 1,
			"Bairro": "Sumarezinho",
			"Zona": "Oeste"
		}
	]
}
```

```json
{
	"predictions": [
		{
			"preco_predito": 389035.91,
			"preco_minimo_sugerido": 369584.11,
			"preco_maximo_sugerido": 408487.71,
			"preco_m2_predito": 6947.07,
			"valor_comissao_estimada": 23342.15
		}
	]
}
```

## 7 – Demonstração do projeto

[![Assistir ao vídeo de demonstração do projeto](https://img.shields.io/badge/🎬%20Assistir%20ao%20vídeo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/M9-cC9Anid8)


[Link do reposítório](https://github.com/rodrigorocha1234/machine_learning_apartamentos_rp_pycarret)
