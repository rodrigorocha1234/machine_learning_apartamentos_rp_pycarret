# Relatório de Entrega — Etapa 4: Modelagem de Regressão e Tuning

**Data de Execução:** 2026-10-26  
**Responsável:** Equipe de Machine Learning & Governança (@ml_engineer, @ml_architect)  
**Status:** CONCLUÍDO COM SUCESSO  
**Governança MLflow:** Ativo em `http://localhost:5000` (Experimento: `imoveis_regressao`)

---

## 1. Visão Geral da Etapa

A **Etapa 4** implementou a infraestrutura desacoplada e configurável de modelagem de machine learning para a previsão de preços de apartamentos em Ribeirão Preto, com:
1. **Padrão GoF Strategy & Factory Method:** Suporte transparente e independente para **PyCaret** nativo e **Scikit-Learn Pipelines** através do contrato [MotorTreinador](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/contratos_modelagem/motor_treinador.py) e da [FabricaTreinadores](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/motores_modelagem/fabrica_treinadores.py).
2. **Blindagem Contra Vazamento de Dados (*Data Leakage*):** Separação estrita de 20% da base tratada (`imoveis_limpos`, 350 registros) em *holdout* antes de qualquer etapa de pré-processamento, imputação, tuning ou validação cruzada.
3. **Governança Hierárquica no MLflow:**
   - **Run Pai:** Nomeado padronizadamente como `tuning_de_parametros_<metodo>` (`tuning_de_parametros_random`), contendo as métricas globais e a referência ao campeão.
   - **Runs Filhos:** Nomeados padronizadamente como `modelo_<nome>_<preprocessamento>` (`modelo_<nome>_log1p`), referenciando explicitamente o `parent_run_id`.

---

## 2. Resumo da Execução Real

- **Origem dos Dados:** Repositório SQLite (`storage-data/staging_imoveis.db`, tabela `imoveis_limpos`).
- **Total de Registros:** 350 imóveis higienizados.
- **Divisão Anti-Leakage:** 280 amostras para treino/CV e 70 amostras reservadas para holdout de validação final.
- **Framework Utilizado:** PyCaret nativo (conforme `04_modelagem_regressao.yaml`).
- **Transformação do Alvo:** `log1p` ativada (mitigação da assimetria do `Valor_da_Venda`).
- **Duração Total do Processo:** 381,31 segundos.

---

## 3. Tabela Comparativa de Performance (Validação Cruzada)

| Modelo | Framework | RMSE (R$) ↓ | MAE (R$) ↓ | R² ↑ | Hiperparâmetros Otimizados | ID Run MLflow |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Lasso (Campeão)** | **PyCaret** | **107.521,12** | **83.778,81** | **0,9395** | `alpha=0.1` | `ce24b0b36a494f21a716d96d3090a322` |
| ElasticNet | PyCaret | 107.620,10 | 83.883,86 | 0,9394 | `alpha=0.01, l1_ratio=0.5` | `4f3dd313078642559150ac590b8d562c` |
| Ridge | PyCaret | 107.696,67 | 83.936,83 | 0,9393 | `alpha=0.1` | `fa199d0a3fa34963806fc77bfe0d5cfc` |
| Regressão Linear | PyCaret | 107.721,62 | 83.981,43 | 0,9393 | `fit_intercept=True` | `29571a8374984cb5ae9ca618ef0510c9` |
| XGBoost | PyCaret | 112.563,70 | 86.046,82 | 0,9336 | `n_estimators=100, max_depth=3, lr=0.1` | `605340588af84fa1aa827263b04772dc` |
| Random Forest | PyCaret | 124.083,72 | 96.136,65 | 0,9194 | `n_estimators=100, max_depth=None` | `84c7ad45f5a64b8baa92032b21ec87f0` |
| LightGBM | PyCaret | 127.181,90 | 98.111,48 | 0,9149 | `n_estimators=100, max_depth=3, lr=0.1` | `56f1cc6d6c2442399bdb841f534ebbf3` |
| SVR | PyCaret | 133.358,88 | 99.428,47 | 0,9062 | `C=10.0, kernel='linear'` | `5ea8bbc2bebd4b5cbf62ca7720aa9634` |
| Árvore de Decisão | PyCaret | 147.551,66 | 112.623,15 | 0,8860 | `max_depth=5, min_samples_split=5` | `145e43da0ad844079012ff838bb8770a` |
| Rede Neural (MLP) | PyCaret | 555.596,07 | 465.366,99 | -0,8068 | `hidden_layer_sizes=(100,), max_iter=500` | `9f7094292fdd4decb79f29009e8afee7` |

---

## 4. Governança e Rastreamento no MLflow

- **URL de Acesso ao Servidor:** [http://localhost:5000](http://localhost:5000)
- **Experimento:** `imoveis_regressao` (ID: `2`)
- **Run Pai:**
  - **ID:** `f09587d44e134c9ab9d7338f77f46942`
  - **Nome:** `tuning_de_parametros_random`
  - **Status:** `FINISHED`
  - **Tags:** `pipeline.etapa: modelagem_regressao`, `pipeline.motor: pycaret`, `pipeline.metodo_tuning: random`, `modelo.campeao: lasso`
  - **Métricas:** `melhor_rmse: 107521.12`, `total_modelos_treinados: 10.0`
- **Runs Filhos:** 10 runs vinculados individualmente com `mlflow.parentRunId = f09587d44e134c9ab9d7338f77f46942`.

---

## 5. Qualidade e Conformidade de Engenharia

- **Tipagem Estrita (Pyright Strict):** **0 errors, 0 warnings, 0 informations** em `app/` e `tests/`. Proibição absoluta de `Any` rigorosamente cumprida.
- **Testes Automatizados (Pytest):** **83 testes executados e aprovados (100% de sucesso)** em 6.10s.
  - [test_fabrica_treinadores.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_fabrica_treinadores.py): Validação do Factory Method GoF.
  - [test_treinador_sklearn.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_treinador_sklearn.py): Pipelines scikit-learn com ColumnTransformer, imputação e RandomizedSearchCV.
  - [test_treinador_pycaret.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_treinador_pycaret.py): Setup, create_model e tune_model nativos.
  - [test_rastreador_mlflow.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_rastreador_mlflow.py): Validação da hierarquia e tags pai/filho no MLflow.
  - [test_orquestrador_modelagem.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_orquestrador_modelagem.py): Orquestração ponta a ponta e blindagem anti-leakage.
- **Convenção de Nomenclatura:** Módulos e pacotes com exatamente duas palavras separadas por `_`.
- **Evidências Persistidas:** [storage-data/relatorio_modelagem.json](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_modelagem.json).
