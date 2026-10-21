# Relatório de Motores de Modelagem e MLflow Serving
**Papel Responsável:** `@ml_engineer`  
**Data:** 2026-10-20  
**Status:** IMPLEMENTADO E TESTADO  

---

## 1. Motores Tabulares e Isolamento Físico

1. **Motor PyCaret (`MotorPycaret`):**
   - Implementado em `app/motores_treino/motor_pycaret.py`.
   - Utiliza estritamente funções nativas da API pública (`RegressionExperiment`): `setup()`, `create_model()`, `tune_model()`, `pull()`, `finalize_model()`.
   - **Inputs Colunares:** Mapeia explicitamente `numeric_features` e `categorical_features` declarados no YAML ou auto-detectados no DataFrame.
   - **Pré-processamento por Modelo e One-Hot Encoding:**
     - Aplica normalização seletiva conforme modelo (`normalize=True, normalize_method="zscore"` para Ridge; sem normalização para Random Forest).
     - Ativa One-Hot Encoding nativo no setup através de `max_encoding_ohe=100`, transformando variáveis categóricas em representação binária esparsa/densa segura.
   - Isolamento total: não importa `sklearn.model_selection` nem realiza GridSearch manual.
2. **Motor Scikit-learn (`MotorSklearn`):**
   - Implementado em `app/motores_treino/motor_sklearn.py`.
   - Utiliza `Pipeline` nativo com `StandardScaler` (ou passthrough) para colunas contínuas e `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` para nominais por modelo.
   - Validação cruzada e hiperparametrização via `RandomizedSearchCV` e `KFold(shuffle=True)`.
   - Isolamento total: não chama nenhuma função do PyCaret.

---

## 2. MLflow Serving e Avaliação Dinâmica de Regras via AST

- **Pacote PyFunc (`ModeloServico`):**
  - Implementado em `app/orquestracao_ml/modelo_servico.py`.
  - Herda de `mlflow.pyfunc.PythonModel`.
  - Executa a inferência do modelo campeão gerando `preco_predito`.
  - Executa dinamicamente o `AvaliadorRegras` para avaliar todas as regras de negócio configuradas no YAML (margem inferior, margem superior, preço por m² e comissão estimada) através do parser AST seguro.
  - Retorna o DataFrame completo enriquecido sem necessidade de FastAPI.
