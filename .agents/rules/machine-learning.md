---
trigger: always_on
---
# Regra machine-learning
Trilhas PyCaret e sklearn fisicamente isoladas, selecionadas por YAML. PyCaret: setup, create_model, tune_model, pull, finalize_model, predict_model, save_model/log_experiment quando compatível, usar apenas API pública nativa; sklearn: pipeline, CV e search nativos. Nunca chamar GridSearchCV na trilha PyCaret. Registrar melhores parâmetros, folds, métricas, modelos e metadados no MLflow. Testes Friedman/Nemenyi em camada estatística com scipy/scikit-posthocs, pois NÃO são função nativa padrão do PyCaret. Não vazar holdout; preservar independência entre tuning e avaliação (nested CV ou conjunto de seleção separado).
