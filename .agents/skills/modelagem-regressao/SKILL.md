---
name: modelagem-regressao
description: Configura e implementa treinamento e tuning de regressão para previsão de preços usando sklearn ou PyCaret nativo, com YAML, validação de dados e registro de experimentos MLflow. Use na etapa de modelagem.
---

# Modelagem e tuning de regressão

## Requisitos
- O YAML escolhe o motor (`sklearn` ou `pycaret`), os modelos, imputação, pré-processamento, técnica de tuning, hiperparâmetros, CV e parâmetros de cada método.
- Usar APIs nativas do framework selecionado. Não inventar wrappers que reimplementem funções já disponíveis; encapsular apenas para adaptar contratos ou orquestrar.
- Modelos requeridos: regressão linear, Lasso, Ridge, regressão múltipla conforme representada por regressão linear com múltiplas features, ElasticNet, árvore de decisão, Random Forest, SVR, rede neural, XGBoost e LightGBM. Verificar dependências e compatibilidade antes de ativar modelos opcionais.
- Imputação e pré-processamento precisam de flags e configurações explícitas.
- O tuning precisa permitir escolher método, espaço de busca, iterações e métrica de otimização via YAML.
- Ajustar imputadores, encoders, scalers e seleção de atributos apenas nos dados de treino de cada fold, dentro do pipeline, para evitar vazamento.
- Separar estratégia de treinamento por framework com Strategy/Adapter quando isso realmente reduzir acoplamento.
- Registrar no MLflow experimentos hierárquicos: pai `tuning_de_parametros_<metodo>` e runs filhos `modelo_<nome_modelo>_<nome_tecnica_preprocessamento>`. Registrar parâmetros, métricas, melhores parâmetros, versões e artefatos.
- Não afirmar que um modelo está disponível sem verificar a instalação e a API da versão estável instalada. Consultar documentação oficial atual das bibliotecas quando necessário.

## Processo
1. Validar configuração e dependências.
2. Criar split de avaliação sem contaminar holdout.
3. Rodar tuning conforme o motor escolhido.
4. Registrar cada execução no MLflow.
5. Testar reprodutibilidade e prevenir leakage.
6. Documentar resultados e limitações.

