# Requisitos funcionais
RF01 YAML seleciona fonte xlsx (extensível CSV/SQL/API), caminho e aba, seleção de colunas ativável, cada coluna selecionável.
RF02 YAML define motores tabulares pandas/polars; plugins adicionais via Strategy/Protocol.
RF03 Filtros `where` e criação/transformação de colunas configuráveis (listas vazias aceitas, ativação por item). Expressões Python devem passar por AST segura, restrita e auditável, não eval/exec.
RF04 Imputação por atributo, definição de colunas numéricas/categóricas e preprocessing específico de cada modelo.
RF05 Staging SQLite com porta Repository; EDA lê do staging e disponibiliza resultados para Grafana.
RF06 Cálculo de EDA flexível; expressões com whitelist e funções agregadas registradas; envio de métricas/dashboard Grafana.
RF07 Framework configurável: trilha PyCaret nativa **ou** trilha sklearn nativa, sem mistura.
RF08 Tuning por modelo e parâmetros YAML; registrar parâmetros e resultados MLflow.
RF09 CV configurável por método e número folds; usar melhores parâmetros tuning, com separação experimental que previna viés e leakage.
RF10 Friedman/Nemenyi sobre observações pareadas válidas e divulgação rankings Grafana.
RF11 Ensemble ativável, método extensível e n melhores modelos; caso contrário primeiro colocado segundo regra explícita de desempate.
RF12 Retreino de modelo campeão e registro MLflow; avaliação holdout antes de modelo final treinado em todos os dados.
RF13 MLflow Serving nativo (`mlflow.pyfunc`) expõe o preço previsto pelo modelo campeão acompanhado das regras de negócio cadastradas dinamicamente no YAML com fórmulas em expressões Python avaliadas via AST seguro; wrapper versionado sem FastAPI.
RF14 Observabilidade no Grafana, com histórico, latência, falhas, drift e qualidade de entrada.
