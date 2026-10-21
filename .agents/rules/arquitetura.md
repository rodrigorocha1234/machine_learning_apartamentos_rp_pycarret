---
trigger: always_on
---
# Regra arquitetura
SOLID com composição e contratos Protocol; GoF Strategy, Factory, Adapter, Repository e Facade somente quando agregarem valor. `app/` é raiz do código de produto. Decisões de domínio complexas devem usar dispatch/polimorfismo, não cascatas if/elif. Contratos para motores de dados pandas/polars, fornecedores de dados, storage e treinamento. Não usar FastAPI: servir via MLflow nativo.
