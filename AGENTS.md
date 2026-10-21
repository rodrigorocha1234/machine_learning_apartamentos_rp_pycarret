# Governança do projeto — Previsão de preços imobiliários
O ponto de partida é `specs/00-visao-geral.md`. Ler `specs/`, `configuracoes/`, `.agents/rules/` e as skills relevantes ANTES de programar.
Papéis: @pm decide escopo e aprova specs com humano; @ml_architect estabelece interfaces; @data_engineer implementa dados; @ml_engineer constrói motores isolados; @statistician verifica inferência e desenho experimental; @observability entrega métricas; @qa faz quality gates; @devops integra infraestrutura existente.
Apenas o humano pode aprovar mudança de escopo, contratos, promoção de modelo ou deployment. Não declarar testes executados sem executá-los. Não declarar API operacional sem evidência.
**Fluxo:** requisitos → especificação → aprovação humana → arquitetura → implementação → validação QA → publicação aprovada. Registrar entregas em `production_artifacts/`.
**Regra central:** PyCaret usa suas próprias funções nativas para tuning/CV/finalização; sklearn possui trilha independente. Os testes Friedman/Nemenyi são camada estatística explicitamente separada; não inventar função nativa de PyCaret inexistente.
**Proteção:** execução de expressões Python configuradas é perigosa: somente configuração confiável, parser AST com lista positiva, sem eval/exec arbitrário e sem acesso a imports, arquivos ou rede.
