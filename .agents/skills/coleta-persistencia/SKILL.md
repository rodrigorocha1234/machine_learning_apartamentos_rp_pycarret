---
name: coleta-persistencia
description: Implementa a coleta de dados históricos de apartamentos a partir de XLSX e persistência intermediária substituível, com validação e padrões de projeto justificados. Use na etapa de ingestão e armazenamento.
---

# Coleta e persistência

## Objetivo
Implementar a etapa de coleta para previsão de preço de apartamentos em Ribeirão Preto, inicialmente a partir de XLSX, com armazenamento intermediário SQLite e possibilidade de substituir fonte ou banco por configuração.

## Regras
- Consulte `resources/requisitos-projeto.md` antes de implementar.
- Configurar caminhos, formato, banco, tabela, schema e opções em YAML; não fixar parâmetros variáveis no código.
- Criar um contrato de carregamento e implementações intercambiáveis por fonte, usando Strategy/Adapter/Factory apenas se trouxerem benefício real.
- Isolar persistência atrás de contrato/repositório para permitir trocar SQLite no futuro.
- Validar schema, tipos, campos obrigatórios e qualidade antes de persistir; registrar métricas de qualidade e erros.
- Não descartar exceções silenciosamente; preservar causa.
- Evitar acoplamento entre leitor XLSX, regras de domínio, banco e orquestração.
- Reexecuções devem ter comportamento definido e documentado; evitar duplicação acidental de registros.

## Entregáveis
- Implementação modular tipada.
- Configuração YAML de fonte e persistência.
- Testes unitários, integração e contrato para carregador e repositório.
- Logs e métricas úteis para diagnosticar falhas.

