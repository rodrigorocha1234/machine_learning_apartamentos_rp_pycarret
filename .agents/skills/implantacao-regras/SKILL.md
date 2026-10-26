---
name: implantacao-regras
description: Implementa a publicação do modelo campeão e o cálculo de regras de negócio sobre o preço previsto, com fórmulas configuradas por YAML e resposta da API validada.
---

# Implantação e regras de negócio

## Requisitos
- A saída do modelo campeão (`y_pred`) é a entrada explícita das fórmulas de negócio; demonstrar e testar esse fluxo.
- Regras de negócio devem ser configuráveis no YAML, com identificador, descrição, flag ativa, condições, parâmetros e fórmula.
- Permitir fórmulas tabulares Pandas ou Polars conforme o motor selecionado, mas não executar expressões arbitrárias irrestritas. Preferir DSL/AST validada, operações permitidas e ambiente de execução restrito.
- A resposta da API deve incluir a previsão e os resultados das regras ativas, com esquema tipado, unidades, validação e tratamento de erros.
- Separar inferência, aplicação de regras, serialização e transporte HTTP.
- Versionar o modelo campeão e registrar a versão usada em cada inferência.
- Testar limites, valores nulos, regras desativadas, múltiplas regras e compatibilidade entre motor tabular e fórmulas.

## Entregáveis
Configuração YAML de exemplo, contrato da resposta, implementação testável e testes que provem que o preço previsto alimenta as regras.

