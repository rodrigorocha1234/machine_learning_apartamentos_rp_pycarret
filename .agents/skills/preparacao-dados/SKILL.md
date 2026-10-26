---
name: preparacao-dados
description: Implementa preparação tabular configurável com Pandas ou Polars, seleção de colunas, filtros e expressões de transformação/criação de atributos ativáveis por YAML. Use na engenharia e validação dos dados.
---

# Preparação e engenharia de atributos

## Requisitos
- O motor tabular deve ser selecionável no YAML: `pandas` ou `polars`, com desenho extensível.
- Configurar seleção de colunas, expressão de filtro/where, transformações e criação de atributos no YAML.
- Cada expressão de transformação/criação deve ter identificador e flag `enabled`; deve existir uma flag global para habilitar/desabilitar o conjunto.
- Definir explicitamente o contexto e o contrato das expressões. Não executar YAML ou expressões arbitrárias sem validação e controles de segurança.
- Preferir operações vetorizadas no Pandas; evitar `iterrows`, `itertuples` e `apply(axis=1)` quando houver alternativa vetorizada.
- Em Polars, preferir expressões nativas e `LazyFrame` quando apropriado; materializar com `collect()` apenas quando necessário.
- Escolher um formato principal por etapa e evitar conversões repetidas entre Pandas e Polars.
- Preservar schema, tipos, nulos, ordem e semântica ao converter formatos.
- Prevenir data leakage: transformadores aprendidos devem ser ajustados apenas nos dados de treino de cada fold, dentro do pipeline correspondente.

## Fluxo
1. Validar configuração e motor selecionado.
2. Validar colunas referenciadas nas expressões.
3. Aplicar somente regras habilitadas na ordem configurada.
4. Verificar schema e qualidade após cada etapa.
5. Testar equivalência de resultados onde Pandas e Polars implementem a mesma regra.

## Entregáveis
Configuração YAML de exemplo, contratos tipados, testes parametrizados para flags e motores, e documentação das expressões suportadas.

