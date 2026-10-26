---
name: eda-visualizacao
description: Projeta análises exploratórias configuradas por YAML e adaptadores de visualização intercambiáveis, incluindo publicação de métricas e resultados para Grafana. Use para EDA, gráficos e relatórios exploratórios.
---

# EDA e visualização

## Requisitos
- Cadastrar análises EDA em YAML: nome, descrição, flag de ativação, expressão/consulta de cálculo, colunas, tipo de gráfico e opções do gráfico.
- Permitir selecionar motor gráfico por configuração, inicialmente Grafana, com arquitetura extensível para Seaborn, Plotly ou outros motores adequados.
- Não presumir que Grafana seja biblioteca de plotting Python: separar geração de dados/consultas, publicação de métricas e visualização.
- Expressões devem ser validadas e executadas em contexto restrito; não usar `eval` irrestrito.
- Separar cálculo estatístico, especificação do gráfico, adaptador de saída e orquestração.
- Tratar dados ausentes, cardinalidade, escalas e agregações explicitamente.
- Os parâmetros de EDA devem ficar em YAML, não hardcoded.

## Entregáveis
Exemplos YAML de gráficos, contratos de adaptadores, testes de configuração e documentação sobre como adicionar outro motor sem reescrever a EDA.

