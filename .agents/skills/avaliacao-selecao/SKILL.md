---
name: avaliacao-selecao
description: Executa validação cruzada e comparação estatística de modelos de regressão, incluindo testes de Friedman e pós-teste de Nemenyi, e seleciona candidatos para votação conforme YAML.
---

# Avaliação e seleção

## Requisitos
- Selecionar método de fold, número de folds, repetições, seed e métricas via YAML.
- Reutilizar os melhores parâmetros do tuning sem usar o holdout para seleção.
- Comparar modelos usando os mesmos folds quando necessário para uma comparação justa.
- Aplicar teste de Friedman e pós-teste de Nemenyi sobre resultados comparáveis; documentar hipótese, unidade de comparação, correção e limitações estatísticas.
- Não tratar teste estatístico como substituto de relevância prática, análise de erro ou validação de negócio.
- Configurar no YAML ativação de votação, técnica de ensemble, número `n` de modelos candidatos e parâmetros. Se desativada, não criar ensemble.
- Selecionar candidatos de forma consistente com resultados estatísticos e métricas configuradas; se a evidência não distinguir modelos, informar isso em vez de alegar superioridade.
- Registrar resultados, tabelas e artefatos no destino de observabilidade configurado, inicialmente Grafana via mecanismo de persistência apropriado.
- Preservar holdout como avaliação final isolada.

## Entregáveis
Relatório com métricas por fold, comparação estatística, seleção justificada, testes e registro de artefatos.

