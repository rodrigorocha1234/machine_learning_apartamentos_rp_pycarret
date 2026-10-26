---
name: configuracao-yaml
description: Projeta e valida os arquivos YAML do pipeline, mantendo configurações centralizadas na classe ConfYAML e verificando consistência entre coleta, preparação, EDA, modelagem, avaliação, API e monitoramento.
---

# Configuração YAML centralizada

## Requisitos
- Centralizar a leitura e validação em `conf_yaml.py`, classe `ConfYAML`.
- Cada etapa deve possuir configuração própria quando isso melhorar clareza, mas a resolução e validação devem permanecer centralizadas.
- Colocar em YAML parâmetros, hiperparâmetros, thresholds, features, target, modelos, scalers, encoders, tuning, CV, drift, métricas, infraestrutura, regras de negócio, expressões e flags.
- Distinguir configuração ausente, valor padrão documentado e erro de configuração.
- Validar schema e referências cruzadas antes de iniciar etapas caras.
- Não permitir que módulos carreguem YAML de forma independente e inconsistente.
- Não armazenar segredos em YAML versionado; usar variáveis de ambiente ou secret manager.
- Fornecer exemplos comentados por etapa e validar YAML com testes.
- Configurar expressões por uma lista permitida de operações, evitando execução arbitrária de código.

## Entregáveis
Estrutura YAML documentada, classe `ConfYAML` tipada, validação de schema, mensagens de erro claras e testes de configuração válida/inválida.

