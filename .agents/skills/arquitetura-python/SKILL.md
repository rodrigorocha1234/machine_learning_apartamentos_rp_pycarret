---
name: arquitetura-python
description: Define e revisa a arquitetura Python do projeto de previsão de preços, aplicando tipagem estrita, SOLID e padrões GoF justificados. Use ao criar módulos, classes, contratos, pacotes ou refatorar arquitetura.
---

# Arquitetura Python do projeto

Use `resources/requisitos-projeto.md` na raiz do pacote entregue como referência de requisitos.

## Regras obrigatórias
- Python 3.12+; tipagem explícita e estrita; nunca usar `Any`, `typing.Any` ou `from typing import *`.
- Nomes próprios de classes, funções, métodos, atributos, enums, protocolos, serviços, adapters, factories, repositórios, exceções, pacotes e módulos em português. Preserve nomes oficiais de bibliotecas externas.
- Módulos e pacotes próprios devem ter exatamente duas palavras separadas por `_`, exceto arquivos especiais exigidos por ferramentas.
- Não sombrear módulos da biblioteca padrão nem dependências.
- Uma classe principal por arquivo; evitar God Classes e hierarquias profundas.
- Usar `@override` em sobrescritas; importar `override` de `typing` no Python 3.12+. Usar `@final`, `@property` e `@abstractmethod` apenas quando semanticamente justificados.
- Exceções próprias em português; preservar causa com `raise ... from erro`; nunca ignorar erros silenciosamente.
- Priorizar composição, SOLID e baixo acoplamento. Aplicar Strategy, Factory, Adapter, Observer, Repository ou Facade somente quando resolverem um problema real.
- Configuração variável fica em YAML; código implementa comportamento.
- Centralizar leitura/configuração YAML em `conf_yaml.py`, classe `ConfYAML`. Os demais pacotes importam a configuração por esse módulo, sem duplicar leitura de YAML.

## Processo
1. Inspecione a estrutura existente antes de criar ou mover arquivos.
2. Identifique responsabilidades, contratos e dependências.
3. Proponha a menor abstração que preserve extensibilidade.
4. Implemente em arquivos pequenos e coesos.
5. Execute testes e type checker em modo strict disponíveis no repositório.
6. Relate decisões, arquivos alterados, verificações executadas e limitações.

