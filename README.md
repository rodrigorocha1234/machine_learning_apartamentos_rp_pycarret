# Skills Google Antigravity — Previsão de preço de apartamentos

Este pacote usa o padrão atual de Agent Skills (`SKILL.md`) e o diretório atual `.agents/skills/`.
Não usa workflows legados nem `.agent/skills`.

## Instalação no workspace
Extraia o ZIP na raiz do projeto. A estrutura `.agents/skills/<nome>/SKILL.md` será descoberta pelo Antigravity.

## Skills incluídas
- `/arquitetura-python`
- `/coleta-persistencia`
- `/preparacao-dados`
- `/eda-visualizacao`
- `/modelagem-regressao`
- `/avaliacao-selecao`
- `/implantacao-regras`
- `/monitoramento-drift`
- `/configuracao-yaml`

## Fonte de requisitos
`resources/requisitos-projeto.md` contém o texto de requisitos fornecido para este projeto.

## Observação sobre "agentes"
A documentação atual descreve estes pacotes como Agent Skills: instruções especializadas carregadas sob demanda. Eles orientam o agente do Antigravity, mas não criam por si só agentes autônomos separados na interface. Use a skill pertinente pelo comando `/nome-da-skill` ou deixe o Antigravity selecioná-la automaticamente.
