# Previsão de Preços de Imóveis (Ribeirão Preto) — Machine Learning & Governança

Sistema corporativo de previsão imobiliária com suporte multi-framework (PyCaret & Scikit-Learn isolados fisicamente), seleção estatística por testes de Friedman/Nemenyi, ensemble com 7 técnicas de votação, processamento tabular duplo (Pandas & Polars vetorizados) e deploy nativo via MLflow Serving com avaliação segura de regras de negócio via AST.

---

## 🚀 Destaques da Arquitetura

1. **Tipagem Estática Estrita (Python 3.12+)**: 100% livre de `typing.Any` e `object`, aderente ao Pyright strict com protocolos semânticos (`ModeloPreditor`, `MotorTreino`, `MotorTabela`, `FonteDados`).
2. **Isolamento de Trilhas ML**: Trilha PyCaret nativa e Trilha Scikit-Learn desacopladas, configuráveis via YAML.
3. **Seleção Estatística Rigorosa**: Teste de Friedman pareado e pós-teste de Nemenyi com cálculo da Diferença Crítica (CD) para eleger o modelo campeão.
4. **7 Técnicas de Votação (Ensemble)**:
   - `media_simples`
   - `voting_regressor` (Scikit-Learn nativo)
   - `media_ponderada_rmse`
   - `media_ponderada_postos` (pesos pelo ranking de Friedman)
   - `mediana`
   - `media_aparada` (Trimmed Mean)
   - `stacking` (StackingRegressor com meta-modelo)
5. **Expressões Python Seguras via AST**: Suporte a expressões declarativas em `configuracoes/projeto.yaml` para filtros, transformações, features e regras de negócio com as sintaxes `df['coluna']` (Pandas) e `pl.col('coluna')` (Polars).
6. **Observabilidade Total**: Dashboard Grafana provisionado cobrindo EDA, latência p50/p95/p99, throughput, postos de Friedman, drift de modelo e logs Loki.

---

## 📂 Estrutura do Projeto

```
├── app/
│   ├── analises_eda/            # Geração de EDA a partir do Staging SQLite
│   ├── analises_estatisticas/   # Testes não-paramétricos de Friedman e Nemenyi
│   ├── configuracao_projeto/    # Leitor de YAML com interpolação de variáveis de ambiente
│   ├── contratos_dados/         # Protocols de interface (SOLID)
│   ├── fontes_dados/            # Adaptadores polimórficos de ingestão (Excel, SQL, API)
│   ├── modelos_votacao/         # RegressorVotacao e FabricaVotacao
│   ├── motores_tabela/          # Motores tabulares vetorizados (Pandas e Polars)
│   ├── motores_treino/          # MotorPycaret e MotorSklearn
│   ├── orquestracao_ml/         # MainPipeline (Facade) e ModeloServico (MLflow PyFunc)
│   ├── regras_negocio/          # Interpretador restrito AST de regras
│   └── repositorios_dados/      # Staging em SQLite
├── config_ob/
│   └── dashboards/              # Dashboard Grafana JSON provisionado
├── configuracoes/
│   └── projeto.yaml             # Configuração centralizada do experimento
├── docs/
│   └── documentacao_projeto.md  # Documentação técnica e estatística detalhada
└── tests/                       # Suíte completa com 48 testes automatizados
```

---

## 🧪 Executando os Testes

```bash
# Testes rápidos (exclui PyCaret demorado)
.venv/bin/pytest tests/ -v -m "not pycaret"

# Testes completos (incluindo PyCaret e Pipeline end-to-end)
.venv/bin/pytest tests/ -v
```

---

## 📖 Documentação Completa

A documentação detalhada do projeto está disponível tanto em formato HTML interativo quanto em Markdown:
* 🌐 **Documentação Interativa (HTML)**: [docs/documentacao_projeto.html](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/docs/documentacao_projeto.html) (ou [docs/index.html](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/docs/index.html))
* 📝 **Documentação Markdown**: [docs/documentacao_projeto.md](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/docs/documentacao_projeto.md)
