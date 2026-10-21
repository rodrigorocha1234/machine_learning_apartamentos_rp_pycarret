# Decisão de Arquitetura e Contratos de Domínio
**Papel Responsável:** `@ml_architect`  
**Data:** 2026-10-20  
**Status:** CONCLUÍDO E HOMOLOGADO  

---

## 1. Visão Geral da Arquitetura (Ports & Adapters)

O sistema adota o padrão Ports & Adapters (Arquitetura Hexagonal), desacoplando o núcleo de negócio dos drivers e frameworks externos:

```text
[Fonte XLSX / API / SQL] ──> [FonteDados Protocol]
                                      │
                                      ▼
                        [MotorTabela Protocol (Pandas/Polars)]
                                      │ (AST Seguro Whitelist)
                                      ▼
                       [RepositorioDados (Staging SQLite)]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
        [AnaliseExploratoria]                    [Separação Treino/Holdout]
        (EDA para Grafana)                                 │
                                            ┌──────────────┴──────────────┐
                                            ▼                             ▼
                                   [MotorPycaret Nativo]        [MotorSklearn Nativo]
                                   (Isolamento Físico)          (Pipeline / CV / Tuning)
                                            │                             │
                                            └──────────────┬──────────────┘
                                                           ▼
                                               [AvaliadorEstatistico]
                                               (Friedman + Pós-Hoc Nemenyi)
                                                           │
                                                           ▼
                                               [Modelo Campeão / Ensemble]
                                                           │
                                                           ▼
                                              [Avaliação Holdout Isolado]
                                                           │
                                                           ▼
                                              [Retreino Final Base Total]
                                                           │
                                                           ▼
                                              [MLflow Registry & Serving]
                                              (pyfunc com RegraNegocio AST)
                                                           │
                                                           ▼
                                              [Observabilidade no Grafana]
```

---

## 2. Contratos Definidos (`app/contratos_dados/`)

| Protocolo | Arquivo | Responsabilidade |
| :--- | :--- | :--- |
| `FonteDados` | `app/contratos_dados/fonte_dados.py` | Ingestão desacoplada de dados tabulares (`carregar_dados`). |
| `MotorTabela` | `app/contratos_dados/motor_tabela.py` | Manipulação tabular vetorizada (`filtrar_dados`, `transformar_dados`, `selecionar_colunas`, `imputar_dados`). |
| `RepositorioDados` | `app/contratos_dados/repositorio_dados.py` | Persistência intermediária em staging SQLite (`gravar_dados`, `ler_dados`). |
| `MotorTreino` | `app/contratos_dados/motor_treino.py` | Interface unificada para trilhas de modelagem isoladas (`ajustar_parametros`, `avaliar_modelos`, `treinar_final`). |
| `RegraNegocio` | `app/contratos_dados/regra_negocio.py` | Execução de fórmulas dinâmicas em regras de negócio sobre previsões do campeão (`aplicar_regra`). |
| `AvaliadorEstatistico` | `app/contratos_dados/avaliador_estatistico.py` | Testes não-paramétricos pareados de Friedman e pós-teste Nemenyi (`testar_diferencas`). |

---

## 3. Diretrizes de Isolamento e Segurança

1. **Separação Físico-Conceitual PyCaret / Sklearn:**
   - O módulo `app/motores_treino/motor_pycaret.py` utiliza exclusivamente a API pública nativa de `RegressionExperiment` (`setup`, `create_model`, `tune_model`, `pull`, `finalize_model`). Não importa `sklearn.model_selection`.
   - O módulo `app/motores_treino/motor_sklearn.py` utiliza `Pipeline`, `StandardScaler`, `OneHotEncoder` e `RandomizedSearchCV`. Não invoca funções do PyCaret.
2. **Segurança de Expressões (AST):**
   - Expressões em filtros `where`, transformações e regras de negócio passam pelo avaliador restrito de AST, impedindo qualquer acesso a funções builtin, chamadas, `__import__` ou I/O.
3. **Anti-Leakage Experimental:**
   - Holdout de 20% é segregado antes de qualquer etapa de modelagem e permanece intocado até a avaliação do modelo campeão selecionado.
