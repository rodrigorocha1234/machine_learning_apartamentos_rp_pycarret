# Relatório de Entrega — Etapa 5: Avaliação Externa, Validação Pareada e Testes Friedman/Nemenyi

**Data de Execução:** 2026-10-26  
**Responsáveis:** @statistician, @ml_engineer, @ml_architect  
**Status:** CONCLUÍDO COM SUCESSO  
**Evidências Persistidas:** [storage-data/relatorio_avaliacao.json](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_avaliacao.json)

---

## 1. Visão Geral da Etapa

A **Etapa 5** realizou a avaliação comparativa dos 10 algoritmos de regressão previamente ajustados, cumprindo rigorosamente os requisitos do projeto:
1. **Validação Cruzada Pareada Externa:** Esquema `RepeatedKFold` (5 splits × 3 repetições = 15 dobras pareadas) aplicado sobre a partição de treino (280 amostras), garantindo que todos os modelos fossem avaliados exatamente sobre os mesmos dados.
2. **Camada de Inferência Estatística Isolada:** Aplicação do **Teste Não-Paramétrico de Friedman** e do **Pós-teste de Nemenyi**, com cálculo exato da Diferença Crítica (CD - *Critical Difference*) via distribuição *Studentized Range*, sem inventar funções inexistentes no PyCaret.
3. **Avaliação Definitiva em Holdout Cego:** Avaliação isolada sobre o conjunto de 20% retido (70 amostras intocadas), extraindo as métricas de generalização sem qualquer contaminação (*data leakage*): RMSE, MAE, R², MAPE e Mediana do Erro Absoluto (MedAE).
4. **Comitê de Votação (Ensemble):** Mecanismo configurável via YAML ([05_avaliacao_selecao.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/05_avaliacao_selecao.yaml)), com suporte a `media_simples`, `media_ponderada_rmse`, `voting_regressor` e `stacking`.

---

## 2. Teste Estatístico Global de Friedman

- **Hipótese Nula ($H_0$):** Todos os 10 algoritmos de regressão apresentam desempenhos equivalentes em termos de erro pareado (RMSE) nas dobras de validação cruzada.
- **Hipótese Alternativa ($H_1$):** Há diferença estatisticamente significativa no desempenho de pelo menos um dos modelos.
- **Nível de Significância ($\alpha$):** 0,05.

### Resultados de Friedman
- **Estatística $\chi_F^2$:** **111,9891**
- **Graus de Liberdade ($df$):** 9 ($k - 1$)
- **$p$-valor:** **$5,7842 \times 10^{-20}$** ($p < 0,05$)
- **Conclusão:** **$H_0$ REJEITADA com altíssima significância estatística**. Há evidência empírica contundente de diferença de performance entre os modelos avaliados.

### Ranks Médios de Friedman (Menor rank = Melhor desempenho)

| Posição | Modelo | Rank Médio de Friedman | Desempenho Relativo |
| :---: | :--- | :---: | :--- |
| **1º** | **LightGBM** | **1,00** | **Melhor modelo absoluto em todas as 15 dobras** |
| 2º | XGBoost | 2,67 | Excelente consistência em dobras de boosting |
| 3º | Random Forest | 4,07 | Bom desempenho ensemble baseado em bagging |
| 4º | Ridge | 4,67 | Regularização L2 equilibrada |
| 5º | Regressão Linear | 5,07 | Linha de base linear padrão |
| 6º | SVR | 5,33 | Margem linear estrita com kernel linear |
| 7º | ElasticNet | 5,53 | Penalidade mista L1/L2 intermediária |
| 8º | Árvore de Decisão | 7,67 | Alta variância nas dobras pareadas |
| 9º | Lasso | 9,00 | Penalização L1 excessiva nas dobras pareadas |
| 10º | Rede Neural (MLP) | 10,00 | Dificuldade de convergência em base tabular restrita |

---

## 3. Pós-teste de Nemenyi e Diferença Crítica (CD)

Com a rejeição de $H_0$, o teste de Nemenyi foi executado para identificar quais modelos são estatisticamente indistinguíveis do melhor ranqueado:

- **Fator de Escala da Variância:** $SE = \sqrt{\frac{k(k + 1)}{6N}} = \sqrt{\frac{10 \times 11}{6 \times 15}} = 1,1055$
- **Quantil Crítico ($q_{\alpha=0.05, k=10, \infty}$):** 4,4744
- **Diferença Crítica ($CD$):** **3,4976**
- **Modelo Campeão Estatístico:** **LightGBM** (Rank: 1,00)
- **Modelos Equivalentes ao Campeão ($|R_j - 1,00| \le 3,4976$):**
  1. **LightGBM** (Rank 1,00, diferença 0,00)
  2. **XGBoost** (Rank 2,67, diferença 1,67 $\le$ CD)
  3. **Random Forest** (Rank 4,07, diferença 3,07 $\le$ CD)

> **Limitação Estatística Formal:** Embora o LightGBM tenha obtido o rank médio 1,00, a evidência estatística pareada não rejeita a equivalência entre **LightGBM**, **XGBoost** e **Random Forest** ao nível $\alpha = 0,05$. Modelos lineares (Ridge, Lasso, Linear) e Redes Neurais tiveram distância superior à CD ($> 3,50$), sendo estatisticamente superados.

---

## 4. Avaliação Cega no Holdout Isolado (20% Retido — 70 Amostras)

A validação no conjunto de teste cego confirmou a superioridade prática do grupo de modelos selecionado pelo teste estatístico:

| Modelo | RMSE (R$) ↓ | MAE (R$) ↓ | R² ↑ | MAPE (%) ↓ | MedAE (R$) ↓ | Status Estatístico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM (Campeão)** | **84.599,21** | **65.292,49** | **0,9592** | **7,73%** | **52.244,44** | **Campeão Estatístico e Empírico** |
| XGBoost | 94.287,07 | 73.065,94 | 0,9494 | 8,53% | 56.856,75 | Equivalente Estatístico (Top 2) |
| Random Forest | 112.324,92 | 85.132,37 | 0,9281 | 10,01% | 66.449,15 | Equivalente Estatístico (Top 3) |
| SVR | 132.759,68 | 104.452,11 | 0,8996 | 11,77% | 94.411,38 | Estatisticamente Inferior |
| Árvore de Decisão | 134.853,57 | 105.609,78 | 0,8964 | 12,48% | 87.865,96 | Estatisticamente Inferior |
| Ridge | 135.326,46 | 107.464,80 | 0,8957 | 11,93% | 99.185,44 | Estatisticamente Inferior |
| Regressão Linear | 135.775,95 | 107.655,51 | 0,8950 | 11,94% | 99.722,29 | Estatisticamente Inferior |
| ElasticNet | 138.492,45 | 106.047,54 | 0,8907 | 11,66% | 93.384,22 | Estatisticamente Inferior |
| Lasso | 199.688,56 | 159.224,46 | 0,7729 | 18,88% | 112.223,82 | Estatisticamente Inferior |
| Rede Neural (MLP) | 329.839,01 | 229.760,89 | 0,3803 | 23,19% | 151.927,54 | Estatisticamente Inferior |

---

## 5. Seleção Final e Recomendação de Implantação

1. **Configuração de Votação (Ensemble):** Conforme configurado em [05_avaliacao_selecao.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/05_avaliacao_selecao.yaml), `votacao.ativo = false`. Nenhum ensemble sintético foi forçado.
2. **Modelo Final Recomendado:** **LightGBM** (`modelo_lightgbm_log1p`).
   - Apresentou **menor RMSE (R$ 84.599,21)**, **maior R² (0,9592)**, **menor erro percentual (MAPE: 7,73%)** e **menor erro mediano (R$ 52.244,44)**.
   - Venceu todos os folds pareados na validação cruzada (Rank médio = 1,00).
3. **Candidatos de Backup Qualificados:** **XGBoost** e **Random Forest** (elegíveis para *fallback* ou ensemble futuro).

---

## 6. Conformidade de Engenharia e Quality Gates

- **Tipagem Estrita (Pyright Strict):** **0 errors, 0 warnings, 0 informations** em `app/` e `tests/`.
- **Testes Automatizados (Pytest):** **98 testes aprovados (100% de sucesso)** em 7.16s.
  - [test_testador_friedman.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_testador_friedman.py): Friedman, Nemenyi e Diferença Crítica.
  - [test_validador_cruzado.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_validador_cruzado.py): Validação pareada RepeatedKFold.
  - [test_comite_sklearn.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_comite_sklearn.py): VotingRegressor, StackingRegressor, Média Ponderada e Simples.
  - [test_avaliador_holdout.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_avaliador_holdout.py): Métricas de teste cego (RMSE, MAE, R², MAPE, MedAE).
  - [test_orquestrador_avaliacao.py](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/tests/test_orquestrador_avaliacao.py): Orquestração integrada da Etapa 5.
- **Convenção de Nomenclatura:** Módulos e pacotes com exatamente duas palavras separadas por `_`.
