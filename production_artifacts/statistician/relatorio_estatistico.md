# Relatório de Inferência e Desenho Experimental Estatístico
**Papel Responsável:** `@statistician`  
**Data:** 2026-10-20  
**Status:** IMPLEMENTADO E HOMOLOGADO  

---

## 1. Princípios do Desenho Experimental

1. **Segregação Estrita de Holdout:**
   - O conjunto de holdout (20%) é segregado antes de qualquer ciclo de busca de hiperparâmetros ou validação cruzada, permanecendo intocado até a avaliação do modelo campeão selecionado.
2. **Comparabilidade Pareada:**
   - Todos os modelos competidores (Ridge, Random Forest) são treinados e avaliados sobre exatamente a mesma partição de dados (`KFold` com mesma semente pseudo-aleatória).
3. **Independência do PyCaret:**
   - O teste de Friedman e o pós-teste de Nemenyi constituem camada estatística externa explícita (`TesteFriedman` em `app/analises_estatisticas/teste_friedman.py`) utilizando `scipy.stats.friedmanchisquare`, e **não** são funções nativas do PyCaret.

---

## 2. Testes de Hipótese e Critério de Decisão

- **Hipótese Nula ($H_0$):** Todos os algoritmos possuem desempenho equivalente (mesma distribuição de erros pareados entre os folds).
- **Hipótese Alternativa ($H_1$):** Há diferença estatisticamente significativa no desempenho de pelo menos um modelo ($\alpha = 0.05$).
- **Pós-Hoc:** Diferença crítica baseada na distribuição de studentized range multiplicada pelo erro padrão dos postos pareados.
- **Eleição do Campeão:** Modelo que obtiver o menor posto médio; desempate por menor RMSE conforme definido no YAML.
