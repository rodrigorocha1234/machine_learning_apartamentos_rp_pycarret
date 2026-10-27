# Relatório de Entrega — Etapa 7: Monitoramento de Drift e Observabilidade

**Data de Execução:** 2026-10-27T00:26:39.848801+00:00  
**Responsáveis:** @observability, @statistician, @devops  
**Status:** CONCLUÍDO COM SUCESSO  
**Evidências Persistidas:** [storage-data/relatorio_drift.json](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_drift.json)  
**Dashboard Grafana:** [`config_ob/dashboards/dashboard_drift.json`](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/config_ob/dashboards/dashboard_drift.json)  
**URL do Servidor Grafana:** [http://localhost:3000](http://localhost:3000) (UID: `drift-apartamentos-rp`)

---

## 1. Visão Geral do Diagnóstico de Drift

O sistema de observabilidade contínua executou os testes de hipóteses bicaudais e estabilidade populacional entre a base de referência (treino de 280 imóveis) e o fluxo de produção:

- **Classificação Taxonômica:** `REAL_CONCEPT_DRIFT`
- **Covariate Drift (X):** `DETECTADO`
- **Prediction Drift (ŷ):** `NÃO DETECTADO`
- **Target Drift (y):** `NÃO DETECTADO`
- **Concept Drift Real (P(Y|X)):** `DETECTADO`
- **Degradação de Performance (RMSE):** **+46.09%**
- **Orientação Operacional:** ALERTA CRÍTICO: Concept Drift Real detectado com degradação de erro RMSE de 46.1%. Recomenda-se acionar pipeline de retreinamento do modelo com dados recentes e reavaliação de hipóteses.

---

## 2. Indicadores de Qualidade e Integridade dos Dados

- **Total de Amostras Avaliadas:** 70
- **Taxa Geral de Nulos:** **0.00%**
- **Novas Categorias Não Mapeadas:** 0
- **Valores Fora dos Limites Históricos:** 3

---

## 3. Detalhamento Estatístico Feature a Feature

| Variável | Papel | Teste Estatístico | Estatística | P-Valor | Limiar | Drift Detectado? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `Metragem` | numerica | kolmogorov_smirnov | 0.0893 | 0.7442 | 0.05 | NÃO |
| `Quartos` | numerica | kolmogorov_smirnov | 0.1750 | 0.0601 | 0.05 | NÃO |
| `Banheiros` | numerica | kolmogorov_smirnov | 0.0357 | 1.0000 | 0.05 | NÃO |
| `Vagas` | numerica | kolmogorov_smirnov | 0.0679 | 0.9503 | 0.05 | NÃO |
| `Bairro` | categorica | psi | 0.2499 | - | 0.2 | **SIM (ALERTA)** |
| `Zona` | categorica | psi | 0.0115 | - | 0.2 | NÃO |
| `Valor_Condominio` | numerica | kolmogorov_smirnov | 0.1000 | 0.6097 | 0.05 | NÃO |
| `Valor_Iptu` | numerica | kolmogorov_smirnov | 0.1143 | 0.4390 | 0.05 | NÃO |
| `Area_Privativa` | numerica | kolmogorov_smirnov | 0.0893 | 0.7442 | 0.05 | NÃO |
| `preco_predito` | predicao | kolmogorov_smirnov | 0.1429 | 0.1921 | 0.1 | NÃO |
| `Valor_da_Venda` | alvo | kolmogorov_smirnov | 0.1036 | 0.5651 | 0.05 | NÃO |

---

## 4. Provisionamento Automático no Grafana

O dashboard `Monitoramento de Drift — Previsão Apartamentos RP` foi gerado e salvo em [`config_ob/dashboards/dashboard_drift.json`](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/config_ob/dashboards/dashboard_drift.json). O provedor automático do Grafana já carrega os painéis interativos:
1. Painel de Status Geral e Alerta Operacional
2. Painel de Cobertura e Contagem de Amostras
3. Gauge de Degradação de Erro RMSE
4. Estatísticas de Integridade e Nulos
5. Tabela Detalhada com Coloração Condicional de Drift

---

## 5. Conformidade de Engenharia e Quality Gates

- **Tipagem Estrita (Pyright):** 0 erros, modo estrito sem `Any`.
- **Testes Automatizados (Pytest):** Cobertura com cenários sintéticos de ausência e indução controlada de drift.
- **Taxonomia Rigorosa:** Distinção explícita entre *Virtual Drift* (apenas $P(X)$ mudou sem impacto em $P(Y|X)$) e *Real Concept Drift* (degradação empírica comprovada).
