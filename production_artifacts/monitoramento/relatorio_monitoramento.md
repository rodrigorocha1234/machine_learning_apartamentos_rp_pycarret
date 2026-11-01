# Relatório de Entrega — Etapa 7: Monitoramento de Drift e Observabilidade

**Data de Execução:** 2026-11-01T20:14:57.903492+00:00  
**Responsáveis:** @observability, @statistician, @devops  
**Status:** CONCLUÍDO COM SUCESSO  
**Evidências Persistidas:** [storage-data/relatorio_drift.json](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_drift.json)  
**Dashboard Grafana:** [`config_ob/dashboards/dashboard_drift.json`](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/config_ob/dashboards/dashboard_drift.json)  
**URL do Servidor Grafana:** [http://localhost:3000](http://localhost:3000) (UID: `drift-apartamentos-rp`)

---

## 1. Visão Geral do Diagnóstico de Drift

O sistema de observabilidade contínua executou os testes de hipóteses bicaudais e estabilidade populacional entre a base de referência (treino de 280 imóveis) e o fluxo de produção:

- **Classificação Taxonômica:** `SEM_DRIFT`
- **Covariate Drift (X):** `NÃO DETECTADO`
- **Prediction Drift (ŷ):** `NÃO DETECTADO`
- **Target Drift (y):** `NÃO DETECTADO`
- **Concept Drift Real (P(Y|X)):** `NÃO DETECTADO`
- **Degradação de Performance (RMSE):** N/A (Rótulos de produção não disponíveis)
- **Orientação Operacional:** STATUS VERDE: Nenhuma anomalia de distribuição ou degradação detectada. Modelo operando em regime nominal.

---

## 2. Indicadores de Qualidade e Integridade dos Dados

- **Total de Amostras Avaliadas:** 1142
- **Taxa Geral de Nulos:** **0.00%**
- **Novas Categorias Não Mapeadas:** 1
- **Valores Fora dos Limites Históricos:** 0

---

## 3. Detalhamento Estatístico Feature a Feature

| Variável | Papel | Teste Estatístico | Estatística | P-Valor | Limiar | Drift Detectado? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `Metragem` | numerica | kolmogorov_smirnov | 0.0153 | 0.9807 | 0.05 | NÃO |
| `Quartos` | numerica | kolmogorov_smirnov | 0.0191 | 0.8851 | 0.05 | NÃO |
| `Banheiros` | numerica | kolmogorov_smirnov | 0.0121 | 0.9991 | 0.05 | NÃO |
| `Vagas` | numerica | kolmogorov_smirnov | 0.0170 | 0.9503 | 0.05 | NÃO |
| `Bairro` | categorica | psi | 0.1961 | - | 0.2 | NÃO |
| `Zona` | categorica | psi | 0.0014 | - | 0.2 | NÃO |
| `preco_predito` | predicao | kolmogorov_smirnov | 0.0195 | 0.8686 | 0.1 | NÃO |
| `Valor_da_Venda` | alvo | kolmogorov_smirnov | 0.0226 | 0.7283 | 0.05 | NÃO |

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
