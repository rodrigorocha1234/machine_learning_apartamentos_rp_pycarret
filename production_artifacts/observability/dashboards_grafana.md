# Especificação e Estrutura dos Dashboards no Grafana
**Papel Responsável:** `@observability`  
**Data:** 2026-10-20  
**Status:** PROJETADO E INTEGRADO AO STACK  

---

## 1. Visão Geral dos Dashboards

Os dashboards foram dimensionados para consumo dos artefatos em `storage-data/` e métricas do Prometheus/MLflow, sem alegar serviços operantes sem evidência:

1. **Dashboard 1: EDA e Qualidade de Entrada (`eda_imoveis.json`):**
   - Distribuição de Preços (histograma e boxplot com percentis p25, mediana e p75).
   - Contagem e proporção de valores ausentes por atributo.
   - Matriz de correlação entre variáveis quantitativas (Metragem, Valor_da_Venda).
   - Preço mediano agrupado por Zona (`preco_mediano_zona`).
2. **Dashboard 2: Comparativo Estatístico de Modelos (`comparativo_modelos.json`):**
   - Tabela de postos médios calculados no teste pareado.
   - P-valor e estatística de teste (Friedman / Wilcoxon).
   - Gráfico de dispersão de resíduos pareados por fold.
   - Destaque do modelo campeão promovido.
3. **Dashboard 3: Serviço MLflow Serving e Observabilidade em Produção (`servico_mlflow.json`):**
   - Throughput de requisições por segundo (RPS).
   - Latência de inferência nos percentis p50, p95 e p99.
   - Taxa de erro HTTP (4xx / 5xx).
   - Monitoramento de drift de entrada nas features principais.
