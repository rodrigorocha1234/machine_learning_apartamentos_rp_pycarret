---
name: monitoramento-drift
description: Define observabilidade e monitoramento de qualidade, performance e drift para o sistema de previsão de preços, com métricas configuráveis e destino Grafana substituível.
---

# Observabilidade, performance e drift

## Monitoramentos
- Qualidade dos dados: nulos, tipos, schema, categorias novas e limites de domínio.
- Data/covariate drift: mudanças na distribuição de X; configurar métodos como PSI, KS e distância de Wasserstein conforme tipo de variável e pressupostos.
- Feature drift: mudança em variável específica, tratado como dimensão de data drift quando apropriado.
- Prediction drift: distribuição de `y_pred`, média, mediana e percentis.
- Target/label drift: distribuição de y real quando os rótulos estiverem disponíveis.
- Concept drift/real concept drift: mudança na relação X→Y; acompanhar MAE, RMSE, R² e resíduos quando valores reais estiverem disponíveis.
- Covariance drift: mudanças em relações entre features.
- Virtual drift: mudança em P(X) sem evidência necessária de mudança em P(Y|X); não inferir automaticamente degradação.
- Classificar padrões temporais quando suportado: sudden, gradual, incremental, recurring e blip.
- Data drift ou prediction drift, isoladamente, são sinais de investigação, não prova de perda de desempenho.

## Arquitetura
- Métricas, janelas temporais, thresholds, features e destinos ficam em YAML.
- Separar cálculo de métricas, detecção, alertas e publicação; manter Grafana como destino inicial substituível.
- Não inventar dados de referência; definir baseline de treino e janela de produção explicitamente.
- Respeitar disponibilidade tardia dos rótulos reais e evitar afirmar concept drift sem evidência supervisionada.
- Testar detectores com dados sintéticos controlados, incluindo ausência de drift e mudanças conhecidas.

