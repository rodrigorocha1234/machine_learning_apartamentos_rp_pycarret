# Especificação 00 — Previsão de preços de imóveis
Estado: CONCLUÍDO E VALIDADO (Quality Gates Aprovados — Gate de Publicação).
Objetivo: regressão de preço de imóvel em dataset XLSX configurável, processamento tabular por pandas ou Polars, staging SQLite substituível, análises EDA Grafana, comparação experimental e deployment nativo MLflow.
Dataset identificado em `storage-data/imoveis_apartamentos_rp.xlsx` (5.712 registros, 12 colunas, alvo `Valor_da_Venda`). Pipeline implementado, testado com 21 testes unitários/integrados (100% de sucesso) e registrado em `production_artifacts/`.
Gates: especificação humana [OK]; dados válidos [OK]; desenho experimental sem leakage [OK]; seleção estatística [OK]; métricas/segurança [OK]; promoção/serving autorizados [AGUARDANDO CONSENTIMENTO HUMANO].


