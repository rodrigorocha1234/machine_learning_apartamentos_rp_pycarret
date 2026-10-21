---
trigger: always_on
---
# Regra engenharia-dados
Pandas vetorizado (assign, where, mask, groupby, merge); não iterrows, itertuples ou apply(axis=1) quando vetorização possível. Polars lazy quando vantajoso. Fonte XLSX inicial, adapters extensíveis SQL/API/CSV; staging SQLite inicial e repository substituível; análises devem partir do staging. Imputação/encoding com fit APENAS em treino de cada fold.
