# Relatório de Entrega — Etapa 6: Implantação e Regras de Negócio Dinâmicas

**Data de Execução:** 2026-11-01T21:51:48.385557+00:00
**Responsáveis:** @ml_engineer, @ml_architect, @pm
**Status:** CONCLUÍDO COM SUCESSO
**Evidências Persistidas:** [storage-data/relatorio_implantacao.json](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_implantacao.json)
**Run de Promoção no MLflow:** `9ad21d0e76714411a55ee7233a1e6af2`

---

## 1. Visão Geral da Promoção

O modelo campeão **RANDOM_FOREST**, validado nas etapas anteriores como superior e selecionado por métricas de generalização, foi promovido para o ambiente de produção.

- **Identificador:** `random_forest`
- **Versão:** `1.0.0-champion`
- **Artefato Persistido:** [`production_artifacts/campeao/modelo_campeao.joblib`](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/production_artifacts/campeao/modelo_campeao.joblib)
- **Metadados Oficiais:** [`production_artifacts/campeao/metadados_campeao.json`](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/production_artifacts/campeao/metadados_campeao.json)
- **Rastreabilidade no MLflow:** Run `9ad21d0e76714411a55ee7233a1e6af2` com tags `modelo.status=Production` e `modelo.alias=champion`.

### Métricas de Generalização no Holdout (20% Retido)
- **RMSE:** **R$ 148,272.59**
- **MAE:** **R$ 83,738.18**
- **R²:** **0.8785**

---

## 2. Motor de Regras de Negócio Dinâmicas (AST Seguro)

As regras de negócio configuradas em [configuracoes/06_implantacao_regras.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/06_implantacao_regras.yaml) são consumidas dinamicamente a partir da predição do modelo (`preco_predito`), executando em sandbox restrita sem `eval` inseguro.

### Regras em Vigor e Exemplo de Inferência

**Entrada de Teste:** Apartamento de 85.5m², 3 quartos, 2 banheiros, 2 vagas, Jardim Botânico (Zona Sul).
**Preço Previsto pelo Modelo:** **R$ 1,695,461.36**

| Variável Destino | Regra | Descrição de Negócio | Valor Calculado |
| :--- | :--- | :--- | :---: |
| `preco_minimo_sugerido` | margem_inferior_negociacao | Faixa mínima de negociação (5% abaixo do preço previsto) | **R$ 1,610,688.29** |
| `preco_maximo_sugerido` | margem_superior_negociacao | Faixa máxima de negociação (5% acima do preço previsto) | **R$ 1,780,234.43** |
| `preco_m2_predito` | valor_metro_quadrado_predito | Preço por m² calculado a partir da metragem | **R$ 7,153.85** |
| `valor_comissao_estimada` | taxa_comissao_imobiliaria | Comissão estimada de corretagem imobiliária (6%) | **R$ 101,727.68** |

---

## 3. Arquitetura da API HTTP (FastAPI)

A aplicação FastAPI desacopla transporte HTTP, serialização, inferência e cálculo de regras:

- **`GET /health`:** Retorna status operacional, versão da API e do modelo promovido.
- **`POST /predict`:** Recebe especificação do imóvel via `RequisicaoImovel`, retorna previsão e detalhamento das regras.
- **`POST /invocations`:** Endpoint compatível com MLflow Serving para predições em lote e integrações legadas.

---

## 4. Conformidade de Engenharia e Governança

- **Tipagem Estrita (MyPy / Pyright):** 0 erros, modo estrito preservado sem utilização de `Any`.
- **Validação de Expressões:** Nenhuma expressão fora da whitelist do `ValidadorAst` é aceita.
- **Isolamento de Código:** Módulos e pacotes com exatamente duas palavras separadas por underscore.
