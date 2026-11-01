# Relatório de Entrega — Etapa 6: Implantação e Regras de Negócio Dinâmicas

**Data de Execução:** 2026-11-01T18:52:49.535792+00:00
**Responsáveis:** @ml_engineer, @ml_architect, @pm
**Status:** CONCLUÍDO COM SUCESSO
**Evidências Persistidas:** [storage-data/relatorio_implantacao.json](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_implantacao.json)
**Run de Promoção no MLflow:** `d26d4a8dd3ef45f28f233562e9a097f0`

---

## 1. Visão Geral da Promoção

O modelo campeão **RIDGE**, validado nas etapas anteriores como superior e selecionado por métricas de generalização, foi promovido para o ambiente de produção.

- **Identificador:** `ridge`
- **Versão:** `1.0.0-champion`
- **Artefato Persistido:** [`/tmp/pytest-of-rodrigo/pytest-23/test_orquestrador_implantacao_0/campeao_ridge/modelo_campeao.joblib`](file:////tmp/pytest-of-rodrigo/pytest-23/test_orquestrador_implantacao_0/campeao_ridge/modelo_campeao.joblib)
- **Metadados Oficiais:** [`/tmp/pytest-of-rodrigo/pytest-23/test_orquestrador_implantacao_0/campeao_ridge/metadados_campeao.json`](file:////tmp/pytest-of-rodrigo/pytest-23/test_orquestrador_implantacao_0/campeao_ridge/metadados_campeao.json)
- **Rastreabilidade no MLflow:** Run `d26d4a8dd3ef45f28f233562e9a097f0` com tags `modelo.status=Production` e `modelo.alias=champion`.

### Métricas de Generalização no Holdout (20% Retido)
- **RMSE:** **R$ 158,973.56**
- **MAE:** **R$ 88,885.25**
- **R²:** **0.8603**

---

## 2. Motor de Regras de Negócio Dinâmicas (AST Seguro)

As regras de negócio configuradas em [configuracoes/06_implantacao_regras.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/06_implantacao_regras.yaml) são consumidas dinamicamente a partir da predição do modelo (`preco_predito`), executando em sandbox restrita sem `eval` inseguro.

### Regras em Vigor e Exemplo de Inferência

**Entrada de Teste:** Apartamento de 85.5m², 3 quartos, 2 banheiros, 2 vagas, Jardim Botânico (Zona Sul).
**Preço Previsto pelo Modelo:** **R$ 2,281,106.15**

| Variável Destino | Regra | Descrição de Negócio | Valor Calculado |
| :--- | :--- | :--- | :---: |
| `preco_minimo_sugerido` | margem_inferior_negociacao | Faixa mínima de negociação (5% abaixo do preço previsto) | **R$ 2,167,050.84** |
| `preco_maximo_sugerido` | margem_superior_negociacao | Faixa máxima de negociação (5% acima do preço previsto) | **R$ 2,395,161.46** |
| `preco_m2_predito` | valor_metro_quadrado_predito | Preço por m² calculado a partir da metragem | **R$ 9,624.92** |
| `valor_comissao_estimada` | taxa_comissao_imobiliaria | Comissão estimada de corretagem imobiliária (6%) | **R$ 136,866.37** |

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
