# Relatório de Entrega — Etapa 6: Implantação e Regras de Negócio Dinâmicas

**Data de Execução:** 2026-11-01T01:00:02.782549+00:00  
**Responsáveis:** @ml_engineer, @ml_architect, @pm  
**Status:** CONCLUÍDO COM SUCESSO  
**Evidências Persistidas:** [storage-data/relatorio_implantacao.json](file:////home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_implantacao.json)  
**Run de Promoção no MLflow:** `cc0ad8d42028439ca65b9c030ee75d6b`  

---

## 1. Visão Geral da Promoção

O modelo campeão **LightGBM** (`LGBMRegressor`), validado na Etapa 5 como superior e de menor rank médio pelo Teste de Friedman, foi promovido para o ambiente de produção.

- **Identificador:** `ridge`
- **Versão:** `1.0.0-champion`
- **Artefato Persistido:** [`/tmp/pytest-of-rodrigo/pytest-21/test_orquestrador_implantacao_0/campeao_ridge/modelo_campeao.joblib`](file:////tmp/pytest-of-rodrigo/pytest-21/test_orquestrador_implantacao_0/campeao_ridge/modelo_campeao.joblib)
- **Metadados Oficiais:** [`/tmp/pytest-of-rodrigo/pytest-21/test_orquestrador_implantacao_0/campeao_ridge/metadados_campeao.json`](file:////tmp/pytest-of-rodrigo/pytest-21/test_orquestrador_implantacao_0/campeao_ridge/metadados_campeao.json)
- **Rastreabilidade no MLflow:** Run `cc0ad8d42028439ca65b9c030ee75d6b` com tags `modelo.status=Production` e `modelo.alias=champion`.

### Métricas de Generalização no Holdout (20% Retido)
- **RMSE:** **R$ 130,971.22**
- **MAE:** **R$ 103,855.22**
- **R²:** **0.9023**

---

## 2. Motor de Regras de Negócio Dinâmicas (AST Seguro)

As regras de negócio configuradas em [configuracoes/06_implantacao_regras.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/06_implantacao_regras.yaml) são consumidas dinamicamente a partir da predição do modelo (`preco_predito`), executando em sandbox restrita sem `eval` inseguro.

### Regras em Vigor e Exemplo de Inferência

**Entrada de Teste:** Apartamento de 85m², 3 quartos, 2 banheiros, 2 vagas, Jardim Botânico (Zona Sul).  
**Preço Previsto pelo Modelo:** **R$ 545,736.18**

| Variável Destino | Regra | Descrição de Negócio | Valor Calculado |
| :--- | :--- | :--- | :---: |
| `preco_minimo_sugerido` | margem_inferior_negociacao | Faixa mínima de negociação (5% abaixo do preço previsto) | **R$ 518,449.37** |
| `preco_maximo_sugerido` | margem_superior_negociacao | Faixa máxima de negociação (5% acima do preço previsto) | **R$ 573,022.99** |
| `preco_m2_predito` | valor_metro_quadrado_predito | Preço por m² calculado a partir da metragem | **R$ 6,420.43** |
| `valor_comissao_estimada` | taxa_comissao_imobiliaria | Comissão estimada de corretagem imobiliária (6%) | **R$ 32,744.17** |

---

## 3. Arquitetura da API HTTP (FastAPI)

A aplicação FastAPI desacopla transporte HTTP, serialização, inferência e cálculo de regras:

- **`GET /health`:** Retorna status operacional, versão da API e do modelo promovido.
- **`POST /predict`:** Recebe especificação do imóvel via `RequisicaoImovel`, retorna previsão e detalhamento das regras.
- **`POST /invocations`:** Endpoint compatível com MLflow Serving para predições em lote e integrações legadas.

---

## 4. Conformidade de Engenharia e Governança

- **Tipagem Estrita (Pyright):** 0 erros, modo estrito preservado sem utilização de `Any`.
- **Validação de Expressões:** Nenhuma expressão fora da whitelist do `ValidadorAst` é aceita.
- **Isolamento de Código:** Módulos e pacotes com exatamente duas palavras separadas por underscore.
