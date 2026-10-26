# Relatório de Análise Exploratória de Dados e Visualização (Etapa 3)
**Papel Responsável:** `@eda_analyst` / `@ml_architect`  
**Data:** 2026-10-26  
**Status:** CONCLUÍDO E VALIDADO NO GRAFANA  

---

## 1. Escopo e Objetivos da Etapa 3

A Etapa 3 implementa a infraestrutura desacoplada de Análise Exploratória de Dados (EDA) e visualização, atendendo a todos os requisitos da skill `eda-visualizacao` e do arquivo de requisitos:
- Separação rigorosa entre cálculo estatístico, contrato de adaptadores visuais e orquestração.
- Configuração declarativa centralizada em [03_analise_eda.yaml](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/configuracoes/03_analise_eda.yaml).
- Suporte a expressões Python seguras validadas via parser AST com whitelist positiva (ex: cálculo dinâmico de `Preco_M2`).
- Padrão GoF **Strategy** e **Factory Method** para seleção do motor gráfico (`"grafana"`, `"relatorios"`, `"seaborn"`, `"plotly"`).
- Provisionamento automático de dashboards no **Grafana** (porta 3000) e exportação de relatórios HTML auto-contidos e JSON.

---

## 2. Componentes e Padrões GoF Implementados

1. **Contratos de Visualização (`app/contratos_visualizacao/`):**
   - [ResultadoAnaliseEda](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/contratos_visualizacao/adaptador_visualizacao.py): modelo tipado com metadados, métricas escalares (média, mediana, IQR, etc.) e dados tabulares resumidos.
   - [ResultadoVisualizacao](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/contratos_visualizacao/adaptador_visualizacao.py): sumário de status e artefatos produzidos.
   - [AdaptadorVisualizacao](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/contratos_visualizacao/adaptador_visualizacao.py): Protocol (*Strategy*) desacoplando o cálculo da apresentação visual.

2. **Cálculo Estatístico Desacoplado:**
   - [CalculadorEstatistico](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/servicos_eda/calculador_estatistico.py):
     - Histograma com bins, bordas e assimetria amostral.
     - Boxplot por categoria com quartis (Q1, Q2, Q3), IQR, limites de Tukey e contagem de outliers.
     - Dispersão bivariada com correlação linear de Pearson e estratificação por cor.
     - Matriz de correlação linear (Pearson) e extração de pares fortemente correlacionados (|r| >= 0.50).
     - Tabela agregada com estatísticas descritivas por categoria com ordenação e limite.
     - Avaliação segura de expressões via AST (*whitelist* de nós e métodos permitidos).

3. **Adaptadores Concretos e Fábrica:**
   - [AdaptadorGrafana](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/adaptadores_visualizacao/adaptador_grafana.py): gera o dashboard provisionado compatível com Grafana Schema v38 em [dashboard_eda.json](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/config_ob/dashboards/dashboard_eda.json) e métricas em [relatorio_eda.json](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_eda.json).
   - [AdaptadorRelatorios](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/adaptadores_visualizacao/adaptador_relatorios.py): renderiza relatório visual em HTML auto-contido com gráficos vetoriais SVG e tabelas modernas em [relatorio_eda.html](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/storage-data/relatorio_eda.html).
   - [FabricaVisualizadores](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/adaptadores_visualizacao/fabrica_visualizadores.py): GoF *Factory Method* permitindo trocar o motor visual sem modificar o código de análise estatística.

4. **Orquestrador de EDA:**
   - [AnalisadorEda](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/app/servicos_eda/analisador_eda.py): consome dados do staging (`imoveis_limpos`), processa o catálogo de análises e despacha para o motor gráfico.

---

## 3. Evidência de Execução Real e Integração com Grafana

### 3.1 Provisionamento no Grafana
- **Dashboard Provisionado:** [dashboard_eda.json](file:///home/rodrigo/PycharmProjects/machine_learning_apartamentos_rp_pycarret/config_ob/dashboards/dashboard_eda.json)
- **UID no Grafana:** `eda-apartamentos-rp`
- **Título:** `EDA - Previsão de Preços de Apartamentos RP`
- **URL no Grafana:** `http://localhost:3000/d/eda-apartamentos-rp/eda-previsao-de-precos-de-apartamentos-rp`
- **Status da API do Grafana:** Confirmado via `GET /api/search?query=EDA` (ID interno provisionado: `2570120777560064`).

### 3.2 Principais Insights Estatísticos dos 350 Apartamentos
1. **Distribuição do Preço de Venda (`Valor_da_Venda`):**
   - Média: R$ 967.101,70 | Mediana: R$ 952.350,00 | Desvio Padrão: R$ 438.711,88
   - Mínimo: R$ 221.700,00 | Máximo: R$ 2.038.600,00
   - Assimetria (Skewness): 0,3329 (leve assimetria à direita, típica de precificação imobiliária).

2. **Preço por Metro Quadrado (`Preco_M2` - calculado via expressão AST):**
   - Média: R$ 7.491,15/m² | Mediana: R$ 7.378,92/m²
   - Mínimo: R$ 4.712,05/m² | Máximo: R$ 10.741,20/m²

3. **Dispersão e Relação Metragem vs Preço:**
   - Coeficiente de correlação linear de Pearson: **r = 0,9888** (fortíssima correlação positiva direta entre metragem e preço de venda).

4. **Pares com Forte Correlação Linear (|r| >= 0.50):**
   - `Valor_da_Venda` & `Metragem`: r = 0,9888
   - `Valor_da_Venda` & `Valor_Condominio`: r = 0,8475
   - `Valor_da_Venda` & `Valor_Iptu`: r = 0,8625
   - `Valor_da_Venda` & `Vagas`: r = 0,7821
   - `Valor_da_Venda` & `Quartos`: r = 0,7412
   - `Valor_da_Venda` & `Banheiros`: r = 0,7230

---

## 4. Evidência de Quality Gates

- **Pytest:** **70 testes executados e aprovados** com 100% de sucesso (`.venv/bin/pytest -v`).
  - `test_calculador_estatistico.py`: 8 aprovados.
  - `test_adaptadores_visualizacao.py`: 3 aprovados.
  - `test_analisador_eda.py`: 4 aprovados.
  - Testes anteriores (Coleta, Preparação, YAML): 55 aprovados.
- **Pyright (Strict):** **0 erros, 0 avisos, 0 informações**.
- **Regras de Nomenclatura:** Módulos e pacotes com exatamente duas palavras em português com tipagem estrita e ausência de `Any`.
