# Revisão de Requisitos, Especificações e Registro de Pendências
**Papel Responsável:** `@pm` (Product Manager e Guardião do Escopo)  
**Skill Ativa:** `ciclo-multiagente` / `escrever-specs`  
**Data:** 2026-10-20  
**Estado:** AGUARDANDO APROVAÇÃO HUMANA (Approval Gate)  

---

## 1. Contexto e Objetivo

Conforme determinado em `AGENTS.md` e na skill `ciclo-multiagente`, o pipeline de regressão para previsão de preços de imóveis deve seguir um ciclo rigoroso de engenharia:
$$\text{Requisitos} \longrightarrow \text{Especificação} \longrightarrow \mathbf{Aprovação\ Humana\ (@pm\ Gate)} \longrightarrow \text{Arquitetura} \longrightarrow \text{Implementação} \longrightarrow \text{Validação\ QA} \longrightarrow \text{Publicação}$$

O presente documento consolida a revisão técnica dos requisitos extraídos de `docs/requisitos_originais.txt` (linhas 1411 a 1438 e regras de código 1 a 1410), confrontando-os com `specs/` e `configuracoes/projeto.yaml`, e estabelece o **Approval Gate** formal prévio à implementação do pipeline.

---

## 2. Inventário de Recursos Identificados

- **Dataset Real:** Identificado em `storage-data/imoveis_apartamentos_rp.xlsx` (5.712 linhas, 12 colunas, alvo `Valor_da_Venda`) e base complementar em `dados/bairro_final_v3_engineered.xlsx`.
- **Configuração Centralizada:** `configuracoes/projeto.yaml` contém todas as parametrizações de seleção, filtros, imputação, preprocessamento, CV externa/interna, tuning, estatística (Friedman/Nemenyi), votação, regras de negócio, MLflow e Grafana.
- **Ambiente e Infraestrutura:** Stack Docker Compose existente com PostgreSQL 15, RustFS (S3), MLflow Server 2.16.0+, Prometheus, Loki e Grafana.
- **Virtualenv (.venv):** Python 3.12 com `pycaret 3.3.2`, `scikit-learn 1.4.2`, `pandas 2.1.4`, `polars 2.0.0`, `scipy 1.11.4`, `mlflow 3.17.1`, `pydantic 2.14.1` e `pytest 9.1.1`.

---

## 3. Matriz de Requisitos Funcionais e Conformidade

| ID | Requisito / Declaração | Conformidade / Estratégia Técnica | Papel Líder |
| :--- | :--- | :--- | :--- |
| **RF01** | YAML seleciona fonte XLSX (extensível CSV/SQL/API), caminho, aba e seleção ativável de colunas com tipos e papéis. | `FonteDados` (Protocol) + `CarregadorExcel` inicial. Validador de YAML com interpolação de variáveis de ambiente (`${IMOVEIS_XLSX_PATH}`). | `@data_engineer` |
| **RF02** | YAML define motores tabulares `pandas` ou `polars`; extensível via Strategy/Protocol. | `MotorTabular` (Protocol) com implementações `MotorPandas` (vetorizado) e `MotorPolars`. Proibido loops imperativos (`iterrows`/`itertuples`). | `@data_engineer` |
| **RF03** | Filtros `where` e criação/transformação de colunas configuráveis em YAML. Expressões Python passam por AST restrito e auditável (sem `eval`/`exec`). | Implementação de `AvaliadorAst` com lista positiva de nós (`Compare`, `BinOp`, `UnaryOp`, `Name`, `Constant`), proibindo chamadas abertas, imports, `__subclasses__` ou acesso a SO/rede. | `@qa` / `@data_engineer` |
| **RF04** | Imputação configurável por atributo e pré-processamento por modelo (escalonador/codificador). | Fit de imputadores e encoders **exclusivamente no conjunto de treino** de cada fold para prevenir leakage. | `@data_engineer` / `@ml_engineer` |
| **RF05** | Staging intermediário em SQLite via Repository; análises posteriores obrigatoriamente leem do staging. | `RepositorioDados` (Protocol) + `RepositorioSqlite`. Grava tabela `imoveis_preparados` e serve de fonte para EDA e ML. | `@data_engineer` |
| **RF06** | EDA a partir do staging com expressões flexíveis (whitelisted) e envio/persistência de métricas e relatório para Grafana. | Módulo `analises_eda` extrai estatísticas descritivas, correlações e agregações agrupadas configuradas no YAML. | `@data_engineer` / `@observability` |
| **RF07** | Framework configurável: trilha **PyCaret nativa** OU **scikit-learn nativa**, com isolamento físico rigoroso. | `MotorPycaret` utiliza apenas APIs públicas nativas (`setup`, `create_model`, `tune_model`, `pull`, `finalize_model`). `MotorSklearn` utiliza `Pipeline`, `RandomizedSearchCV` e `cross_validate`. Nenhuma mistura de chamadas. | `@ml_architect` / `@ml_engineer` |
| **RF08** | Hiperparametrização/tuning por modelo conforme YAML; log de parâmetros e métricas no MLflow. | Tuning via busca aleatória nativa do framework escolhido; registro detalhado de execuções no MLflow Tracking. | `@ml_engineer` |
| **RF09** | Validação cruzada configurável (KFold / RepeatedKFold); estimativa de desempenho livre de viés de seleção (nested CV / separação estrita). | Particionamento idêntico entre modelos competidores; fixação prévia de holdout (20%) totalmente intocado. | `@statistician` / `@ml_engineer` |
| **RF10** | Testes de Friedman e pós-teste de Nemenyi aplicados sobre observações pareadas da validação cruzada. | Camada estatística externa (`scipy.stats.friedmanchisquare` + pós-teste de comparações múltiplas pareadas). **Atenção:** Declarado explicitamente que Friedman/Nemenyi NÃO são funções nativas do PyCaret. | `@statistician` |
| **RF11** | Votação / Ensemble configurável (`votacao.ativo`, `melhores_n`, técnica). Desempate determinístico caso inativo. | Se ativo, instancia VotingRegressor com os $N$ melhores ranqueados; se inativo, elege o 1º colocado segundo critério de desempate do YAML (`menor_rmse`). | `@ml_engineer` |
| **RF12** | Avaliação do campeão no holdout retido; retreino final na base completa e registro no MLflow Model Registry. | Holdout avaliado uma única vez. Modelo final treinado em todos os dados apenas no estágio de promoção. | `@ml_engineer` |
| **RF13** | MLflow Serving nativo (`mlflow.pyfunc`) expondo inferência de preço e regras de negócio YAML com fórmulas em expressões Python dinâmicas; sem FastAPI. | Pacote `pyfunc` encapsula modelo campeão e executa `AvaliadorRegras` (`RegraNegocio`) para calcular dinamicamente fórmulas como `preco_predito * 0.95`, `preco_predito / Metragem`, etc. via AST seguro. | `@ml_engineer` / `@devops` |
| **RF14** | Observabilidade e monitoramento no Grafana (EDA, ranking de modelos, latência p50/p95/p99, taxa de erros e drift). | Integração de dashboards Grafana consumindo SQLite, MLflow e métricas de serviço do Prometheus. | `@observability` |

---

## 4. Registro de Pendências e Alinhamentos Técnicos

### Pendência 1: Motor de Avaliação Segura de AST (Segurança Crítica)
- **Risco:** Requisitos originais solicitam "expressões python para where, transformações e atributos".
- **Mitigação Aprovada:** Expressões serão processadas exclusivamente por um parser AST seguro (`AvaliadorAst`), validando uma árvore sintática com whitelist restrita (apenas operadores aritméticos, relacionais e nomes de colunas validadas). Nenhuma execução via `eval()` ou `exec()` irrestritos.

### Pendência 2: Mapeamento de Dependências Estatísticas (Nemenyi)
- **Situação:** O pacote `scikit-posthocs` pode não estar pré-instalado no ambiente.
- **Mitigação Aprovada:** Caso `scikit-posthocs` não esteja disponível, `@statistician` utilizará `scipy.stats` com teste pareado de post-hoc (Wilcoxon com correção de Bonferroni / Holm-Bonferroni ou cálculo exato da estatística de Nemenyi por diferenças de postos médios e valor crítico de studentized range).

### Pendência 3: Resolução de Variáveis de Ambiente no YAML
- **Situação:** O arquivo `configuracoes/projeto.yaml` contém referências como `${IMOVEIS_XLSX_PATH}` e `${STAGING_DATABASE_URL}`.
- **Mitigação Aprovada:** O leitor de configuração interpolará valores do `.env` ou variáveis de ambiente do sistema, aplicando valores padrão quando omitidos.

### Pendência 4: Restrições de Nomenclatura e Tipagem
- **Regras:**
  - Nomes de classes e métodos em português.
  - Nomes de arquivos com **exatamente duas palavras** separadas por `_` (ex.: `fonte_excel.py`, `repositorio_sqlite.py`, `avaliador_ast.py`).
  - Proibição absoluta de `typing.Any` e casts evasivos; Pyright em modo strict.

---

## 5. Ordem de Execução do Ciclo Multiagente Após Aprovação

Uma vez emitida a aprovação humana:
1. **Fase 1 (@ml_architect):** Formalização dos contratos Protocol (`app/contratos_dados/`) e diagrama de componentes em `production_artifacts/ml_architect/`.
2. **Fase 2 (@data_engineer & @qa):** Implementação do carregador de dados, parser AST seguro, motor tabular e repositório SQLite, validados com testes unitários e de contrato.
3. **Fase 3 (@ml_engineer & @statistician):** Implementação dos motores isolados (PyCaret nativo e scikit-learn nativo), validação cruzada pareada, Friedman e pós-teste, e ensemble.
4. **Fase 4 (@qa, @observability & @devops):** Quality gates (Pyright strict + pytest), empacotamento MLflow pyfunc com regras de negócio, e provisionamento de métricas no Grafana.
5. **Fase 5 (Gate de Publicação):** Apresentação das evidências reais em `production_artifacts/` para autorização de deploy.

---

## 6. Declaração do Gate de Aprovação (@pm Approval Gate)

> **ATENÇÃO:**  
> O pipeline encontra-se em estado de **bloqueio deliberado**. Nenhuma alteração arquitetural destrutiva ou execução de treino de modelos em produção será iniciada sem o consentimento formal e explícito do usuário.
