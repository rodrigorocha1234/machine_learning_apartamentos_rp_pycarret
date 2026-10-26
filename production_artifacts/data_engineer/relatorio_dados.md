# Relatório de Engenharia de Dados e Staging (Etapas 1 e 2)
**Papel Responsável:** `@data_engineer`  
**Data:** 2026-10-26  
**Status:** CONCLUÍDO E VALIDADO  

---

## 1. Etapa 1: Coleta e Persistência em Staging

### 1.1 Componentes Implementados e Padrões GoF
1. **Contratos e Protocolos (`app/contratos_dados/`):**
   - `FonteDados` (Strategy/Adapter Protocol): interface tipada para ingestão desacoplada de dados tabulares (`carregar_dados() -> pd.DataFrame`).
   - `RepositorioDados` (Repository Protocol): interface tipada para persistência intermediária em staging (`gravar_dados`, `ler_dados`, `verificar_existencia`).

2. **Ingestão Excel (`FonteExcel` em `app/fontes_dados/fonte_excel.py`):**
   - Suporte a `.xlsx` e `.xls` via `pandas` e `openpyxl`.
   - Tratamento estrito de erros e exceções próprias em português (`ErroFonteExcel`).
   - Rejeição de arquivos inexistentes, extensões inválidas e planilhas vazias.

3. **Fábrica de Fontes (`FabricaFontes` em `app/fontes_dados/fabrica_fontes.py`):**
   - GoF **Factory Method** para criação polimórfica de fontes a partir de `ConfiguracaoFonte` (`01_coleta_dados.yaml`).

4. **Staging SQLite (`RepositorioSqlite` em `app/repositorios_dados/repositorio_sqlite.py`):**
   - Implementa `RepositorioDados` para persistência intermediária desacoplada na tabela `imoveis_preparados`.
   - Modos idempotentes (`substituir=True/False`) e normalização de caminhos/URIs (`sqlite:///`).

5. **Fábrica de Repositórios (`FabricaRepositorios` em `app/repositorios_dados/fabrica_repositorios.py`):**
   - GoF **Factory Method** para instanciação de repositórios a partir de `ConfiguracaoRepositorioStaging`.

6. **Serviço de Coleta (`ColetorImoveis` em `app/servicos_coleta/coletor_imoveis.py`):**
   - Orquestra a extração, validação da variável alvo (`Valor_da_Venda`) e persistência no staging com métricas em `ResultadoColeta`.

### 1.2 Evidência de Execução Real no Staging
- **Total de Registros Persistidos:** 350 apartamentos.
- **Total de Colunas:** 13 atributos.
- **Variável Alvo:** `Valor_da_Venda` presente e íntegra.
- **Tabela Gerada:** `imoveis_preparados` em `storage-data/staging_imoveis.db`.

---

## 2. Etapa 2: Preparação de Dados e Motores Tabulares

### 2.1 Componentes Implementados e Padrões de Projeto
1. **Contrato de Motor Tabular (`MotorTabela` em `app/contratos_dados/motor_tabela.py`):**
   - Protocol GoF **Strategy** que padroniza as operações essenciais de preparação:
     - `filtrar_dados(dados: pd.DataFrame, expressao: str) -> pd.DataFrame`
     - `transformar_dados(dados: pd.DataFrame, expressao: str, destino: str) -> pd.DataFrame`
     - `selecionar_colunas(dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame`
     - `obter_nome_motor() -> str`

2. **Motor Pandas Vetorizado (`MotorPandas` em `app/motores_tabela/motor_pandas.py`):**
   - Avaliação segura via AST (`ast.parse` em modo eval com whitelist de operadores e métodos).
   - Processamento 100% vetorizado sem laços lentos (`iterrows` / `apply`).
   - Suporte a operadores aritméticos, comparações lógicas, slicing de colunas e métodos autorizados (`clip`, `abs`, `round`).

3. **Motor Polars Lazy (`MotorPolars` em `app/motores_tabela/motor_polars.py`):**
   - Avaliação nativa em `pl.LazyFrame` e `pl.Expr` traduzindo a árvore sintática para otimizações de query pushdown.
   - Execução com `.collect().to_pandas()`.

4. **Fábrica de Motores Tabulares (`FabricaMotores` em `app/motores_tabela/fabrica_motores.py`):**
   - GoF **Factory Method** para seleção desacoplada do motor via `tabular.motor` em `02_preparacao_dados.yaml` (`"pandas"` ou `"polars"`).

5. **Serviço Orquestrador de Preparação (`PreparadorDados` em `app/servicos_preparacao/preparador_dados.py`):**
   - Executa a sequência configurável declarada no YAML:
     1. Filtros condicionais (`filtros.expressoes`).
     2. Transformações de atributos existentes (`transformacoes.expressoes`).
     3. Engenharia de novos atributos derivados (`atributos_novos.expressoes`).
     4. Seleção e ordenação de colunas ativas (`selecao.colunas`).
     5. Validação rigorosa de integridade da coluna alvo (`Valor_da_Venda`).
   - Retorna o DataFrame preparado e o modelo Pydantic `ResultadoPreparacao`.

### 2.2 Evidência de Equivalência e Execução Real
Executado sobre os dados do staging (`storage-data/staging_imoveis.db`):

| Métrica | Motor Pandas | Motor Polars | Equivalência |
| :--- | :--- | :--- | :--- |
| **Linhas Originais** | 350 | 350 | Idêntico |
| **Linhas Filtradas** | 350 | 350 | Idêntico |
| **Linhas Descartadas** | 0 | 0 | Idêntico |
| **Colunas Finais** | 10 | 10 | Idêntico |
| **Duração da Execução** | 0.0027 s | 0.0228 s | Ambos sub-segundo |
| **Equivalência Numérica/Booleana** | — | — | **True (100% idênticos)** |

### 2.3 Atributos Preparados e Materializados (`imoveis_limpos`)
Persistidos na tabela `imoveis_limpos` em `storage-data/staging_imoveis.db`:

1. `Valor_da_Venda` (alvo - int64)
2. `Metragem` (preditor - float64)
3. `Quartos` (preditor - int64)
4. `Banheiros` (preditor - int64)
5. `Vagas` (preditor - int64)
6. `Bairro` (preditor - object/string)
7. `Zona` (preditor - object/string)
8. `Valor_Condominio` (preditor - float64)
9. `Valor_Iptu` (preditor - float64)
10. `Area_Privativa` (preditor - float64)

### 2.4 Estatísticas Descritivas do Dataset Preparado
- **Preço Médio de Venda:** R$ 967.101,70 (Mín: R$ 221.700,00 | Máx: R$ 2.038.600,00 | Mediana: R$ 952.350,00)
- **Metragem Média:** 129,51 m² (Mín: 40,98 m² | Máx: 219,84 m² | Mediana: 131,32 m²)
- **Quartos:** Média de 2,61 (1 a 4 quartos)
- **Banheiros:** Média de 2,57 (1 a 4 banheiros)
- **Vagas:** Média de 2,15 (1 a 4 vagas)
- **Valores Ausentes (NaN/Null):** 0 em todas as 10 colunas selecionadas.

---

## 3. Evidência de Quality Gates e Tipagem

- **Pytest:** **55 testes executados e aprovados** com 100% de sucesso em 1.99s.
  - `test_coletor_imoveis.py`: 3 aprovados
  - `test_conf_yaml.py`: 20 aprovados
  - `test_fonte_dados.py`: 6 aprovados
  - `test_motor_pandas.py`: 6 aprovados
  - `test_motor_polars.py`: 8 aprovados
  - `test_preparador_dados.py`: 5 aprovados
  - `test_repositorio_dados.py`: 7 aprovados
- **Pyright:** **0 erros, 0 avisos, 0 informações** em modo `strict` (`typeCheckingMode = "strict"`).
- **Segurança AST:** Testes automatizados confirmam a rejeição de injeção arbitrária (`__import__`, `eval`, `exec`, atributos privados `__class__`).
- **Nomenclatura:** Todos os módulos e pacotes respeitam a regra de exatamente duas palavras em português com tipagem estrita e ausência de `Any`.
