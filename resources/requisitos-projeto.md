# Requisitos de referência — Previsão de preço de apartamentos

Este arquivo consolida o texto de requisitos fornecido pelo usuário. Consulte-o como fonte de verdade do projeto; não invente requisitos ausentes.

# Nome do projeto: Previsão de preço de apartamento para a cidade de Ribeirão Preto
## Antes: Olhar o docker compose que vai estar em execução:
## Crie arquivos yaml de acordo com a funcionalidades
## Etapa 1: Coleta : 
### Com base nas regras acima, crie uma classe carregador onde vou, inicialmente, carregar do arquivo xlsx, deixe extensível, usando padrões de projetos gof
### Crie um banco intermediário, poder sqlite, use padrões de projeto gof para, se eu quiser trocar o banco no futuro

### Carregue os dados históricos

### Vou usar os motores sklearn e pycarret, no arquivo yaml, deve ter uma opção para escolher o tipo de motor de treinamento, nó código python, deve ser organizado por padrão de projeto GOF, importate, pycarret, deve usar todos os métodos nativas, não misturar ou criar métodos caso já exista. Pycarret deve ser usado a versão estável

## Etapa 2: Preparação de dados

### Permita escolher o métodos de processamento pandas ou polars, quero escolher no arquivo yaml, deixe flexível no código python para adicionar outros tipos de processamentos.


 Regras gerais de Python do projeto

- Utilizar **Python 3.12+** como versão mínima oficial do projeto.

- Usar **tipagem forte e explícita** em todo código próprio.

- É proibido utilizar:

```python
Any
typing.Any
```

- Não utilizar `Any` direta ou indiretamente como forma de contornar problemas de tipagem.

- Preferir recursos modernos de tipagem quando houver necessidade semântica real:

```python
Protocol
TypeVar
Generic
Self
Literal
Final
@final
ClassVar
TypeAlias
@override
@overload
@runtime_checkable
```

- Em Python 3.12+, utilizar:

```python
from typing import override
```

- Não utilizar:

```python
from typing import *
```

- Importar somente os recursos efetivamente utilizados.

---

## Nomenclatura

- Nomes próprios do projeto deverão ser escritos em **português**.

Isso se aplica a:

```text
classes
métodos
funções
atributos
enums
protocolos
serviços
adapters
factories
repositórios
exceções
pacotes
módulos
```

- Bibliotecas externas deverão manter seus nomes oficiais.

Exemplo:

```python
from sklearn.pipeline import Pipeline
```

---

## Módulos e pacotes

- Todo módulo ou pacote Python próprio deverá possuir **exatamente duas palavras**, separadas por `_`.

Exemplos:

```text
registro_logs.py
contrato_dados.py
coletor_metricas.py
configuracao_logs.py
modelos_regressao/
motor_sklearn/
gestao_dados/
```

- Não utilizar hífen em módulos ou pacotes importáveis.

Evitar:

```text
registro-logs.py
```

- Exceções são permitidas quando o Python ou alguma ferramenta exigir nomes específicos:

```text
__init__.py
__main__.py
conftest.py
```

---

## Não sombrear bibliotecas

Não criar módulos ou pacotes próprios com nomes iguais aos da biblioteca padrão ou dependências externas.

Evitar:

```text
logging.py
json.py
typing.py
pathlib.py
asyncio.py
enum.py
dataclasses.py
statistics.py
random.py
collections.py
itertools.py
functools.py

pandas.py
numpy.py
sklearn.py
mlflow.py
requests.py
pyspark.py
```

---

## Métodos

- Métodos deverão representar ações claras.

Preferir:

```python
carregar_dados()
validar_schema()
separar_holdout()
treinar_modelo()
calcular_metricas()
registrar_resultado()
verificar_drift()
selecionar_modelo()
```

- Evitar nomes excessivamente genéricos:

```python
run()
execute()
process()
handle()
```

quando existir uma ação de domínio mais clara.

---

# Encapsulamento

O projeto deverá utilizar encapsulamento de forma consistente.

Em Python, os níveis de acesso são representados principalmente por convenções de nomenclatura.

## Atributos públicos

Atributos públicos podem ser acessados diretamente quando fizer sentido expô-los como parte da interface da classe.

Exemplo:

```python
class ResultadoModelo:
    def __init__(self, nome_modelo: str) -> None:
        self.nome_modelo = nome_modelo
```

Usar atributos públicos somente quando seu acesso direto não violar invariantes ou regras de domínio.

---

## Atributos protegidos

Atributos destinados ao uso interno da classe e de subclasses deverão utilizar um `_` inicial.

Exemplo:

```python
class ModeloBase:
    def __init__(self, nome_modelo: str) -> None:
        self._nome_modelo = nome_modelo
```

A convenção:

```text
_nome
```

significa:

```text
uso interno/protegido por convenção
```

Não significa proteção real em runtime.

---

## Atributos privados

Quando houver necessidade real de evitar colisões em subclasses ou proteger detalhes internos de implementação, utilizar `__`.

Exemplo:

```python
class GerenciadorModelo:
    def __init__(self, versao: str) -> None:
        self.__versao = versao
```

Python aplica *name mangling*, transformando internamente o atributo.

Entretanto, isso não representa privacidade absoluta como em Java ou C#.

Utilizar atributos privados somente quando houver justificativa de encapsulamento.

Não transformar todos os atributos em privados indiscriminadamente.

---

# Métodos públicos

Métodos públicos representam operações oferecidas oficialmente pela classe.

Exemplo:

```python
class TreinadorModelo:
    def treinar_modelo(self) -> None:
        ...
```

São a interface principal utilizada por outros componentes.

---

# Métodos protegidos

Métodos auxiliares destinados à própria classe ou subclasses deverão utilizar `_`.

Exemplo:

```python
class TreinadorBase:

    def treinar_modelo(self) -> None:
        dados = self._preparar_dados()
        self._ajustar_modelo(dados)

    def _preparar_dados(self) -> pd.DataFrame:
        ...

    def _ajustar_modelo(
        self,
        dados: pd.DataFrame,
    ) -> None:
        ...
```

---

# Métodos privados

Métodos estritamente internos poderão utilizar `__`.

Exemplo:

```python
class ValidadorDados:

    def validar_dados(self) -> bool:
        return self.__validar_regras_internas()

    def __validar_regras_internas(self) -> bool:
        ...
```

Utilizar apenas quando houver necessidade real.

Preferir métodos protegidos para extensibilidade quando subclasses puderem precisar sobrescrever o comportamento.

---

# Properties

Quando o acesso a um atributo exigir:

```text
validação
transformação
controle
cálculo
proteção de invariantes
```

utilizar `@property`.

Exemplo:

```python
class ConfiguracaoModelo:
    def __init__(self, nome_modelo: str) -> None:
        self._nome_modelo = nome_modelo

    @property
    def nome_modelo(self) -> str:
        return self._nome_modelo
```

Quando houver alteração controlada:

```python
class ConfiguracaoModelo:
    def __init__(self, limite: float) -> None:
        self._limite = limite

    @property
    def limite(self) -> float:
        return self._limite

    @limite.setter
    def limite(self, valor: float) -> None:
        if valor < 0:
            raise ValueError("O limite não pode ser negativo.")

        self._limite = valor
```

Não criar getters e setters artificiais no estilo:

```python
get_nome()
set_nome()
```

quando uma `property` representar melhor o contrato.

---

# Uma classe principal por arquivo

Preferir:

```text
uma classe principal por arquivo .py
```

Exceções podem existir para:

```text
dataclasses pequenas
Enums relacionados
estruturas fortemente coesas
```

Evitar:

```text
God Classes
classes gigantes
arquivos com dezenas de responsabilidades
```

---

# SOLID

Aplicar de forma prática:

```text
Single Responsibility Principle
Open/Closed Principle
Liskov Substitution Principle
Interface Segregation Principle
Dependency Inversion Principle
```

Não criar interfaces, factories ou abstrações apenas para demonstrar padrões de projeto.

Toda abstração deverá resolver um problema concreto.

---

# Composição e herança

Preferir:

```text
composição
```

quando não houver relação clara de:

```text
"é um"
```

Utilizar herança quando existir:

```text
contrato compartilhado
relação semântica apropriada
Template Method
comportamento comum legítimo
```

Evitar hierarquias profundas.

---

# Classes abstratas

Quando herança fizer parte do design, utilizar:

```python
from abc import ABC, abstractmethod
```

Exemplo:

```python
class EstrategiaModelo(ABC):

    @abstractmethod
    def criar_modelo(self) -> RegressorMixin:
        ...
```

Não utilizar permanentemente:

```python
raise NotImplementedError
```

como substituto de `@abstractmethod`.

---

# @override

Sempre utilizar `@override` quando um método sobrescrever explicitamente um método da classe base.

Exemplo:

```python
from typing import override


class ModeloRidge(EstrategiaModelo):

    @override
    def criar_modelo(self) -> Ridge:
        ...
```

Isso ajuda o type checker a detectar mudanças incompatíveis.

---

# @final

Utilizar o decorador `@final` para indicar classes que não devem ser herdadas e métodos que não devem ser sobrescritos por subclasses.

Em Python 3.12+, importar diretamente de `typing`:

```python
from typing import final
```

## Classes finais

Aplicar `@final` quando uma classe possuir uma implementação completa e não fizer sentido permitir sua especialização por herança.

Exemplo:

```python
from typing import final


@final
class CalculadoraMetricas:

    def calcular_rmse(
        self,
        valores_reais: list[float],
        valores_previstos: list[float],
    ) -> float:
        ...
```

Regras:

- Utilizar `@final` quando a classe não fizer parte de uma hierarquia extensível.
- Não aplicar `@final` em classes abstratas destinadas à herança.
- Não aplicar indiscriminadamente em todas as classes concretas.
- Preferir composição quando for necessário reutilizar funcionalidades de uma classe final.

## Métodos finais

Aplicar `@final` quando um método herdado não puder ter seu comportamento alterado por subclasses.

Exemplo:

```python
from typing import final, override


class TreinadorBase:

    def validar_modelo(self) -> bool:
        ...


class TreinadorRidge(TreinadorBase):

    @final
    @override
    def validar_modelo(self) -> bool:
        return True
```

Nesse exemplo, subclasses de `TreinadorRidge` não deverão sobrescrever `validar_modelo()`.

## Integração com `@override`

Quando um método sobrescrever outro da classe base e também precisar impedir novas sobrescritas, utilizar ambos os decoradores:

```python
@final
@override
def validar_modelo(self) -> bool:
    ...
```

## Validação estática

O decorador `@final` é utilizado principalmente por verificadores estáticos de tipos, como Pyright e mypy.

**Importante:** `@final` não impede herança ou sobrescrita durante a execução normal do Python. A restrição deve ser verificada pelo type checker configurado em modo `strict`.

## Regra arquitetural

Utilizar os decoradores conforme a responsabilidade:

| Decorador | Aplicação |
|---|---|
| `@final` | Impedir herança de classes ou sobrescrita de métodos no contrato estático. |
| `@override` | Declarar explicitamente a sobrescrita de um método herdado. |
| `@abstractmethod` | Exigir implementação de métodos em subclasses concretas. |
| `@property` | Controlar acesso, validação ou transformação de atributos. |
| `@staticmethod` | Definir operações que não dependem do estado da instância ou da classe. |
| `@classmethod` | Definir operações relacionadas à classe, incluindo métodos de fábrica. |

**Princípio:** utilizar `@final` para proteger contratos arquiteturais que não devem ser estendidos, sem prejudicar a extensibilidade legítima do projeto.

---

# Exceções

Criar exceções próprias em português.

Exemplo:

```python
class ErroCarregamentoDados(RuntimeError):
    pass
```

Preservar a exceção original:

```python
try:
    ...
except OSError as erro:
    raise ErroCarregamentoDados(
        "Não foi possível carregar os dados."
    ) from erro
```

Evitar:

```python
except Exception:
    pass
```

Nunca ignorar erros silenciosamente.

Evitar capturar `Exception` de forma indiscriminada.

---

# Configuração

Toda configuração variável deverá vir de arquivos YAML.

Evitar hardcoded de:

```text
parâmetros
hiperparâmetros
thresholds
features
target
modelos
scalers
encoders
tuning
validação cruzada
drift
métricas
infraestrutura
regras de negócio
```

O código Python deverá implementar comportamento.

O YAML deverá definir configuração.

---

# Pandas

Priorizar processamento vetorizado.

Evitar:

```python
iterrows()
itertuples()
apply(axis=1)
```

quando houver solução vetorizada.

Evitar:

```text
for sobre linhas
for sobre índices
atribuição célula a célula
transformação manual registro por registro
```

A ordem preferencial é:

```text
operações vetorizadas
↓
Series.str / Series.dt / Series.cat
↓
assign / where / mask
↓
numpy.where / numpy.select
↓
groupby.agg / groupby.transform
↓
merge / join
↓
isin / between
↓
fillna / replace / clip
```

Exemplo incorreto:

```python
for indice, linha in dataframe.iterrows():
    dataframe.loc[indice, "preco_m2"] = (
        linha["Valor_da_Venda"] / linha["Metragem"]
    )
```

Preferir:

```python
dataframe = dataframe.assign(
    preco_m2=dataframe["Valor_da_Venda"].div(
        dataframe["Metragem"]
    )
)
```

---

# Polars

Utilizar `polars` quando o processamento tabular se beneficiar de execução eficiente, avaliação lazy, paralelismo ou redução do consumo de memória.

A escolha entre `pandas` e `polars` deverá considerar o volume dos dados, as operações necessárias, a compatibilidade com as bibliotecas do projeto e a clareza do código. Não migrar automaticamente um fluxo para Polars sem benefício técnico demonstrável.

## Regras de uso

- Manter os nomes oficiais da biblioteca externa: `polars`, `pl.DataFrame`, `pl.LazyFrame` e demais APIs.
- Preferir operações nativas e expressões Polars a loops Python sobre linhas.
- Preferir `select`, `with_columns`, `filter`, `group_by`, `agg`, `join`, `when/then/otherwise` e expressões `pl.col(...)` quando forem adequadas.
- Preferir `LazyFrame` e a cadeia de operações lazy para fluxos tabulares extensos, permitindo ao otimizador planejar a execução.
- Utilizar `collect()` somente no ponto em que seja necessário materializar o resultado de uma consulta lazy.
- Usar `scan_csv`, `scan_parquet` ou outras funções de leitura lazy compatíveis com a fonte quando isso evitar carregar os dados antecipadamente.
- Evitar converter repetidamente entre `pandas.DataFrame`, `polars.DataFrame`, listas e dicionários sem necessidade.
- Não presumir que uma operação lazy será sempre mais rápida; medir o desempenho no fluxo real.
- Verificar a compatibilidade de cada modelo, transformador e ferramenta de Machine Learning antes de escolher Polars como formato de entrada.
- Não usar Polars como justificativa para ignorar validação de schema, tipagem, qualidade dos dados ou prevenção de data leakage.

## Exemplo vetorizado

```python
import polars as pl

dados = pl.DataFrame(
    {
        "Valor": [250_000.0, 320_000.0, 410_000.0],
        "Metragem": [70.0, 80.0, 100.0],
    }
)

dados = dados.with_columns(
    (pl.col("Valor") / pl.col("Metragem")).alias("preco_m2")
)
```

## Exemplo com avaliação lazy

```python
import polars as pl

consulta = (
    pl.scan_parquet("dados/imoveis.parquet")
    .filter(pl.col("Metragem") > 0)
    .with_columns(
        (pl.col("Valor") / pl.col("Metragem")).alias("preco_m2")
    )
    .group_by("Bairro")
    .agg(
        pl.col("Valor").median().alias("mediana_valor"),
        pl.col("preco_m2").mean().alias("media_preco_m2"),
    )
)

resultado = consulta.collect()
```

## Integração com Pandas e Machine Learning

- Escolher um formato de dados principal por etapa do pipeline, evitando conversões desnecessárias.
- Converter para Pandas apenas quando uma biblioteca ou API exigir esse formato, ou quando a conversão tiver sido validada como apropriada.
- Confirmar que a conversão preserva tipos, valores ausentes, categorias, ordem das colunas e semântica dos dados.
- Ajustar imputadores, encoders, scalers e demais transformadores exclusivamente nos dados de treino de cada fold.
- Em validação cruzada, construir e ajustar o pré-processamento dentro do pipeline de cada fold para evitar vazamento de dados.
- Testar resultados e métricas após qualquer mudança de Pandas para Polars; não assumir equivalência apenas porque as operações parecem semelhantes.

---

# Loops Python

Loops não são proibidos no projeto.

Podem ser utilizados para:

```text
modelos
folds
strategies
experimentos
arquivos
configurações
objetos
componentes de infraestrutura
```

A restrição aplica-se principalmente ao processamento tabular linha a linha no pandas quando houver solução vetorizada.

---

# map

`map()` poderá ser utilizado quando representar de forma clara uma transformação funcional.

Exemplo:

```python
nomes_normalizados = list(
    map(str.lower, nomes_modelos)
)
```

Preferir `map()` principalmente quando:

```text
a transformação já existe como função
não há necessidade de lógica complexa
a intenção permanecer clara
```

Não utilizar `map()` apenas para evitar uma compreensão mais legível.

---

# filter

`filter()` poderá ser utilizado para filtrar iteráveis quando houver um predicado claro.

Exemplo:

```python
modelos_ativos = list(
    filter(
        verificar_modelo_ativo,
        configuracoes_modelos,
    )
)
```

Para estruturas pandas, continuar priorizando:

```python
loc
query
isin
between
where
mask
```

em vez de converter dados para listas e usar `filter()`.

---

# reduce

Quando for necessário realizar uma redução funcional, utilizar:

```python
from functools import reduce
```

Exemplo:

```python
resultado = reduce(
    combinar_resultados,
    resultados_modelos,
)
```

Usar `reduce()` somente quando representar claramente uma composição/redução sucessiva.

Para operações simples, preferir funções nativas:

```python
sum()
min()
max()
all()
any()
```

---

# zip

Utilizar `zip()` para percorrer iteráveis relacionados em paralelo.

Exemplo:

```python
for nome_modelo, resultado in zip(
    nomes_modelos,
    resultados,
    strict=True,
):
    ...
```

Em Python moderno, quando os iteráveis deveriam obrigatoriamente possuir o mesmo tamanho, preferir:

```python
zip(..., strict=True)
```

Isso permite detectar inconsistências silenciosas.

---

# zip_longest

Quando os iteráveis puderem possuir comprimentos diferentes e isso for parte intencional da regra, utilizar:

```python
from itertools import zip_longest
```

Exemplo:

```python
for esperado, obtido in zip_longest(
    valores_esperados,
    valores_obtidos,
    fillvalue=None,
):
    ...
```

Não utilizar `zip_longest()` quando diferenças de tamanho representarem erro. Nesse caso, preferir:

```python
zip(..., strict=True)
```

---

# itertools

Preferir ferramentas da biblioteca padrão quando reduzirem código manual e deixarem a intenção clara.

Recursos úteis incluem:

```python
chain
combinations
combinations_with_replacement
permutations
product
islice
takewhile
dropwhile
groupby
cycle
repeat
accumulate
pairwise
zip_longest
```

Exemplos de aplicação no projeto:

```text
combinações de modelos
combinações de parâmetros
encadeamento de coleções
processamento lazy
comparações entre modelos
criação de pares
geração de produtos cartesianos
```

Exemplo:

```python
from itertools import combinations

pares_modelos = combinations(
    nomes_modelos,
    2,
)
```

Evitar materializar iteradores em listas sem necessidade.

---

# Iteradores e avaliação lazy

Quando possível, preservar processamento lazy.

Preferir:

```python
iterador = map(...)
```

quando não for necessário criar imediatamente uma lista.

Converter para:

```python
list(...)
tuple(...)
```

somente quando o consumidor exigir materialização.

Isso é especialmente importante para conjuntos grandes de dados que não estejam sendo manipulados por pandas.

---

# collections

Utilizar estruturas especializadas de `collections` quando representarem melhor o problema.

Recursos recomendados:

```python
Counter
defaultdict
deque
ChainMap
namedtuple
```

Usar somente quando houver vantagem sobre estruturas básicas.

---

# Counter

Utilizar:

```python
from collections import Counter
```

para contagem de elementos em iteráveis Python.

Exemplo:

```python
frequencias = Counter(nomes_modelos)
```

Para colunas pandas, normalmente preferir:

```python
serie.value_counts()
```

---

# defaultdict

Utilizar:

```python
from collections import defaultdict
```

quando houver necessidade legítima de valores padrão durante agrupamento ou acumulação.

Exemplo:

```python
resultados: defaultdict[str, list[float]] = defaultdict(list)
```

Não utilizar quando `dict` normal oferecer contrato mais explícito.

---

# deque

Utilizar:

```python
from collections import deque
```

quando houver necessidade de:

```text
fila
pilha
buffer
janela deslizante
inserções e remoções eficientes nas extremidades
```

Evitar `list.pop(0)` para implementação frequente de filas.

---

# ChainMap

Utilizar:

```python
from collections import ChainMap
```

quando for necessário representar múltiplas camadas de configuração sem copiar todos os dicionários.

Possíveis casos:

```text
configuração padrão
configuração de ambiente
configuração específica do modelo
override de execução
```

Utilizar apenas se o comportamento de precedência estiver claramente documentado.

---

# namedtuple

`namedtuple` poderá ser utilizado para estruturas leves e imutáveis.

Entretanto, quando houver regras de domínio, validação, tipos mais ricos ou comportamento associado, preferir:

```python
@dataclass(frozen=True)
```

ou uma classe de domínio dedicada.

---

# Compreensões

Compreensões são permitidas quando simples e legíveis:

```python
modelos_ativos = [
    nome
    for nome in nomes_modelos
    if nome in configuracoes
]
```

Evitar compreensões profundamente aninhadas ou contendo múltiplas regras de negócio.

Quando a lógica crescer, extrair para uma função com nome semântico.

---

# Generators e iterators

Utilizar `generators` e `iterators` para percorrer sequências de forma incremental, especialmente quando não for necessário manter todos os elementos simultaneamente em memória.

## Iterators

Um iterador é um objeto que fornece elementos um de cada vez e mantém seu estado de iteração. Em geral, pode ser consumido uma única vez.

- Preferir `iter()` e `next()` quando o controle explícito da iteração for necessário.
- Não presumir que um iterador possa ser reutilizado depois de consumido.
- Materializar com `list()` ou `tuple()` somente quando o contrato do consumidor exigir uma coleção concreta ou múltiplas passagens.
- Tratar o fim normal da iteração por meio do protocolo de iteração; evitar tratar `StopIteration` indiscriminadamente fora desse contexto.
- Documentar quando uma função retorna um iterador consumível uma única vez.

Exemplo:

```python
nomes_modelos = iter(("Ridge", "RandomForest", "HistGradientBoosting"))

primeiro_modelo = next(nomes_modelos)
```

## Generators

Um generator é uma forma de criar um iterador, normalmente usando uma função com `yield` ou uma generator expression.

- Preferir generators quando os elementos puderem ser produzidos sob demanda.
- Usar `yield` para emitir resultados incrementalmente sem construir uma coleção completa.
- Evitar guardar em memória todos os resultados de grandes sequências quando o consumidor puder processá-los um por vez.
- Não usar generators quando for necessário acessar elementos por índice, percorrer a sequência várias vezes ou disponibilizar uma coleção materializada.
- Manter funções geradoras simples, com responsabilidade clara e tipos de retorno explícitos, como `Iterator[ResultadoModelo]` ou `Generator[ResultadoModelo, None, None]`, conforme o contrato necessário.
- Não confundir avaliação lazy com execução paralela: um generator, por si só, não executa trabalho em paralelo.

Exemplo:

```python
from collections.abc import Iterator

def gerar_nomes_modelos() -> Iterator[str]:
    yield "Ridge"
    yield "RandomForest"
    yield "HistGradientBoosting"
```

## Aplicações no projeto

Generators e iterators podem ser úteis para:

```text
leitura incremental de registros
processamento de arquivos grandes
produção de resultados de experimentos
encadeamento de etapas compatíveis com iteráveis
avaliação sob demanda de combinações
redução do uso de memória fora de operações tabulares vetorizadas
```

Não utilizar generators para substituir operações vetorizadas de Pandas ou expressões nativas de Polars quando estas forem mais adequadas ao processamento tabular.

---

# Generator expressions

Preferir generator expressions quando não houver necessidade de materializar todos os elementos.

Exemplo:

```python
quantidade_ativos = sum(
    1
    for modelo in modelos
    if modelo.ativo
)
```

---

# enumerate

Quando for necessário índice e elemento ao mesmo tempo, utilizar:

```python
for indice, modelo in enumerate(modelos):
    ...
```

em vez de:

```python
for indice in range(len(modelos)):
    modelo = modelos[indice]
```

Essa regra não autoriza processamento linha a linha de `DataFrame`.

---

# Padrões de projeto

Utilizar quando resolverem problemas concretos:

```text
Strategy
Factory
Adapter
Observer
Repository
Facade
```

Outros padrões poderão ser utilizados quando tecnicamente justificados.

Não adicionar padrões apenas para aumentar a quantidade de abstrações do projeto.

---

# Qualidade arquitetural

O código deverá ser:

```text
desacoplado
testável
reutilizável
coeso
extensível
fortemente tipado
observável
configurável
```

Separar claramente:

```text
domínio
infraestrutura
configuração
persistência
orquestração
Machine Learning
observabilidade
regras de negócio
```

---

# Testes

Utilizar:

```text
pytest
```

Criar:

```text
testes unitários
testes de integração
testes de contrato
testes parametrizados
```

Utilizar quando apropriado:

```python
@pytest.fixture
@pytest.mark.parametrize
```

Mocks deverão ser utilizados principalmente para dependências externas e efeitos colaterais.

---

# Type checking

Configurar o type checker em modo:

```text
strict
```

Violações críticas de tipagem deverão falhar no CI.

O objetivo é detectar antecipadamente:

```text
tipos incompatíveis
retornos incorretos
parâmetros incorretos
atributos inexistentes
sobrescritas incompatíveis
uso indevido de Optional
```

---

# Regra de prioridade para código Python

Quando houver mais de uma solução possível, preferir nesta ordem:

```text
1. recurso nativo da linguagem
2. biblioteca padrão
3. operação vetorizada da biblioteca utilizada
4. recurso moderno e idiomático do pacote
5. abstração própria
6. implementação manual
```

Exemplo:

```text
sum()
```

é preferível a implementar manualmente um acumulador quando ambos resolvem exatamente o mesmo problema.

Da mesma forma:

```text
zip(..., strict=True)
itertools
collections
functools
```

devem ser preferidos a implementações manuais equivalentes quando aumentarem clareza, segurança e manutenção.

---

# Princípio geral

O projeto deve buscar código:

```text
Python moderno
+
tipagem forte
+
encapsulamento adequado
+
baixo acoplamento
+
alta coesão
+
configuração externa
+
processamento vetorizado
+
biblioteca padrão antes de reinventar soluções
+
testes automatizados
+
arquitetura extensível
```



### Crie uma sessão no arquivo yaml, de transformação de atributos, quero cadastrar uma expressão python pandas ou polars, posso cadastrar uma ou mais expressões de transformação. Deixe uma flag para ativar ou desativar uma ou mais expressões ou todas

### Crie uma sessão no arquivo yaml, de Criação de atributos, quero cadastrar uma expressão python pandas ou polars, posso cadastrar uma ou mais expressões de transformação. Deixe uma flag para ativar ou desativar uma ou mais expressões ou todas

### No arquivo yaml, crie um campo para escolher as colunas

### No arquivo yaml, crie uma coluna para criar a expressão python where, de acordo com o método de processamento de dados

## Etapa3: Análse Exploratória de dados

### Inicialmente, vou mandar para o grafana, mas deixe flexível no código python, e adicione uma opção no arquivo yaml, para a trocar do motor gráfico, exemplo: posso trocar para seaborn, ploty e outros. 

### Crie um arquivo yaml para casdatrar as análise eda, com expressão python, incluindo o tipo de gráfico para ser cadastrado.

## Etapa 4: Modelagem 
### Crie no arquivo yaml, para configurar os métodos de imputação de valores, de acordo com o framework escolhido, sklearn ou pycarret,, deixe uma flag para ativar e desativar, 

### Crie no arquivo yaml, para configurar os métodos de preprocessamento, de acordo com o framework escolhido, sklearn ou pycarret, , deixe uma flag para ativar e desativar


### Faça tuning dos parâmetros, de acordo com framework escolhido (sklean ou pycarret) dos modelos de regressão linear, lasso, ridge, múltipla, elasticnet, árvore de decisão, random florest, SVR, rede neural, xboost, lightgtm, lembrando que todos devem usar os métodos nativos do sklearn e pycarret, de acordo com a flag escolhida no yaml, organize o código python com os padrões de projeto gof e configuravel no arquivo yaml

### Quero escolher o método de turning de parametros no arquivo yaml

### Gravar os resultados o tuning de parâmetros no mlflow para cada modelo, por exemplo, no pai, deve estar, tuning_de_parametros_{metodo_de_tuning_escolhidos} e nos filhos deve estar modelo_{nome_do_modelo}_{nome_da_tecnica_preprocessamento} e registrar os melhores parâmetros

## Etapa 5: Seleção do modelo 
### Com os melhores parâmetros escolhidos no tuning, faça a validação cruzada com o método de fold, método de fold, deve ser escolhido no arquivo yaml, 

## Etapa 6: Avaliação:

### Aplique os teste de friedman e neyemi para escolher o melhor modelo de regressão, salve resultados no grafana, deixe flexível no código python,para caso eu queira trocar o método de salvamento no futuro

### Técnicas de votação:
#### Crie uma flag para verificar se vai usar técnicas de votação, se for true, use uma técnica de votação, que vai ser configura no arquivo yaml, para escolher as n melhores técnicas de acordo com os resultados dos testes de friedman e neyemi, (n vai ser configurável no arquivo yaml). No código python, configure as técnicas de votação, com as bibliotecas nativas  do sklearn e pycarret, todos os parâmetros devem ser configurados no arquivo yaml. Lembrando que o motor de machine learning, vai ser escolhido no arquivo yaml. Se for false, vai us

## Etapa 7: Implantação
### Depois que eu escolher o modelo campeão, use o valor predito para fazer os cálculos da regra de negócio que vai ser configurada no arquivo yaml, ou seja, quando eu cadastrar a regra de negócio no arquivo yaml, deve ser disponibilizada no retorno da api. Quero usar expressões python pandas ou polars para criar as fórmulas das regras de negócio.

## Etapa 8: Monitoramento
### Construa um dashboard de observabilidade no grafana 

## Apêndices 
### Sobre drift: Use a explicação abaixo e exiba os drifts no grafana. Deixe configurável no código python caso eu queira subtituir no grafana futuramente


Em Machine Learning, data drift ocorre quando as características dos dados mudam ao longo do tempo, podendo prejudicar o desempenho do modelo em produção. Existem vários tipos de drift, e cada um monitora uma parte diferente do sistema.

Principais tipos de drift em Machine Learning
Tipo de drift

O que muda?

Exemplo

1. Data Drift (Covariate Drift)

Distribuição das variáveis de entrada 
P
(
X
)
P(X)

Em um modelo de preços de automóveis, aumenta a proporção de veículos elétricos no conjunto de dados.

2. Concept Drift

Relação entre as entradas e o alvo 
P
(
Y
∣
X
)
P(Y∣X)

Um carro com determinadas características passa a valer menos devido a uma mudança no mercado.

3. Label Drift (Prior Probability Shift)

Distribuição da variável-alvo 
P
(
Y
)
P(Y)

A proporção de automóveis classificados como alto valor aumenta.

4. Prediction Drift

Distribuição das previsões do modelo 
P
(
Y
^
)
P( 
Y
^
 )

O modelo começa a prever preços sistematicamente mais altos.

5. Feature Drift

Distribuição de uma variável específica

A quilometragem média dos automóveis anunciados aumenta.

6. Target Drift

Mudança na distribuição do alvo real ao longo do tempo

O preço real dos veículos usados cai significativamente.

7. Covariance Drift

Mudança na relação entre variáveis de entrada 
P
(
X
i
,
X
j
)
P(X 
i
​
 ,X 
j
​
 )

A relação entre ano de fabricação e quilometragem muda.

8. Virtual Drift

Mudança em 
P
(
X
)
P(X) sem alteração necessária da relação 
P
(
Y
∣
X
)
P(Y∣X)

Aumenta a quantidade de carros populares, mas a relação entre características e preço permanece estável.

9. Real Concept Drift

A relação entre entradas e alvo realmente muda

O mercado passa a valorizar mais veículos híbridos do que veículos a combustão equivalentes.

Observação: alguns desses termos se sobrepõem ou são usados de maneiras diferentes na literatura. Por exemplo, feature drift costuma ser tratado como um caso de data drift, enquanto target drift pode se referir à mudança na distribuição do alvo real.

Diferenças entre os três principais tipos
Data Drift

O perfil dos dados de entrada mudou.

Pergunta: os dados que chegam hoje são diferentes dos dados usados no treinamento?

Concept Drift

A relação entre os dados de entrada e o resultado esperado mudou.

Pergunta: as mesmas características ainda levam ao mesmo resultado?

Prediction Drift

A distribuição das previsões do modelo mudou.

Pergunta: o modelo está produzindo previsões diferentes das que produzia anteriormente?

Outros aspectos importantes do drift
Além de identificar o tipo, é comum classificar o drift pelo comportamento temporal:

Sudden Drift: mudança abrupta, como uma alteração regulatória.

Gradual Drift: a distribuição antiga e a nova coexistem durante uma transição.

Incremental Drift: mudança progressiva ao longo do tempo.

Recurring Drift: padrões anteriores retornam, como a sazonalidade do mercado.

Blip Drift: mudança temporária, seguida do retorno ao comportamento anterior.

Como monitorar em produção
Para um projeto de previsão de preços de automóveis com Python, PyCaret e MLflow, eu organizaria o monitoramento assim:

Monitoramento

O que acompanhar

Data Drift

PSI, KS test, Wasserstein distance

Concept Drift

MAE, RMSE e erro residual ao longo do tempo, usando preços reais

Prediction Drift

Distribuição dos preços previstos, média, mediana e percentis

Target Drift

Distribuição dos preços reais, quando os rótulos estiverem disponíveis

Qualidade dos dados

Valores ausentes, categorias novas, limites e tipos de dados

Performance do modelo

MAE, RMSE, 
R
2
R 
2
  e métricas segmentadas por marca e ano

Atenção: prediction drift ou data drift não significam necessariamente que o modelo perdeu qualidade. São sinais para investigar. Para confirmar degradação de desempenho, é importante comparar as previsões com os valores reais quando estiverem disponíveis.




### Importante: todos os parâmetros de regras de negócio e dos modelos de machine learning, devem sem configuradas nos arquivos yaml, fora do código python para que eu consiga usar esse motor de regressão machine learning, em outros projetos de regressão

### Centralize as configurações no arquivo yaml criando um módulo conf_yaml.py e classe ConfYAML os outros pacotes python, que for usar um atributo configurado do yaml, deverá importar desse módulo


