"""Serviço orquestrador genérico para a Etapa 1: Coleta de Dados e Persistência no Staging."""

from __future__ import annotations

import time
from typing import cast, final
from pydantic import BaseModel, Field

from app.conf_yaml import ConfYAML
from app.contratos_dados.fonte_dados import FonteDados
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.fontes_dados.fabrica_fontes import FabricaFontes
from app.repositorios_dados.fabrica_repositorios import FabricaRepositorios


class ErroColetaDados(Exception):
    """Exceção levantada quando ocorre erro durante a execução da coleta ou validação inicial."""


class ResultadoColeta(BaseModel):
    """Sumário com métricas e evidências da execução da etapa de coleta e persistência."""

    total_registros: int = Field(ge=0, description="Total de registros coletados e persistidos")
    total_colunas: int = Field(ge=0, description="Total de colunas presentes no conjunto")
    colunas: list[str] = Field(description="Nomes das colunas carregadas")
    alvo_presente: bool = Field(description="Indica se a coluna alvo foi detectada nos dados")
    origem_dados: str = Field(description="Identificador descritivo da fonte")
    tabela_destino: str = Field(description="Nome da tabela de persistência no staging")
    tempo_execucao_segundos: float = Field(
        ge=0.0, description="Duração total da coleta e persistência em segundos"
    )


@final
class ColetorDados:
    """Coordenador genérico de ingestão e materialização de dados na área de staging intermediária."""

    def __init__(
        self,
        fonte: FonteDados,
        repositorio: RepositorioDados,
        nome_alvo: str | None = None,
        colunas_obrigatorias: list[str] | None = None,
        validar_nulos_alvo: bool = False,
        tabela_destino: str = "",
    ) -> None:
        self._fonte = fonte
        self._repositorio = repositorio
        self._nome_alvo = nome_alvo
        self._colunas_obrigatorias = list(colunas_obrigatorias or [])
        self._validar_nulos_alvo = validar_nulos_alvo
        self._tabela_destino = tabela_destino

    @classmethod
    def da_configuracao(cls, conf: ConfYAML) -> ColetorDados:
        """Instancia o coletor desacoplado utilizando fábricas GoF e regras do ConfYAML."""
        config_fonte = conf.obter_fonte()
        config_staging = conf.obter_staging()
        config_coleta = conf.obter_coleta()
        nome_alvo = config_coleta.nome_alvo or conf.obter_projeto().alvo

        fonte = FabricaFontes.criar_fonte(config_fonte)
        repositorio = FabricaRepositorios.criar_repositorio(config_staging)

        return cls(
            fonte=fonte,
            repositorio=repositorio,
            nome_alvo=nome_alvo,
            colunas_obrigatorias=config_coleta.colunas_obrigatorias,
            validar_nulos_alvo=config_coleta.validar_nulos_alvo,
            tabela_destino=config_staging.tabela,
        )

    def executar_coleta(self, substituir_staging: bool = True) -> ResultadoColeta:
        """Executa a ingestão, validação de integridade de negócio e materialização no repositório."""
        inicio = time.perf_counter()

        try:
            dados = self._fonte.carregar_dados()
        except Exception as erro_carga:
            raise ErroColetaDados(
                f"Falha ao carregar dados a partir de '{self._fonte.obter_origem()}': {erro_carga}"
            ) from erro_carga

        if dados.empty:
            raise ErroColetaDados("A fonte de dados retornou um conjunto de dados vazio.")

        # 1. Validação configurável da coluna alvo
        if self._nome_alvo is not None:
            if self._nome_alvo not in dados.columns:
                raise ErroColetaDados(
                    f"A coluna alvo obrigatória '{self._nome_alvo}' não está presente nas colunas da fonte. "
                    f"Colunas encontradas: {dados.columns.tolist()}"
                )
            if self._validar_nulos_alvo:
                total_nulos = int(cast(int, dados[self._nome_alvo].isna().sum()))
                if total_nulos > 0:
                    raise ErroColetaDados(
                        f"A coluna alvo '{self._nome_alvo}' possui {total_nulos} valores ausentes (nulos), "
                        f"violando a regra de negócio da coleta."
                    )

        # 2. Validação configurável de colunas de negócio obrigatórias
        if self._colunas_obrigatorias:
            colunas_ausentes = [c for c in self._colunas_obrigatorias if c not in dados.columns]
            if colunas_ausentes:
                raise ErroColetaDados(
                    f"As seguintes colunas de negócio obrigatórias configuradas no YAML estão ausentes na fonte: {colunas_ausentes}. "
                    f"Colunas encontradas: {dados.columns.tolist()}"
                )

        # 3. Materialização no repositório de staging
        try:
            total_gravado = self._repositorio.gravar_dados(dados, substituir=substituir_staging)
        except Exception as erro_persistencia:
            raise ErroColetaDados(
                f"Falha ao persistir dados no repositório de staging: {erro_persistencia}"
            ) from erro_persistencia

        duracao = time.perf_counter() - inicio

        colunas_lista = cast(list[object], list(dados.columns))
        alvo_presente = (self._nome_alvo in dados.columns) if self._nome_alvo else False

        return ResultadoColeta(
            total_registros=total_gravado,
            total_colunas=len(dados.columns),
            colunas=[str(c) for c in colunas_lista],
            alvo_presente=alvo_presente,
            origem_dados=self._fonte.obter_origem(),
            tabela_destino=self._tabela_destino,
            tempo_execucao_segundos=round(duracao, 4),
        )


# Alias de compatibilidade retroativa
ColetorImoveis = ColetorDados
