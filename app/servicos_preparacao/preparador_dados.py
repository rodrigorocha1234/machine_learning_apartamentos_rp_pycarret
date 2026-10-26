"""Serviço orquestrador para a Etapa 2: Preparação de Dados e Engenharia de Atributos."""

from __future__ import annotations

import time
from typing import cast, final
import pandas as pd
from pydantic import BaseModel, Field

from app.conf_yaml import ConfYAML
from app.contratos_dados.motor_tabela import MotorTabela
from app.esquema_configuracao import (
    ConfiguracaoAtributosNovos,
    ConfiguracaoFiltros,
    ConfiguracaoSelecao,
    ConfiguracaoTransformacoes,
)
from app.motores_tabela.fabrica_motores import FabricaMotores


class ErroPreparacaoDados(Exception):
    """Exceção levantada em falhas durante a preparação e engenharia de dados."""


class ResultadoPreparacao(BaseModel):
    """Sumário com métricas e evidências da execução da etapa de preparação de dados."""

    total_linhas_originais: int = Field(ge=0, description="Quantidade de registros antes dos filtros")
    total_linhas_filtradas: int = Field(ge=0, description="Quantidade de registros após filtragem")
    linhas_removidas_filtros: int = Field(ge=0, description="Registros descartados pelos filtros")
    total_colunas_finais: int = Field(ge=0, description="Quantidade de colunas selecionadas")
    colunas_finais: list[str] = Field(description="Nomes das colunas preparadas")
    motor_utilizado: str = Field(description="Identificador do motor tabular utilizado")
    transformacoes_aplicadas: list[str] = Field(description="Identificadores das transformações ativas")
    atributos_novos_criados: list[str] = Field(description="Identificadores dos novos atributos criados")
    tempo_execucao_segundos: float = Field(
        ge=0.0, description="Duração da preparação e transformações em segundos"
    )


@final
class PreparadorDados:
    """Coordenador desacoplado da preparação tabular, filtros e engenharia de atributos."""

    def __init__(
        self,
        motor: MotorTabela,
        selecao: ConfiguracaoSelecao,
        filtros: ConfiguracaoFiltros,
        transformacoes: ConfiguracaoTransformacoes,
        atributos_novos: ConfiguracaoAtributosNovos,
        nome_alvo: str = "Valor_da_Venda",
    ) -> None:
        self._motor = motor
        self._selecao = selecao
        self._filtros = filtros
        self._transformacoes = transformacoes
        self._atributos_novos = atributos_novos
        self._nome_alvo = nome_alvo

    @classmethod
    def da_configuracao(
        cls, conf: ConfYAML, motor_sobrescrito: MotorTabela | None = None
    ) -> PreparadorDados:
        """Instancia o PreparadorDados a partir das definições do ConfYAML."""
        motor = (
            motor_sobrescrito
            if motor_sobrescrito is not None
            else FabricaMotores.criar_motor(conf.obter_tabular())
        )
        return cls(
            motor=motor,
            selecao=conf.obter_selecao(),
            filtros=conf.obter_filtros(),
            transformacoes=conf.obter_transformacoes(),
            atributos_novos=conf.obter_atributos_novos(),
            nome_alvo=conf.obter_projeto().alvo,
        )

    def preparar(self, dados: pd.DataFrame) -> tuple[pd.DataFrame, ResultadoPreparacao]:
        """Aplica filtros, transformações, novos atributos e seleção de colunas na ordem do YAML."""
        inicio = time.perf_counter()

        if dados.empty:
            raise ErroPreparacaoDados("DataFrame de entrada para preparação está vazio.")

        total_linhas_originais = len(dados)
        df_atual = dados.copy()
        nome_motor = self._motor.obter_nome_motor()

        # 1. Aplicação de Filtros Condicionais
        if self._filtros.ativo:
            for filtro in self._filtros.expressoes:
                if filtro.ativo:
                    expressao = filtro.expressao.get(
                        nome_motor, filtro.expressao.get("pandas", "")
                    )
                    if expressao:
                        try:
                            df_atual = self._motor.filtrar_dados(df_atual, expressao)
                        except Exception as erro_filtro:
                            raise ErroPreparacaoDados(
                                f"Erro ao aplicar filtro '{filtro.nome}' com motor '{nome_motor}': {erro_filtro}"
                            ) from erro_filtro

        total_linhas_filtradas = len(df_atual)
        if df_atual.empty:
            raise ErroPreparacaoDados("Nenhum registro restante após a aplicação dos filtros configurados.")

        # 2. Aplicação de Transformações de Colunas
        transformacoes_aplicadas: list[str] = []
        if self._transformacoes.ativo:
            for transformacao in self._transformacoes.expressoes:
                if transformacao.ativo:
                    expressao = transformacao.expressao.get(
                        nome_motor, transformacao.expressao.get("pandas", "")
                    )
                    if expressao:
                        try:
                            df_atual = self._motor.transformar_dados(
                                df_atual, expressao, destino=transformacao.destino
                            )
                            transformacoes_aplicadas.append(transformacao.nome)
                        except Exception as erro_transf:
                            raise ErroPreparacaoDados(
                                f"Erro na transformação '{transformacao.nome}': {erro_transf}"
                            ) from erro_transf

        # 3. Criação de Novos Atributos
        atributos_criados: list[str] = []
        if self._atributos_novos.ativo:
            for atributo in self._atributos_novos.expressoes:
                if atributo.ativo:
                    expressao = atributo.expressao.get(
                        nome_motor, atributo.expressao.get("pandas", "")
                    )
                    if expressao:
                        try:
                            df_atual = self._motor.transformar_dados(
                                df_atual, expressao, destino=atributo.destino
                            )
                            atributos_criados.append(atributo.nome)
                        except Exception as erro_novo_atr:
                            raise ErroPreparacaoDados(
                                f"Erro ao criar atributo '{atributo.nome}': {erro_novo_atr}"
                            ) from erro_novo_atr

        # 4. Seleção e Ordenação de Colunas Ativas
        if self._selecao.ativo:
            colunas_selecionadas: list[str] = [
                col.nome
                for col in self._selecao.colunas
                if col.ativo and col.nome in df_atual.columns
            ]
            # Inclui colunas de atributos novos ativos caso não estejam explicitadas no YAML de seleção
            for nome_dest in atributos_criados:
                if nome_dest not in colunas_selecionadas and nome_dest in df_atual.columns:
                    colunas_selecionadas.append(nome_dest)

            if not colunas_selecionadas:
                raise ErroPreparacaoDados(
                    "Nenhuma coluna ativa configurada foi encontrada no conjunto de dados preparado."
                )

            try:
                df_atual = self._motor.selecionar_colunas(df_atual, colunas_selecionadas)
            except Exception as erro_selecao:
                raise ErroPreparacaoDados(
                    f"Erro ao selecionar colunas configuradas: {erro_selecao}"
                ) from erro_selecao

        if self._nome_alvo not in df_atual.columns:
            raise ErroPreparacaoDados(
                f"Coluna alvo '{self._nome_alvo}' ausente após a etapa de preparação."
            )

        duracao = time.perf_counter() - inicio
        colunas_finais_cast = cast(list[object], list(df_atual.columns))

        resultado = ResultadoPreparacao(
            total_linhas_originais=total_linhas_originais,
            total_linhas_filtradas=total_linhas_filtradas,
            linhas_removidas_filtros=total_linhas_originais - total_linhas_filtradas,
            total_colunas_finais=len(df_atual.columns),
            colunas_finais=[str(c) for c in colunas_finais_cast],
            motor_utilizado=nome_motor,
            transformacoes_aplicadas=transformacoes_aplicadas,
            atributos_novos_criados=atributos_criados,
            tempo_execucao_segundos=round(duracao, 4),
        )

        return df_atual, resultado
