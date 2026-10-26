"""Serviço central de inferência que une predição do modelo campeão e cálculo de regras."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import final
import pandas as pd

from app.contratos_implantacao.avaliador_regras import (
    AvaliadorRegras,
    RequisicaoImovel,
    RespostaPredicao,
    RespostaRegra,
)
from app.contratos_implantacao.provedor_campeao import ProvedorCampeao
from app.esquema_configuracao import ConfiguracaoImplantacao


class ErroServicoInferencia(Exception):
    """Exceção levantada em falhas operacionais do serviço de inferência."""


@final
class ServicoInferencia:
    """Orquestrador de predição e aplicação de regras de negócio em produção."""

    def __init__(
        self,
        provedor_campeao: ProvedorCampeao,
        avaliador_regras: AvaliadorRegras,
        config_implantacao: ConfiguracaoImplantacao,
    ) -> None:
        self._provedor = provedor_campeao
        self._avaliador = avaliador_regras
        self._config = config_implantacao

    def predizer_imovel(self, requisicao: RequisicaoImovel) -> RespostaPredicao:
        """Executa a previsão para um imóvel único e calcula as regras de negócio."""
        dados_dict = requisicao.model_dump()
        df_imovel = pd.DataFrame([dados_dict])

        # 1. Inferência com o modelo campeão
        predicoes = self._provedor.prever(df_imovel)
        if not predicoes:
            raise ErroServicoInferencia("O modelo campeão não retornou predição para o imóvel.")

        preco_predito = predicoes[0]

        # 2. Montagem do contexto para as regras dinâmicas
        nome_alvo_predito = self._config.coluna_alvo_predito
        contexto: dict[str, float | int | str] = {
            nome_alvo_predito: preco_predito,
        }
        for k, v in dados_dict.items():
            contexto[str(k)] = v

        # 3. Aplicação das regras de negócio
        regras_calculadas = self._avaliador.avaliar_regras_registro(contexto)

        # 4. Detalhamento das regras calculadas
        detalhes: list[RespostaRegra] = []
        for regra_item in self._config.regras:
            if not regra_item.ativo:
                continue
            valor = regras_calculadas.get(regra_item.destino, 0.0)
            detalhes.append(
                RespostaRegra(
                    nome=regra_item.nome,
                    destino=regra_item.destino,
                    descricao=regra_item.descricao,
                    valor=valor,
                )
            )

        # 5. Metadados do modelo
        meta = self._provedor.obter_metadados()
        data_hora_atual = datetime.now(timezone.utc).isoformat()

        return RespostaPredicao(
            preco_predito=preco_predito,
            regras_calculadas=regras_calculadas,
            detalhes_regras=detalhes,
            modelo_campeao=meta.nome_modelo,
            versao_modelo=meta.versao,
            data_hora_iso=data_hora_atual,
        )

    def predizer_lote(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Executa a inferência em lote e enriquece a tabela com as regras calculadas."""
        if dados.empty:
            return dados.copy()

        # 1. Inferência em lote
        predicoes = self._provedor.prever(dados)
        coluna_predita = self._config.coluna_alvo_predito

        df_com_predicao = dados.copy()
        df_com_predicao[coluna_predita] = predicoes

        # 2. Aplicação das regras na tabela
        df_enriquecido = self._avaliador.avaliar_regras_tabela(
            df_com_predicao, coluna_predita=coluna_predita
        )
        return df_enriquecido

    def obter_provedor(self) -> ProvedorCampeao:
        """Acesso ao provedor do modelo para inspeção ou telemetria."""
        return self._provedor

    def obter_avaliador(self) -> AvaliadorRegras:
        """Acesso ao avaliador de regras."""
        return self._avaliador
