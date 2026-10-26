"""Serviço para serialização, carregamento e execução do modelo campeão promovido."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast, final, override
import joblib
import numpy as np
import pandas as pd

from app.contratos_implantacao.provedor_campeao import MetadadosModelo, ProvedorCampeao


class EstimadorComPredict(Protocol):
    """Protocolo estrutural para estimadores que expõem método predict."""

    def predict(self, X: pd.DataFrame) -> object: ...


class ErroCarregadorCampeao(Exception):
    """Exceção levantada em falhas ao carregar ou executar o modelo campeão."""


@final
class CarregadorCampeao(ProvedorCampeao):
    """Implementação do ProvedorCampeao carregando artefatos persistidos do disco."""

    def __init__(
        self,
        caminho_modelo: Path | str,
        caminho_metadados: Path | str,
    ) -> None:
        self._caminho_modelo = Path(caminho_modelo)
        self._caminho_metadados = Path(caminho_metadados)
        self._estimador: EstimadorComPredict | None = None
        self._metadados: MetadadosModelo | None = None
        self._carregar_artefatos()

    def _carregar_artefatos(self) -> None:
        """Carrega o estimador serializado e o arquivo de metadados associado."""
        if not self._caminho_modelo.exists():
            raise ErroCarregadorCampeao(
                f"Artefato do modelo campeão não encontrado em '{self._caminho_modelo}'."
            )
        if not self._caminho_metadados.exists():
            raise ErroCarregadorCampeao(
                f"Arquivo de metadados do modelo campeão não encontrado em '{self._caminho_metadados}'."
            )

        try:
            objeto_carregado = cast(object, joblib.load(self._caminho_modelo))
            if not hasattr(objeto_carregado, "predict"):
                raise ErroCarregadorCampeao("O objeto carregado não possui método 'predict'.")
            self._estimador = cast(EstimadorComPredict, objeto_carregado)
        except Exception as erro_modelo:
            raise ErroCarregadorCampeao(
                f"Falha ao desserializar modelo de '{self._caminho_modelo}': {erro_modelo}"
            ) from erro_modelo

        try:
            with open(self._caminho_metadados, "r", encoding="utf-8") as arq_meta:
                dados_json = cast(dict[str, object], json.load(arq_meta))
            self._metadados = MetadadosModelo.model_validate(dados_json)
        except Exception as erro_meta:
            raise ErroCarregadorCampeao(
                f"Falha ao ler metadados de '{self._caminho_metadados}': {erro_meta}"
            ) from erro_meta

    @override
    def prever(self, dados: pd.DataFrame) -> list[float]:
        """Realiza a predição para o DataFrame informado retornando preços em R$."""
        if self._estimador is None or self._metadados is None:
            raise ErroCarregadorCampeao("Estimador ou metadados não inicializados.")

        if dados.empty:
            return []

        try:
            predicoes_brutas = self._estimador.predict(dados)
            array_predicoes = np.asarray(predicoes_brutas, dtype=np.float64)

            # Caso a transformação de alvo não tenha sido embutida no TransformedTargetRegressor
            if self._metadados.transformacao_alvo == "log1p_manual":
                array_predicoes = np.expm1(array_predicoes)

            # Garantir que não haja previsões negativas para imóveis
            array_predicoes = np.clip(array_predicoes, a_min=10_000.0, a_max=None)
            lista_valores = [round(float(v), 2) for v in array_predicoes.tolist()]
            return lista_valores
        except Exception as erro_pred:
            raise ErroCarregadorCampeao(f"Falha durante a inferência do modelo campeão: {erro_pred}") from erro_pred

    @override
    def obter_metadados(self) -> MetadadosModelo:
        """Retorna os metadados oficiais do modelo promovido."""
        if self._metadados is None:
            raise ErroCarregadorCampeao("Metadados não inicializados.")
        return self._metadados

    @classmethod
    def salvar_modelo_e_metadados(
        cls,
        estimador: object,
        metadados: MetadadosModelo,
        diretorio_destino: Path | str,
    ) -> CarregadorCampeao:
        """Salva estimador e metadados no diretório informado e retorna a instância carregada."""
        diretorio = Path(diretorio_destino)
        diretorio.mkdir(parents=True, exist_ok=True)

        caminho_modelo = diretorio / "modelo_campeao.joblib"
        caminho_metadados = diretorio / "metadados_campeao.json"

        try:
            joblib.dump(estimador, caminho_modelo)
            with open(caminho_metadados, "w", encoding="utf-8") as arq_meta:
                json.dump(metadados.model_dump(), arq_meta, indent=2, ensure_ascii=False)
        except Exception as erro_salvar:
            raise ErroCarregadorCampeao(
                f"Erro ao salvar artefatos do campeão em '{diretorio}': {erro_salvar}"
            ) from erro_salvar

        return cls(caminho_modelo=caminho_modelo, caminho_metadados=caminho_metadados)
