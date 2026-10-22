from __future__ import annotations
from typing import final, override
import pandas as pd
from pycaret.regression import RegressionExperiment
from app.contratos_dados.motor_treino import ModeloPreditor, MotorTreino, ValorMetricaParametro
from app.tipos_enumerados import (
    EstrategiaFold,
    MetodoTransformacao,
    MetodoTuning,
    TipoCodificador,
    TipoEscalonador,
    TipoModelo,
)


class ErroMotorPycaret(Exception):
    """Exceção levantada em operações do motor PyCaret."""
    pass


@final
class MotorPycaret(MotorTreino):
    """Trilha EXCLUSIVAMENTE PyCaret usando API nativa pública com suporte aos 10 modelos configuráveis."""

    _DEFAULTS_MODELOS: list[dict[str, str | bool]] = [
        {"nome": TipoModelo.REGRESSAO_LINEAR.value, "pycaret_id": TipoModelo.REGRESSAO_LINEAR.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.LASSO.value, "pycaret_id": TipoModelo.LASSO.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.RIDGE.value, "pycaret_id": TipoModelo.RIDGE.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.ELASTICNET.value, "pycaret_id": TipoModelo.ELASTICNET.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.ARVORE_DECISAO.value, "pycaret_id": TipoModelo.ARVORE_DECISAO.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.RANDOM_FOREST.value, "pycaret_id": TipoModelo.RANDOM_FOREST.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.SVR.value, "pycaret_id": TipoModelo.SVR.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.REDE_NEURAL.value, "pycaret_id": TipoModelo.REDE_NEURAL.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.XGBOOST.value, "pycaret_id": TipoModelo.XGBOOST.obter_pycaret_id(), "ativo": True},
        {"nome": TipoModelo.LIGHTGBM.value, "pycaret_id": TipoModelo.LIGHTGBM.obter_pycaret_id(), "ativo": True},
    ]

    def __init__(
        self,
        semente: int = 42,
        folds: int = 3,
        metodo_folds: str = EstrategiaFold.KFOLD.value,
        embaralhar_folds: bool = True,
        colunas_numericas: list[str] | None = None,
        colunas_categoricas: list[str] | None = None,
        preprocessamento_por_modelo: dict[str, dict[str, str]] | None = None,
        modelos_configurados: list[dict[str, str | bool]] | None = None,
        parametros_tuning: dict[str, dict[str,
                                          list[str | int | float | bool]]] | None = None,
        metodo_tuning: str = MetodoTuning.RANDOM.value,
        transformar_alvo: bool = False,
        metodo_transformacao_alvo: str = MetodoTransformacao.YEO_JOHNSON.value,
        transformar_atributos: bool = False,
        metodo_transformacao_atributos: str = MetodoTransformacao.YEO_JOHNSON.value,
        colunas_transformacao_atributos: list[str] | None = None,
    ) -> None:
        self._semente = semente
        self._folds = folds
        self._metodo_folds = metodo_folds
        self._embaralhar_folds = embaralhar_folds
        self._colunas_numericas = colunas_numericas
        self._colunas_categoricas = colunas_categoricas
        self._preprocessamento_por_modelo = preprocessamento_por_modelo or {}
        self._modelos_configurados = modelos_configurados or self._DEFAULTS_MODELOS
        self._parametros_tuning = parametros_tuning or {}
        self._metodo_tuning = metodo_tuning.lower().strip()
        self._transformar_alvo = transformar_alvo
        self._metodo_transformacao_alvo = metodo_transformacao_alvo.lower().strip()
        self._transformar_atributos = transformar_atributos
        self._metodo_transformacao_atributos = metodo_transformacao_atributos.lower().strip()
        self._colunas_transformacao_atributos = colunas_transformacao_atributos
        self._experimento = RegressionExperiment()
        self._modelo_campeao: ModeloPreditor | None = None
        self._nome_modelo_campeao: str = TipoModelo.RIDGE.value


    def preparar_experimento(
        self,
        dados: pd.DataFrame,
        alvo: str,
        modelo_nome: str | None = None,
        colunas_numericas: list[str] | None = None,
        colunas_categoricas: list[str] | None = None,
        escalonador: str | None = None,
        codificador: str | None = None,
        metodo_folds: str | None = None,
        embaralhar_folds: bool | None = None,
    ) -> None:
        try:
            # 1. Identificação e validação dos inputs colunares
            colunas_disponiveis = [c for c in dados.columns if c != alvo]

            num_features = colunas_numericas or self._colunas_numericas
            if num_features is None:
                num_features = [
                    c for c in colunas_disponiveis
                    if pd.api.types.is_numeric_dtype(dados[c])
                ]
            else:
                num_features = [
                    c for c in num_features if c in colunas_disponiveis]

            cat_features = colunas_categoricas or self._colunas_categoricas
            if cat_features is None:
                cat_features = [
                    c for c in colunas_disponiveis
                    if c not in num_features
                ]
            else:
                cat_features = [
                    c for c in cat_features if c in colunas_disponiveis]

            # 2. Configuração de pré-processamento por modelo (escalonador e One-Hot Encoder via Enums)
            config_modelo = self._preprocessamento_por_modelo.get(
                modelo_nome or "", {})

            valor_escalonador_raw = escalonador or config_modelo.get(
                "escalonador", TipoEscalonador.STANDARD.value)
            escalonador_enum = TipoEscalonador.de_string(valor_escalonador_raw)

            valor_codificador_raw = codificador or config_modelo.get(
                "codificador", TipoCodificador.ONE_HOT.value)
            codificador_enum = TipoCodificador.de_string(valor_codificador_raw)

            normalizar = escalonador_enum in (
                TipoEscalonador.STANDARD,
                TipoEscalonador.ZSCORE,
                TipoEscalonador.MINMAX,
                TipoEscalonador.ROBUST,
            )
            metodo_normalizacao = "minmax" if escalonador_enum == TipoEscalonador.MINMAX else "zscore"
            usar_one_hot = (codificador_enum == TipoCodificador.ONE_HOT)

            # 3. Configuração do método e estratégia de folds via Enum
            metodo_fold_raw = metodo_folds or self._metodo_folds
            estrategia_fold = EstrategiaFold.de_string(metodo_fold_raw).value
            embaralhar = self._embaralhar_folds if embaralhar_folds is None else embaralhar_folds

            # 4. Setup nativo oficial do PyCaret com fold_strategy, One-Hot Encoding e Normalização
            metodo_target_enum = MetodoTransformacao.de_string(self._metodo_transformacao_alvo)
            metodo_target = "box-cox" if metodo_target_enum == MetodoTransformacao.BOX_COX else "yeo-johnson"

            metodo_features_enum = MetodoTransformacao.de_string(self._metodo_transformacao_atributos)
            metodo_features = "box-cox" if metodo_features_enum == MetodoTransformacao.BOX_COX else "yeo-johnson"

            self._experimento.setup(
                data=dados,
                target=alvo,
                session_id=self._semente,
                fold=self._folds,
                fold_strategy=estrategia_fold,
                fold_shuffle=embaralhar,
                numeric_features=num_features if num_features else None,
                categorical_features=cat_features if cat_features else None,
                normalize=normalizar,
                normalize_method=metodo_normalizacao,
                max_encoding_ohe=100 if usar_one_hot else 25,
                transform_target=self._transformar_alvo,
                transform_target_method=metodo_target,
                transformation=self._transformar_atributos,
                transformation_method=metodo_features,
                verbose=False
            )

        except Exception as erro:
            raise ErroMotorPycaret(
                f"Falha ao inicializar setup nativo do PyCaret com fold_strategy '{metodo_folds or self._metodo_folds}': {erro}"
            ) from erro

    def _obter_pycaret_id(self, item_modelo: dict[str, str | bool]) -> str:
        pycaret_id_raw = item_modelo.get("pycaret_id")
        if pycaret_id_raw:
            return str(pycaret_id_raw).lower()
        nome_modelo_str = str(item_modelo.get("nome", TipoModelo.RIDGE.value))
        try:
            modelo_enum = TipoModelo.de_string(nome_modelo_str)
            return modelo_enum.obter_pycaret_id()
        except ValueError:
            return nome_modelo_str.lower()

    @override
    def avaliar_modelos(self, dados: pd.DataFrame, alvo: str) -> pd.DataFrame:
        resultados: dict[str, list[float]] = {}
        modelos_ativos = [
            m for m in self._modelos_configurados if m.get("ativo", True)]

        if not modelos_ativos:
            modelos_ativos = [
                {"nome": "ridge", "pycaret_id": "ridge", "ativo": True},
                {"nome": "random_forest", "pycaret_id": "rf", "ativo": True}
            ]

        for mod in modelos_ativos:
            nome_modelo = str(mod.get("nome", "modelo"))
            pycaret_id = self._obter_pycaret_id(mod)
            try:
                self.preparar_experimento(dados, alvo, modelo_nome=nome_modelo)
                self._experimento.create_model(pycaret_id, verbose=True)
                tabela = self._experimento.pull()
                # Extrai os scores RMSE dos folds individuais (desconsiderando média e std nas últimas 2 linhas)
                erros = [float(v)
                         for v in tabela.loc[tabela.index[:-2], "RMSE"]]
                resultados[nome_modelo] = erros
            except Exception as erro:
                raise ErroMotorPycaret(
                    f"Falha ao avaliar modelo '{nome_modelo}' (id: '{pycaret_id}') com PyCaret nativo: {erro}"
                ) from erro

        return pd.DataFrame(resultados)

    @override
    def ajustar_parametros(
        self,
        dados: pd.DataFrame,
        alvo: str,
        iteracoes: int,
        modelo_nome: str | None = None,
        metodo_tuning: str | None = None,
    ) -> dict[str, ValorMetricaParametro]:
        nome_alvo = modelo_nome or self._nome_modelo_campeao or "ridge"
        self._nome_modelo_campeao = nome_alvo

        # Encontra a configuração do modelo
        pycaret_id = "ridge"
        for mod in self._modelos_configurados:
            if mod.get("nome") == nome_alvo:
                pycaret_id = self._obter_pycaret_id(mod)
                break

        self.preparar_experimento(dados, alvo, modelo_nome=nome_alvo)
        try:
            inicial = self._experimento.create_model(pycaret_id, verbose=False)
            custom_grid = self._parametros_tuning.get(nome_alvo)
            metodo_candidato = (
                metodo_tuning or self._metodo_tuning).lower().strip()
            algoritmo_busca = metodo_candidato if metodo_candidato in (
                "random", "grid", "bayesian", "optuna", "tune-sklearn") else "random"

            if custom_grid:
                ajustado = self._experimento.tune_model(
                    inicial,
                    n_iter=iteracoes,
                    custom_grid=custom_grid,
                    search_algorithm=algoritmo_busca,
                    verbose=True
                )
            else:
                ajustado = self._experimento.tune_model(
                    inicial,
                    n_iter=iteracoes,
                    search_algorithm=algoritmo_busca,
                    verbose=True
                )

            self._modelo_campeao = ajustado
            relatorio = self._experimento.pull()
            return {
                "modelo": nome_alvo,
                "pycaret_id": pycaret_id,
                "metricas_tuning": relatorio.to_dict()
            }
        except Exception as erro:
            raise ErroMotorPycaret(
                f"Falha no tuning nativo do PyCaret para modelo '{nome_alvo}': {erro}"
            ) from erro

    @override
    def treinar_final(
        self,
        dados: pd.DataFrame,
        alvo: str,
        modelo_nome: str | None = None
    ) -> ModeloPreditor:
        nome_alvo = modelo_nome or self._nome_modelo_campeao or "ridge"
        self.preparar_experimento(dados, alvo, modelo_nome=nome_alvo)
        try:
            if self._modelo_campeao is None:
                pycaret_id = "ridge"
                for mod in self._modelos_configurados:
                    if mod.get("nome") == nome_alvo:
                        pycaret_id = self._obter_pycaret_id(mod)
                        break
                self._modelo_campeao = self._experimento.create_model(
                    pycaret_id, verbose=False)

            finalizado = self._experimento.finalize_model(self._modelo_campeao)
            return finalizado
        except Exception as erro:
            raise ErroMotorPycaret(
                f"Falha na finalização nativa de modelo '{nome_alvo}' com PyCaret: {erro}"
            ) from erro
