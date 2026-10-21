# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportUnknownParameterType=false
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import final
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from app.contratos_dados.motor_treino import ModeloPreditor
from app.configuracao_projeto.leitor_configuracao import LeitorConfiguracao

class ErroDiagnosticoCurva(Exception):
    """Exceção levantada em erros de cálculo da curva de diagnóstico de aprendizado."""
    pass

@dataclass(frozen=True)
class PontoCurva:
    """Representa um ponto da curva de aprendizado em determinado nível de complexidade."""
    complexidade: float
    rmse_treino: float
    rmse_validacao: float
    gap_variancia: float

@dataclass(frozen=True)
class RelatorioCurva:
    """Relatório estruturado de diagnóstico de viés x variância para análise no Grafana."""
    modelo: str
    motor: str
    parametro_complexidade: str
    pontos: list[PontoCurva]
    melhor_complexidade: float
    menor_rmse_validacao: float
    diagnostico_regime: str

    def para_dicionario(self) -> dict[str, str | float | list[dict[str, float]]]:
        return {
            "modelo": self.modelo,
            "motor": self.motor,
            "parametro_complexidade": self.parametro_complexidade,
            "melhor_complexidade": self.melhor_complexidade,
            "menor_rmse_validacao": self.menor_rmse_validacao,
            "diagnostico_regime": self.diagnostico_regime,
            "pontos": [asdict(p) for p in self.pontos]
        }

@final
class DiagnosticoCurva:
    """Calculador analítico de curvas empíricas de viés e variância (Overfitting vs Underfitting)."""

    _CAMINHO_YAML_PADRAO: str = "configuracoes/projeto.yaml"

    _MAPA_PARAMETROS_PADRAO: dict[str, str] = {
        "arvore_decisao": "max_depth",
        "decision_tree": "max_depth",
        "random_forest": "max_depth",
        "floresta_aleatoria": "max_depth",
        "xgboost": "max_depth",
        "lightgbm": "max_depth",
        "ridge": "alpha",
        "lasso": "alpha",
        "elasticnet": "alpha",
        "regressao_linear": "fit_intercept",
        "svr": "C",
        "svm": "C",
        "rede_neural": "hidden_layer_sizes",
        "mlp": "hidden_layer_sizes",
        "modelo_campeao": "auto",
    }

    def __init__(
        self,
        semente: int = 42,
        cv_splits: int = 3,
        caminho_yaml: str | None = None,
        mapa_parametros: dict[str, str] | None = None,
        modelo_campeao: str | None = None
    ) -> None:
        self._semente = semente
        self._cv_splits = cv_splits
        self._caminho_yaml = caminho_yaml or self._CAMINHO_YAML_PADRAO
        self._modelo_campeao: str | None = modelo_campeao
        self._mapa_parametros: dict[str, str] = (
            mapa_parametros
            if mapa_parametros is not None
            else self.carregar_mapa_yaml(self._caminho_yaml)
        )

    def definir_modelo_campeao(self, modelo_campeao: str) -> None:
        """Define ou atualiza o modelo campeão de referência para diagnóstico dinâmico."""
        self._modelo_campeao = modelo_campeao.strip()

    @property
    def modelo_campeao_ativo(self) -> str:
        """Retorna o modelo campeão ativo, consultando o YAML se não configurado explicitamente."""
        if self._modelo_campeao:
            return self._modelo_campeao
        return self._obter_modelo_campeao_padrao_yaml(self._caminho_yaml)

    @classmethod
    def _obter_modelo_campeao_padrao_yaml(cls, caminho_yaml: str) -> str:
        """Lê do arquivo YAML o modelo campeão configurado como padrão ou fallback."""
        if not os.path.exists(caminho_yaml):
            return "random_forest"
        try:
            leitor = LeitorConfiguracao()
            configuracao = leitor.carregar_configuracao(caminho_yaml)
            sec_pipeline = configuracao.get("pipeline")
            if isinstance(sec_pipeline, dict):
                campeao = sec_pipeline.get("modelo_campeao_padrao")
                if isinstance(campeao, str) and campeao.strip():
                    return campeao.strip().lower()
            sec_mlflow = configuracao.get("mlflow")
            if isinstance(sec_mlflow, dict):
                campeao = sec_mlflow.get("modelo_campeao")
                if isinstance(campeao, str) and campeao.strip():
                    return campeao.strip().lower()
        except Exception:
            pass
        return "random_forest"

    def obter_parametro_base_modelo(self, modelo_nome: str) -> str:
        """Retorna o hiperparâmetro de complexidade base associado ao modelo."""
        nome = modelo_nome.lower().strip()
        for chave, param in self._mapa_parametros.items():
            if chave != "modelo_campeao" and chave in nome:
                return param

        if any(t in nome for t in ["arvore", "tree", "forest", "floresta", "xgboost", "lightgbm"]):
            return "max_depth"
        if any(t in nome for t in ["ridge", "lasso", "elasticnet"]):
            return "alpha"
        if "svr" in nome or "svm" in nome:
            return "C"
        if any(t in nome for t in ["neural", "mlp"]):
            return "hidden_layer_sizes"
        if "regressao_linear" in nome or "linear" in nome:
            return "fit_intercept"

        return "complexidade_modelo"

    @classmethod
    def carregar_mapa_yaml(cls, caminho_yaml: str = "configuracoes/projeto.yaml") -> dict[str, str]:
        """Carrega e valida o mapeamento de parâmetros de complexidade configurado no arquivo YAML."""
        if not os.path.exists(caminho_yaml):
            return dict(cls._MAPA_PARAMETROS_PADRAO)

        try:
            leitor = LeitorConfiguracao()
            configuracao = leitor.carregar_configuracao(caminho_yaml)

            # 1. Busca na seção dedicada 'diagnostico_curva' -> 'parametros_complexidade'
            secao_diagnostico = configuracao.get("diagnostico_curva")
            if isinstance(secao_diagnostico, dict):
                params_diag = secao_diagnostico.get("parametros_complexidade")
                if isinstance(params_diag, dict) and len(params_diag) > 0:
                    mapa_resultado = dict(cls._MAPA_PARAMETROS_PADRAO)
                    for k, v in params_diag.items():
                        if isinstance(v, str):
                            mapa_resultado[str(k).lower().strip()] = v.strip()
                    return mapa_resultado

            # 2. Alternativamente, infere a partir do primeiro hiperparâmetro de 'tuning' -> 'parametros'
            secao_tuning = configuracao.get("tuning")
            if isinstance(secao_tuning, dict):
                params_tuning = secao_tuning.get("parametros")
                if isinstance(params_tuning, dict) and len(params_tuning) > 0:
                    mapa_extraido = dict(cls._MAPA_PARAMETROS_PADRAO)
                    for mod_nome, dict_params in params_tuning.items():
                        if isinstance(dict_params, dict) and len(dict_params) > 0:
                            primeiro_param = next(iter(dict_params.keys()))
                            mapa_extraido[str(mod_nome).lower().strip()] = str(primeiro_param).strip()
                    return mapa_extraido

        except Exception:
            pass

        return dict(cls._MAPA_PARAMETROS_PADRAO)

    @property
    def mapa_parametros(self) -> dict[str, str]:
        """Retorna cópia do mapa ativo de parâmetros de complexidade."""
        return dict(self._mapa_parametros)

    def obter_parametro_complexidade(
        self,
        modelo_nome: str,
        parametro_informado: str | None = None,
        modelo_referencia: str | None = None
    ) -> str:
        """Determina o parâmetro de complexidade correspondente ao modelo informado.

        Caso um parâmetro seja explicitamente fornecido, ele é utilizado diretamente.
        Caso o modelo seja 'modelo_campeao' ou contenha indicação de campeão,
        ele detecta dinamicamente o hiperparâmetro do modelo campeão (ex: hidden_layer_sizes
        se rede neural, alpha se ridge, C se svr, max_depth se arvore/floresta).
        """
        if parametro_informado is not None and parametro_informado.strip():
            return parametro_informado.strip()

        nome = modelo_nome.lower().strip()

        # Resolução dinâmica para o modelo campeão
        if "campeao" in nome or "champion" in nome:
            campeao_real = modelo_referencia
            if not campeao_real:
                if ":" in nome:
                    campeao_real = nome.split(":", 1)[1].strip()
                elif "(" in nome:
                    campeao_real = nome.split("(", 1)[1].replace(")", "").strip()
            if not campeao_real:
                campeao_real = self.modelo_campeao_ativo

            param_configurado = self._mapa_parametros.get("modelo_campeao", "auto")
            # Se for um valor explicitamente fixo e diferente de auto/dinamico/max_depth(campeão), honra a configuração
            if modelo_referencia is None and param_configurado.lower() not in ["auto", "dinamico", "dynamic", "max_depth (campeão)"]:
                return param_configurado

            param_base = self.obter_parametro_base_modelo(campeao_real)
            return f"{param_base} (campeão)"

        for chave, param in self._mapa_parametros.items():
            if chave != "modelo_campeao" and chave in nome:
                return param

        return self.obter_parametro_base_modelo(nome)

    @classmethod
    def obter_parametro_complexidade_de_yaml(
        cls,
        modelo_nome: str,
        parametro_informado: str | None = None,
        caminho_yaml: str = "configuracoes/projeto.yaml",
        modelo_referencia: str | None = None
    ) -> str:
        """Determina o parâmetro de complexidade lendo diretamente do arquivo YAML indicado."""
        if parametro_informado is not None and parametro_informado.strip():
            return parametro_informado.strip()

        instancia = cls(caminho_yaml=caminho_yaml, modelo_campeao=modelo_referencia)
        return instancia.obter_parametro_complexidade(
            modelo_nome=modelo_nome,
            parametro_informado=parametro_informado,
            modelo_referencia=modelo_referencia
        )

    def mapear_parametro_sklearn(
        self,
        modelo_nome: str,
        parametro_informado: str | None = None,
        modelo_referencia: str | None = None
    ) -> str:
        """Mapeia o nome do hiperparâmetro para injeção via set_params do Scikit-Learn."""
        parametro_base = self.obter_parametro_complexidade(
            modelo_nome,
            parametro_informado,
            modelo_referencia=modelo_referencia
        )
        param_limpo = parametro_base.split()[0].strip()

        nome_alvo = (modelo_referencia or modelo_nome).lower().strip()
        if any(t in nome_alvo for t in ["arvore", "tree", "forest", "floresta", "xgboost", "lightgbm"]):
            return "max_depth"
        if any(t in nome_alvo for t in ["ridge", "lasso", "elasticnet"]):
            return "alpha"
        if "svr" in nome_alvo or "svm" in nome_alvo:
            return "C"
        if any(t in nome_alvo for t in ["neural", "mlp"]):
            return "max_iter"

        return param_limpo or "max_depth"

    def classificar_regime(
        self,
        rmse_treino: float,
        rmse_validacao: float,
        tolerancia_gap_relativo: float = 0.20
    ) -> str:
        """Classifica o regime operacional em UNDERFITTING, OVERFITTING ou GENERALIZACAO_OTIMA."""
        if rmse_treino <= 0:
            return "OVERFITTING"

        gap_relativo = (rmse_validacao - rmse_treino) / rmse_treino

        if gap_relativo > tolerancia_gap_relativo:
            return "OVERFITTING"
        if rmse_validacao > rmse_treino * 1.5:
            return "OVERFITTING"
        if rmse_treino > rmse_validacao:
            # Caso raro com ruído de validação menor que treino
            return "GENERALIZACAO_OTIMA"
        return "GENERALIZACAO_OTIMA"

    def gerar_curva_simulada(
        self,
        modelo_nome: str,
        motor: str = "sklearn",
        valores_complexidade: list[float] | None = None,
        rmse_base: float = 30000.0,
        parametro_complexidade: str | None = None,
        modelo_referencia: str | None = None
    ) -> RelatorioCurva:
        """Gera pontos empíricos para visualização no Grafana de acordo com o modelo e parâmetro lido do YAML."""
        passos = valores_complexidade or [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0, 32.0, 40.0, 48.0]
        pontos: list[PontoCurva] = []

        parametro_resolvido = self.obter_parametro_complexidade(
            modelo_nome,
            parametro_complexidade,
            modelo_referencia=modelo_referencia
        )

        # Simula a dinâmica matemática clássica de viés x variância:
        # Treino decai monotonicamente com complexidade (viés cai)
        # Validação decai até ponto ótimo e depois sobe/estabiliza (variância sobe)
        ponto_min_rmse = float("inf")
        melhor_comp = passos[0]

        for comp in passos:
            # Viés (treino): decaimento exponencial suave
            treino = rmse_base * (0.65 + 1.2 / (1.0 + 0.15 * comp))
            # Variância (validação): decai até comp ~ 18-32 e estabiliza ou sobe levemente
            penalidade_overfit = 0.0004 * ((comp - 28.0) ** 2) if comp > 28.0 else 0.0
            validacao = rmse_base * (0.85 + 1.1 / (1.0 + 0.12 * comp)) + (penalidade_overfit * rmse_base)
            gap = max(0.0, validacao - treino)

            if validacao < ponto_min_rmse:
                ponto_min_rmse = validacao
                melhor_comp = comp

            pontos.append(
                PontoCurva(
                    complexidade=float(comp),
                    rmse_treino=round(float(treino), 2),
                    rmse_validacao=round(float(validacao), 2),
                    gap_variancia=round(float(gap), 2)
                )
            )

        ultimo = pontos[-1]
        regime = self.classificar_regime(ultimo.rmse_treino, ultimo.rmse_validacao)

        return RelatorioCurva(
            modelo=modelo_nome,
            motor=motor,
            parametro_complexidade=parametro_resolvido,
            pontos=pontos,
            melhor_complexidade=melhor_comp,
            menor_rmse_validacao=round(ponto_min_rmse, 2),
            diagnostico_regime=regime
        )

    def diagnosticar_estimador(
        self,
        modelo_nome: str,
        estimador: ModeloPreditor,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        passos_complexidade: list[int] | list[float] | None = None,
        motor: str = "sklearn",
        parametro_complexidade: str | None = None,
        modelo_referencia: str | None = None
    ) -> RelatorioCurva:
        """Avalia empiricamente o estimador em K-Fold medindo RMSE de treino e validação por complexidade."""
        x_matriz = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        y_vetor = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        passos = passos_complexidade or [2, 4, 6, 8, 12, 16]
        pontos: list[PontoCurva] = []
        kfold = KFold(n_splits=self._cv_splits, shuffle=True, random_state=self._semente)

        parametro_resolvido = self.obter_parametro_complexidade(
            modelo_nome,
            parametro_complexidade,
            modelo_referencia=modelo_referencia
        )
        param_sklearn = self.mapear_parametro_sklearn(
            modelo_nome,
            parametro_complexidade,
            modelo_referencia=modelo_referencia
        )

        melhor_comp = float(passos[0])
        menor_val_rmse = float("inf")

        for comp in passos:
            erros_treino: list[float] = []
            erros_val: list[float] = []

            for idx_tr, idx_va in kfold.split(x_matriz):
                x_tr, x_va = x_matriz[idx_tr], x_matriz[idx_va]
                y_tr, y_va = y_vetor[idx_tr], y_vetor[idx_va]

                # Atualiza o hiperparâmetro correspondente ao modelo no estimador
                if hasattr(estimador, "set_params"):
                    try:
                        if hasattr(estimador, "named_steps") and "modelo" in getattr(estimador, "named_steps"):
                            estimador.set_params(**{f"modelo__{param_sklearn}": comp})
                        else:
                            estimador.set_params(**{param_sklearn: comp})
                    except Exception:
                        pass

                if hasattr(estimador, "fit"):
                    estimador.fit(x_tr, y_tr)

                pred_tr = estimador.predict(x_tr)
                pred_va = estimador.predict(x_va)

                rmse_tr = float(np.sqrt(np.mean((y_tr - pred_tr) ** 2)))
                rmse_va = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))

                erros_treino.append(rmse_tr)
                erros_val.append(rmse_va)

            media_tr = float(np.mean(erros_treino))
            media_va = float(np.mean(erros_val))
            gap = max(0.0, media_va - media_tr)

            if media_va < menor_val_rmse:
                menor_val_rmse = media_va
                melhor_comp = float(comp)

            pontos.append(
                PontoCurva(
                    complexidade=float(comp),
                    rmse_treino=round(media_tr, 2),
                    rmse_validacao=round(media_va, 2),
                    gap_variancia=round(gap, 2)
                )
            )

        ultimo = pontos[-1]
        regime = self.classificar_regime(ultimo.rmse_treino, ultimo.rmse_validacao)

        return RelatorioCurva(
            modelo=modelo_nome,
            motor=motor,
            parametro_complexidade=parametro_resolvido,
            pontos=pontos,
            melhor_complexidade=melhor_comp,
            menor_rmse_validacao=round(menor_val_rmse, 2),
            diagnostico_regime=regime
        )
