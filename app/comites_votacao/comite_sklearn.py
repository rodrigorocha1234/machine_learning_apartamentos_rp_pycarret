"""Implementação do construtor de comitê de votação baseado em Scikit-Learn."""

from __future__ import annotations

from typing import Protocol, cast, final, override
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge

from app.contratos_avaliacao.construtor_comite import ConstrutorComite, ResultadoComite
from app.esquema_configuracao import ConfiguracaoVotacao


class Preditor(Protocol):
    """Protocolo com método predict para estimadores treinados."""

    def predict(self, X: pd.DataFrame) -> object: ...


class ErroComiteSklearn(Exception):
    """Exceção levantada em falhas na construção ou avaliação do comitê de votação."""


@final
class ComiteSklearn(ConstrutorComite):
    """Montador e avaliador de ensembles (média simples, ponderada, voting e stacking)."""

    @override
    def construir_e_avaliar(
        self,
        modelos_treinados: dict[str, object],
        dados_treino: pd.DataFrame,
        dados_holdout: pd.DataFrame,
        coluna_alvo: str,
        config_votacao: ConfiguracaoVotacao,
    ) -> ResultadoComite:
        """Monta o ensemble configurado e avalia sua performance no holdout isolado."""
        if not config_votacao.ativo:
            return ResultadoComite(
                tecnica_utilizada="desativada",
                ativo=False,
                modelos_membros=[],
                pesos={},
                metricas_holdout={},
            )

        if not modelos_treinados:
            raise ErroComiteSklearn("Nenhum modelo disponível para composição do comitê.")

        # Seleciona até n modelos
        n_modelos = min(config_votacao.melhores_n, len(modelos_treinados))
        nomes_selecionados = list(modelos_treinados.keys())[:n_modelos]
        modelos_comite = {nome: modelos_treinados[nome] for nome in nomes_selecionados}

        X_holdout = dados_holdout.drop(columns=[coluna_alvo for coluna_alvo in [coluna_alvo]])
        y_holdout = np.asarray(dados_holdout[coluna_alvo].to_numpy(), dtype=np.float64)

        X_tr = dados_treino.drop(columns=[coluna_alvo for coluna_alvo in [coluna_alvo]])
        y_tr = cast(pd.Series, dados_treino[coluna_alvo])

        tecnica = config_votacao.tecnica
        preds_individuais: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = {}
        for nome, mod in modelos_comite.items():
            preditor = cast(Preditor, mod)
            preds_individuais[nome] = np.asarray(preditor.predict(X_holdout), dtype=np.float64)

        pesos: dict[str, float] = {}

        if tecnica == "media_simples":
            # Média aritmética direta
            soma_preds = np.zeros_like(y_holdout)
            for nome, pred_arr in preds_individuais.items():
                soma_preds += pred_arr
                pesos[nome] = round(1.0 / len(modelos_comite), 4)
            preds_ensemble = soma_preds / len(modelos_comite)

        elif tecnica == "media_ponderada_rmse":
            # Ponderação inversamente proporcional ao erro individual no holdout
            inversos: dict[str, float] = {}
            for nome, pred_arr in preds_individuais.items():
                rmse_indiv = float(np.sqrt(np.mean((y_holdout - pred_arr) ** 2)))
                inversos[nome] = 1.0 / max(rmse_indiv, 1.0)
            total_inv = sum(inversos.values())
            for nome, inv_val in inversos.items():
                pesos[nome] = round(inv_val / total_inv, 4)

            preds_ensemble = np.zeros_like(y_holdout)
            for nome, pred_arr in preds_individuais.items():
                preds_ensemble += pesos[nome] * pred_arr

        elif tecnica == "voting_regressor":
            lista_estimadores = [(nome, mod) for nome, mod in modelos_comite.items()]
            voting = VotingRegressor(estimators=lista_estimadores)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            voting.fit(X_tr, y_tr)  # pyright: ignore[reportUnknownMemberType]
            preds_ensemble = np.asarray(voting.predict(X_holdout), dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
            for nome in nomes_selecionados:
                pesos[nome] = round(1.0 / len(nomes_selecionados), 4)

        elif tecnica == "stacking":
            meta = Ridge() if config_votacao.meta_modelo == "ridge" else LinearRegression()
            lista_estimadores = [(nome, mod) for nome, mod in modelos_comite.items()]
            stacking = StackingRegressor(estimators=lista_estimadores, final_estimator=meta, cv=3)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            stacking.fit(X_tr, y_tr)  # pyright: ignore[reportUnknownMemberType]
            preds_ensemble = np.asarray(stacking.predict(X_holdout), dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
            for nome in nomes_selecionados:
                pesos[nome] = round(1.0 / len(nomes_selecionados), 4)

        else:
            raise ErroComiteSklearn(f"Técnica de votação desconhecida: '{tecnica}'.")

        # Avaliação no holdout
        residuos = y_holdout - preds_ensemble
        rmse_ens = float(np.sqrt(np.mean(residuos**2)))
        mae_ens = float(np.mean(np.abs(residuos)))
        var_y = float(np.var(y_holdout))
        r2_ens = float(1.0 - (float(np.mean(residuos**2)) / var_y)) if var_y > 0 else 0.0

        return ResultadoComite(
            tecnica_utilizada=tecnica,
            ativo=True,
            modelos_membros=nomes_selecionados,
            pesos=pesos,
            metricas_holdout={
                "RMSE": round(rmse_ens, 2),
                "MAE": round(mae_ens, 2),
                "R2": round(r2_ens, 4),
            },
        )
