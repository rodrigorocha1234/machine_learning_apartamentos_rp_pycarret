from __future__ import annotations
from typing import final
from app.configuracao_projeto.leitor_configuracao import ValorConfiguracao
from app.contratos_dados.motor_treino import MotorTreino
from app.motores_treino.motor_pycaret import MotorPycaret
from app.motores_treino.motor_sklearn import MotorSklearn


@final
class PreparadorTreino:
    """Instancia e parametriza o motor de treinamento apropriado (PyCaret ou Scikit-Learn) a partir do YAML."""

    def criar_motor(
        self,
        config: dict[str, ValorConfiguracao],
        semente: int = 42,
    ) -> tuple[MotorTreino, str, bool, str, bool, str, list[str] | None]:
        """Cria a instância isolada do motor de treino com base nas definições declaradas no YAML.

        Args:
            config: Dicionário completo de configurações do projeto.
            semente: Semente pseudoaleatória para reprodutibilidade estrita.

        Returns:
            Tupla contendo:
            (motor_treino, motor_nome, transformar_alvo, metodo_alvo, transformar_atributos, metodo_atributos, colunas_atributos)
        """
        config_framework = config.get("framework", {})
        motor_nome = (
            str(config_framework.get("motor", "sklearn")).lower()
            if isinstance(config_framework, dict)
            else "sklearn"
        )

        config_selecao = config.get("selecao", {})
        colunas_lista = (
            config_selecao.get("colunas", [])
            if isinstance(config_selecao, dict) and isinstance(config_selecao.get("colunas"), list)
            else []
        )

        colunas_num = [
            str(c["nome"]) for c in colunas_lista
            if isinstance(c, dict) and c.get("papel") == "numerica" and c.get("ativo", False)
        ]
        colunas_cat = [
            str(c["nome"]) for c in colunas_lista
            if isinstance(c, dict) and c.get("papel") == "categorica" and c.get("ativo", False)
        ]

        config_preprocessamento = config.get("preprocessamento", {})
        config_prep = (
            config_preprocessamento.get("modelos", {})
            if isinstance(config_preprocessamento, dict) and isinstance(config_preprocessamento.get("modelos"), dict)
            else {}
        )
        transformar_alvo = (
            bool(config_preprocessamento.get("transformar_alvo", False))
            if isinstance(config_preprocessamento, dict)
            else False
        )
        metodo_transformacao_alvo = (
            str(config_preprocessamento.get("metodo_transformacao_alvo", "log1p"))
            if isinstance(config_preprocessamento, dict)
            else "log1p"
        )

        config_transf_attr = (
            config_preprocessamento.get("transformar_atributos", {})
            if isinstance(config_preprocessamento, dict) and isinstance(config_preprocessamento.get("transformar_atributos"), dict)
            else {}
        )
        transformar_atributos = bool(config_transf_attr.get("ativo", False))
        metodo_transformacao_atributos = str(config_transf_attr.get("metodo", "log1p"))
        colunas_transf_attr_raw = config_transf_attr.get("colunas", ["Metragem"])
        colunas_transf_attr = (
            [str(c) for c in colunas_transf_attr_raw]
            if isinstance(colunas_transf_attr_raw, list)
            else None
        )

        config_val = config.get("validacao", {})
        config_cv = (
            config_val.get("cv_interna", {})
            if isinstance(config_val, dict) and isinstance(config_val.get("cv_interna"), dict)
            else {}
        )
        metodo_folds = str(config_cv.get("metodo", "kfold"))
        splits_folds_val = config_cv.get("splits", 3)
        splits_folds = int(splits_folds_val) if isinstance(splits_folds_val, (int, float, str)) else 3
        embaralhar_folds = bool(config_cv.get("embaralhar", True))

        modelos_configurados_raw = config.get("modelos", [])
        modelos_configurados = modelos_configurados_raw if isinstance(modelos_configurados_raw, list) else []

        config_tuning = config.get("tuning", {})
        parametros_tuning = (
            config_tuning.get("parametros", {})
            if isinstance(config_tuning, dict) and isinstance(config_tuning.get("parametros"), dict)
            else {}
        )
        metodo_tuning = (
            str(config_tuning.get("metodo", "random"))
            if isinstance(config_tuning, dict)
            else "random"
        )

        motor_treino: MotorTreino
        if motor_nome == "pycaret":
            motor_treino = MotorPycaret(
                semente=semente,
                folds=splits_folds,
                metodo_folds=metodo_folds,
                embaralhar_folds=embaralhar_folds,
                colunas_numericas=colunas_num,
                colunas_categoricas=colunas_cat,
                preprocessamento_por_modelo=config_prep,
                modelos_configurados=modelos_configurados,
                parametros_tuning=parametros_tuning,
                metodo_tuning=metodo_tuning,
                transformar_alvo=transformar_alvo,
                metodo_transformacao_alvo=metodo_transformacao_alvo,
                transformar_atributos=transformar_atributos,
                metodo_transformacao_atributos=metodo_transformacao_atributos,
                colunas_transformacao_atributos=colunas_transf_attr,
            )
        else:
            motor_treino = MotorSklearn(
                semente=semente,
                folds=splits_folds,
                metodo_folds=metodo_folds,
                embaralhar_folds=embaralhar_folds,
                colunas_numericas=colunas_num,
                colunas_categoricas=colunas_cat,
                preprocessamento_por_modelo=config_prep,
                modelos_configurados=modelos_configurados,
                parametros_tuning=parametros_tuning,
                metodo_tuning=metodo_tuning,
                transformar_alvo=transformar_alvo,
                metodo_transformacao_alvo=metodo_transformacao_alvo,
                transformar_atributos=transformar_atributos,
                metodo_transformacao_atributos=metodo_transformacao_atributos,
                colunas_transformacao_atributos=colunas_transf_attr,
            )

        return (
            motor_treino,
            motor_nome,
            transformar_alvo,
            metodo_transformacao_alvo,
            transformar_atributos,
            metodo_transformacao_atributos,
            colunas_transf_attr,
        )
