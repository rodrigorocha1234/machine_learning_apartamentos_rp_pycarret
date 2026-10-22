from __future__ import annotations
from typing import final
from app.configuracao_projeto.leitor_configuracao import ValorConfiguracao
from app.contratos_dados.motor_treino import ModeloPreditor
from app.contratos_dados.regra_negocio import DefinicaoRegra
from app.orquestracao_ml.modelo_servico import ModeloServico


@final
class MontadorServico:
    """Monta a instância envelopada de ModeloServico agregando regras de negócio declaradas no YAML."""

    def montar_servico(
        self,
        modelo_final: ModeloPreditor,
        config_regras: dict[str, ValorConfiguracao] | None = None,
    ) -> ModeloServico:
        """Instancia o ModeloServico com as regras de negócio em conformidade com o AST seguro.

        Args:
            modelo_final: Estimador campeão final treinado.
            config_regras: Configuração de regras de negócio extraída do arquivo de projeto.

        Returns:
            Instância pronta de ModeloServico para inferência e publicação no MLflow.
        """
        cfg = config_regras or {}
        coluna_alvo_predito = str(cfg.get("coluna_alvo_predito", "preco_predito"))

        regras_brutas = cfg.get("regras", [])
        lista_regras: list[DefinicaoRegra] = []

        if isinstance(regras_brutas, list):
            for r in regras_brutas:
                if isinstance(r, dict):
                    lista_regras.append(
                        DefinicaoRegra(
                            nome=str(r.get("nome", "")),
                            ativo=bool(r.get("ativo", True)),
                            descricao=str(r.get("descricao", "")),
                            destino=str(r.get("destino", "")),
                            expressao=str(r.get("expressao", "")),
                        )
                    )

        return ModeloServico(
            modelo_campeao=modelo_final,
            regras=lista_regras,
            coluna_alvo_predito=coluna_alvo_predito,
        )
