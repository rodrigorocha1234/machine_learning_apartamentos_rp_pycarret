from __future__ import annotations
import os
from typing import final
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from app.analises_eda.analise_exploratoria import AnaliseExploratoria
from app.analises_estatisticas.teste_friedman import TesteFriedman
from app.configuracao_projeto.leitor_configuracao import LeitorConfiguracao, ValorConfiguracao
from app.contratos_dados.fonte_dados import FonteDados
from app.contratos_dados.motor_tabela import MotorTabela
from app.contratos_dados.motor_treino import ModeloPreditor, MotorTreino, ValorMetricaParametro
from app.contratos_dados.regra_negocio import DefinicaoRegra
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.fontes_dados.fabrica_fontes import FabricaFontes
from app.modelos_votacao.fabrica_votacao import FabricaVotacao
from app.motores_tabela.fabrica_motores import FabricaMotores
from app.motores_treino.motor_pycaret import MotorPycaret
from app.motores_treino.motor_sklearn import MotorSklearn
from app.orquestracao_ml.modelo_servico import ModeloServico
from app.orquestracao_ml.observador_auditoria import ObservadorAuditoria
from app.orquestracao_ml.observador_metricas import ObservadorMetricas
from app.orquestracao_ml.publicador_servico import PublicadorServico
from app.repositorios_dados.repositorio_sqlite import RepositorioSqlite




type ValorResultadoPipeline = (
    str
    | int
    | float
    | bool
    | None
    | list[str]
    | dict[str, str | int | float | bool | dict[str, float]]
    | dict[str, str | list[str]]
    | dict[str, str]
    | dict[str, ValorMetricaParametro]
)

class ErroPipeline(Exception):
    """Exceção levantada em falhas de execução do pipeline integrado."""
    pass

@final
class MainPipeline:
    """Facade central que agrega ingestão, staging, EDA, modelagem, estatística e MLflow Serving."""

    def __init__(self, caminho_configuracao: str = "configuracoes/projeto.yaml") -> None:
        self._caminho_configuracao = caminho_configuracao
        self._leitor = LeitorConfiguracao()

    def executar_pipeline(
        self,
        modo: str | None = None,
        modelo_campeao_forcado: str | None = None
    ) -> dict[str, ValorResultadoPipeline]:
        # 1. Carrega e valida configurações com variáveis de ambiente
        config = self._leitor.carregar_configuracao(self._caminho_configuracao)
        config_projeto = config.get("projeto", {})
        alvo = str(config_projeto.get("alvo", "Valor_da_Venda"))
        semente = int(config_projeto.get("semente", 42))

        # Determina modo de execução (completo vs apenas_campeao)
        config_pipeline = config.get("pipeline", {})
        modo_efetivo = str(config_pipeline.get("modo", "completo")).lower() if modo is None else modo.lower()

        # 2. Ingestão via FonteDados (polimórfica via FabricaFontes: Excel, SQL, API)
        config_fonte = config.get("fonte", {})
        fonte: FonteDados = FabricaFontes.criar_fonte(config_fonte)
        dados_brutos = fonte.carregar_dados()

        # 3. Processamento Tabular com MotorTabular (Pandas ou Polars via FabricaMotores)
        config_tabular = config.get("tabular", {})
        identificador_motor = str(config_tabular.get("motor", "pandas")).lower()
        motor_tabela: MotorTabela = FabricaMotores.criar_motor(identificador_motor)
        dados_processados = (
            dados_brutos.to_pandas() if hasattr(dados_brutos, "to_pandas") else dados_brutos.copy()
        )



        def extrair_expressao_motor(item: dict[str, ValorConfiguracao], motor: str) -> str:
            campo_especifico = f"expressao_{motor}"
            if campo_especifico in item and isinstance(item[campo_especifico], str):
                return item[campo_especifico]
            valor_expressao = item.get("expressao")
            if isinstance(valor_expressao, dict):
                if motor in valor_expressao and isinstance(valor_expressao[motor], str):
                    return valor_expressao[motor]
                if "padrao" in valor_expressao and isinstance(valor_expressao["padrao"], str):
                    return valor_expressao["padrao"]
                if "pandas" in valor_expressao and isinstance(valor_expressao["pandas"], str):
                    return valor_expressao["pandas"]
                return ""
            if isinstance(valor_expressao, str):
                return valor_expressao
            return ""

        # Aplica filtros ativos
        config_filtros = config.get("filtros", {})
        if config_filtros.get("ativo", False):
            for item in config_filtros.get("expressoes", []):
                if item.get("ativo", False):
                    expr = extrair_expressao_motor(item, identificador_motor)
                    if expr.strip():
                        dados_processados = motor_tabela.filtrar_dados(dados_processados, expr)

        # Aplica transformações ativas
        config_transformacoes = config.get("transformacoes", {})
        if config_transformacoes.get("ativo", False):
            for item in config_transformacoes.get("expressoes", []):
                if item.get("ativo", False) and "destino" in item:
                    expr = extrair_expressao_motor(item, identificador_motor)
                    if expr.strip():
                        dados_processados = motor_tabela.transformar_dados(
                            dados_processados,
                            expr,
                            str(item["destino"])
                        )

        # Aplica novos atributos ativos
        config_atributos = config.get("atributos_novos", {})
        if config_atributos.get("ativo", False):
            for item in config_atributos.get("expressoes", []):
                if item.get("ativo", False) and "destino" in item:
                    expr = extrair_expressao_motor(item, identificador_motor)
                    if expr.strip():
                        dados_processados = motor_tabela.transformar_dados(
                            dados_processados,
                            expr,
                            str(item["destino"])
                        )

        # Seleciona colunas ativas
        config_selecao = config.get("selecao", {})
        if config_selecao.get("ativo", False):
            colunas_ativas = [
                c["nome"] for c in config_selecao.get("colunas", [])
                if c.get("ativo", False) and c["nome"] in dados_processados.columns
            ]
            if colunas_ativas:
                dados_processados = motor_tabela.selecionar_colunas(dados_processados, colunas_ativas)

        # 4. Staging SQLite via RepositorioDados
        config_staging = config.get("staging", {})
        caminho_staging = str(config_staging.get("url", "storage-data/staging_imoveis.db"))
        tabela_staging = str(config_staging.get("tabela", "imoveis_preparados"))

        repositorio: RepositorioDados = RepositorioSqlite(caminho=caminho_staging, tabela=tabela_staging)
        dados_staging = dados_processados.to_pandas() if hasattr(dados_processados, "to_pandas") else dados_processados
        repositorio.gravar_dados(dados_staging)


        # 5. EDA a partir do staging SQLite
        dados_staging = repositorio.ler_dados()
        eda = AnaliseExploratoria(repositorio)
        relatorio_eda = eda.executar_eda(coluna_alvo=alvo)
        
        caminho_eda_json = "storage-data/relatorio_eda.json"
        eda.salvar_relatorio_json(relatorio_eda, caminho_eda_json)

        # 6. Segregação rigorosa de Holdout (Anti-Leakage)
        config_val = config.get("validacao", {})
        fracao_holdout = float(config_val.get("holdout", 0.2))

        df_dev, df_holdout = train_test_split(
            dados_staging,
            test_size=fracao_holdout,
            random_state=semente
        )

        # 7. Seleção do Motor de Treino (PyCaret ou Sklearn, isolados)
        config_framework = config.get("framework", {})
        motor_nome = str(config_framework.get("motor", "sklearn")).lower()

        colunas_num = [
            c["nome"] for c in config.get("selecao", {}).get("colunas", [])
            if c.get("papel") == "numerica" and c.get("ativo", False)
        ]
        colunas_cat = [
            c["nome"] for c in config.get("selecao", {}).get("colunas", [])
            if c.get("papel") == "categorica" and c.get("ativo", False)
        ]
        config_preprocessamento = config.get("preprocessamento", {})
        config_prep = config_preprocessamento.get("modelos", {})
        transformar_alvo = bool(config_preprocessamento.get("transformar_alvo", False))
        metodo_transformacao_alvo = str(config_preprocessamento.get("metodo_transformacao_alvo", "log1p"))

        config_transf_attr = config_preprocessamento.get("transformar_atributos", {})
        transformar_atributos = bool(config_transf_attr.get("ativo", False))
        metodo_transformacao_atributos = str(config_transf_attr.get("metodo", "log1p"))
        colunas_transf_attr_raw = config_transf_attr.get("colunas", ["Metragem"])
        colunas_transf_attr = (
            [str(c) for c in colunas_transf_attr_raw]
            if isinstance(colunas_transf_attr_raw, list)
            else None
        )

        config_cv = config_val.get("cv_interna", {})
        metodo_folds = str(config_cv.get("metodo", "kfold"))
        splits_folds = int(config_cv.get("splits", 3))
        embaralhar_folds = bool(config_cv.get("embaralhar", True))

        modelos_configurados = config.get("modelos", [])
        config_tuning = config.get("tuning", {})
        parametros_tuning = config_tuning.get("parametros", {})
        iteracoes = int(config_tuning.get("iteracoes", 5))
        metodo_tuning = str(config_tuning.get("metodo", "random"))

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


        config_mlflow = config.get("mlflow", {})
        tracking_uri = str(config_mlflow.get("tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")))

        # 8, 9, 10. Ramificação de acordo com o modo de execução
        resultado_tuning: dict[str, ValorMetricaParametro] = {}
        resultado_estatistico_dict: dict[str, str | int | float | bool | dict[str, float]] = {}

        if modo_efetivo == "apenas_campeao":
            publicador = PublicadorServico(tracking_uri=tracking_uri)
            if modelo_campeao_forcado:
                modelo_campeao_nome = modelo_campeao_forcado
            else:
                modelo_campeao_padrao = str(
                    config.get("pipeline", {}).get(
                        "modelo_campeao_padrao",
                        config_mlflow.get("modelo_campeao", "random_forest")
                    )
                )
                nome_registrado = str(config_mlflow.get("modelo_registrado", "previsao_preco_imoveis"))
                modelo_campeao_nome = publicador.obter_modelo_campeao_salvo(
                    nome_modelo_registrado=nome_registrado,
                    fallback_padrao=modelo_campeao_padrao
                )

            # Treina diretamente o modelo campeão na base completa de staging (sem CV e sem Tuning)
            modelo_final = motor_treino.treinar_final(dados_staging, alvo=alvo, modelo_nome=modelo_campeao_nome)
            resultado_tuning = {
                "modo": "apenas_campeao",
                "mensagem": "Validacao cruzada pareada e tuning ignorados; modelo campeao treinado diretamente.",
                "modelo_campeao": modelo_campeao_nome
            }
            resultado_estatistico_dict = {
                "modo": "apenas_campeao",
                "modelo_campeao": modelo_campeao_nome,
                "rejeitou_nula": False
            }
        else:
            # 8. Validação Cruzada Pareada de Todos os Modelos Ativos
            perdas_pareadas = motor_treino.avaliar_modelos(df_dev, alvo=alvo)

            # 9. Camada Estatística Independente (Friedman e Pós-Hoc Pareado)
            avaliador_estatistico = TesteFriedman()
            config_estatistica = config.get("estatistica", {})
            alpha = float(config_estatistica.get("alpha", 0.05))
            resultado_estatistico = avaliador_estatistico.testar_diferencas(perdas_pareadas, alpha=alpha)
            modelo_campeao_nome = resultado_estatistico.modelo_campeao
            resultado_estatistico_dict = {
                "estatistica_friedman": resultado_estatistico.estatistica_friedman,
                "valor_p_friedman": resultado_estatistico.valor_p_friedman,
                "rejeitou_nula": resultado_estatistico.hipotese_nula_rejeitada,
                "postos_medios": resultado_estatistico.postos_medios.to_dict(),
                "modelo_campeao": modelo_campeao_nome
            }

            # 10. Ensemble de Votação (caso ativo) ou Tuning do Modelo Campeão Individual
            config_votacao = config.get("votacao", {})
            usar_votacao = bool(config_votacao.get("ativo", False))

            if usar_votacao:
                tecnica_votacao = str(config_votacao.get("tecnica", "media_simples"))
                melhores_n = int(config_votacao.get("melhores_n", 3))

                # Ordena os modelos pelos postos médios de Friedman (menor posto = melhor)
                modelos_ordenados = list(resultado_estatistico.postos_medios.index)
                top_modelos_nomes = modelos_ordenados[:melhores_n]

                # Treina os N melhores modelos na base completa
                estimadores_top: list[tuple[str, ModeloPreditor]] = []
                for nome_m in top_modelos_nomes:
                    est = motor_treino.treinar_final(dados_staging, alvo=alvo, modelo_nome=nome_m)
                    estimadores_top.append((nome_m, est))

                metricas_rmse = perdas_pareadas.mean().to_dict()
                postos_friedman = resultado_estatistico.postos_medios.to_dict()

                modelo_final = FabricaVotacao.criar_ensemble_votacao(
                    estimadores=estimadores_top,
                    tecnica=tecnica_votacao,
                    metricas_rmse=metricas_rmse,
                    postos_friedman=postos_friedman,
                    cv_folds=splits_folds
                )
                modelo_campeao_nome = f"ensemble_{tecnica_votacao}"
                resultado_estatistico_dict["modelo_campeao"] = modelo_campeao_nome
                resultado_tuning = {
                    "ensemble": tecnica_votacao,
                    "modelos_componentes": top_modelos_nomes
                }

            else:
                # Tuning de Hiperparâmetros do Modelo Campeão Eleito Estatisticamente
                resultado_tuning = motor_treino.ajustar_parametros(
                    df_dev,
                    alvo=alvo,
                    iteracoes=iteracoes,
                    modelo_nome=modelo_campeao_nome,
                    metodo_tuning=metodo_tuning,
                )
                # Retreino final na base completa para produção
                modelo_final = motor_treino.treinar_final(dados_staging, alvo=alvo, modelo_nome=modelo_campeao_nome)

        # 11. Montagem do ModeloServico com Regras de Negócio Dinâmicas
        config_regras = config.get("regras_negocio", {})
        coluna_alvo_predito = str(config_regras.get("coluna_alvo_predito", "preco_predito"))
        
        lista_regras: list[DefinicaoRegra] = []
        for r in config_regras.get("regras", []):
            lista_regras.append(
                DefinicaoRegra(
                    nome=r.get("nome", ""),
                    ativo=bool(r.get("ativo", True)),
                    descricao=r.get("descricao", ""),
                    destino=r.get("destino", ""),
                    expressao=r.get("expressao", "")
                )
            )

        servico_mlflow = ModeloServico(
            modelo_campeao=modelo_final,
            regras=lista_regras,
            coluna_alvo_predito=coluna_alvo_predito
        )

        # 12. Publicação e Registro no MLflow Tracking & Model Registry
        info_publicacao: dict[str, str] = {
            "status": "ignorado",
            "mensagem": "Registro de modelo no MLflow desativado na configuracao."
        }
        if bool(config_mlflow.get("registrar_modelo_campeao", True)):
            try:
                publicador = PublicadorServico(tracking_uri=tracking_uri)
                publicador.adicionar_observador(ObservadorAuditoria())
                publicador.adicionar_observador(ObservadorMetricas())
                tags = {
                    "modelo_campeao": modelo_campeao_nome,
                    "modo_execucao": modo_efetivo,
                    "framework": motor_nome,
                    "transformar_alvo": str(transformar_alvo),
                    "metodo_transformacao_alvo": metodo_transformacao_alvo,
                    "transformar_atributos": str(transformar_atributos),
                    "metodo_transformacao_atributos": metodo_transformacao_atributos,
                }
                parametros_log: dict[str, str | int | float] = {
                    "modelo_campeao": modelo_campeao_nome,
                    "modo_execucao": modo_efetivo,
                    "linhas_staging": len(dados_staging),
                    "colunas_staging": len(dados_staging.columns),
                    "transformar_alvo": str(transformar_alvo),
                    "metodo_transformacao_alvo": metodo_transformacao_alvo,
                    "transformar_atributos": str(transformar_atributos),
                    "metodo_transformacao_atributos": metodo_transformacao_atributos,
                }
                info_publicacao = publicador.publicar_modelo_servico(
                    modelo_servico=servico_mlflow,
                    experimento=str(config_mlflow.get("experimento", "imoveis_regressao")),
                    nome_modelo_registrado=str(config_mlflow.get("modelo_registrado", "previsao_preco_imoveis")),
                    tags=tags,
                    parametros=parametros_log
                )
            except Exception as erro:
                info_publicacao = {
                    "status": "erro_registro",
                    "detalhe": str(erro)
                }

        return {
            "status": "sucesso",
            "modo": modo_efetivo,
            "linhas_processadas": len(dados_staging),
            "colunas_processadas": len(dados_staging.columns),
            "tuning": resultado_tuning,
            "estatistica": resultado_estatistico_dict,
            "servico_mlflow_pronto": servico_mlflow is not None,
            "publicacao_mlflow": info_publicacao,
            "transformacao_alvo": {
                "ativo": transformar_alvo,
                "metodo": metodo_transformacao_alvo,
            },
            "transformacao_atributos": {
                "ativo": transformar_atributos,
                "metodo": metodo_transformacao_atributos,
                "colunas": colunas_transf_attr,
            }
        }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Executa o pipeline completo de treinamento e avaliação imobiliária.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configuracoes/projeto.yaml",
        help="Caminho para o arquivo de configuração YAML (padrão: configuracoes/projeto.yaml)"
    )
    parser.add_argument(
        "--modo",
        "-m",
        type=str,
        choices=["completo", "apenas_campeao"],
        default=None,
        help="Modo de execução: 'completo' (com CV e tuning) ou 'apenas_campeao' (treino direto sem CV e sem tuning)"
    )
    parser.add_argument(
        "--apenas-campeao",
        action="store_true",
        help="Atalho para executar em modo 'apenas_campeao' (pula validação cruzada pareada e tuning de parâmetros)"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        help="Nome específico do modelo campeão para treinar diretamente (ex: random_forest, lightgbm)"
    )
    parser.add_argument(
        "--subir-api",
        action="store_true",
        help="Sinaliza intenção de inicialização do MLflow Serving"
    )
    args = parser.parse_args()

    modo_selecionado = "apenas_campeao" if args.apenas_campeao else args.modo

    print("=" * 70)
    print(" INICIANDO PIPELINE DE TREINAMENTO — IMÓVEIS RIBEIRÃO PRETO")
    print("=" * 70)
    print(f"Arquivo de configuração : {args.config}")
    print(f"Modo de execução        : {modo_selecionado or 'definido pelo YAML'}")
    if args.modelo:
        print(f"Modelo forçado          : {args.modelo}")
    print("=" * 70 + "\n")

    try:
        pipeline = MainPipeline(caminho_configuracao=args.config)
        resultado = pipeline.executar_pipeline(
            modo=modo_selecionado,
            modelo_campeao_forcado=args.modelo
        )
        print("\n" + "=" * 70)
        print(" TREINAMENTO E REGISTRO CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as erro:
        print(f"\n[ERRO CRÍTICO NO PIPELINE]: {erro}", file=sys.stderr)
        sys.exit(1)

