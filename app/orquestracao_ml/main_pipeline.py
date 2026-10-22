from __future__ import annotations
import os
from typing import final
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from app.analises_eda.analise_exploratoria import AnaliseExploratoria
from app.configuracao_projeto.leitor_configuracao import LeitorConfiguracao, ValorConfiguracao
from app.contratos_dados.fonte_dados import FonteDados
from app.contratos_dados.motor_tabela import MotorTabela
from app.contratos_dados.motor_treino import ModeloPreditor, MotorTreino, ValorMetricaParametro
from app.contratos_dados.repositorio_dados import RepositorioDados
from app.fontes_dados.fabrica_fontes import FabricaFontes
from app.motores_tabela.fabrica_motores import FabricaMotores
from app.orquestracao_ml.executor_avaliacao import ExecutorAvaliacao
from app.orquestracao_ml.modelo_servico import ModeloServico
from app.orquestracao_ml.montador_servico import MontadorServico
from app.orquestracao_ml.observador_auditoria import ObservadorAuditoria
from app.orquestracao_ml.observador_metricas import ObservadorMetricas
from app.orquestracao_ml.otimizador_campeao import OtimizadorCampeao
from app.orquestracao_ml.preparador_treino import PreparadorTreino
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
        self._preparador_treino = PreparadorTreino()
        self._executor_avaliacao = ExecutorAvaliacao()
        self._otimizador_campeao = OtimizadorCampeao()
        self._montador_servico = MontadorServico()

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
        config_eda = config.get("eda", {})
        ativo_eda = bool(config_eda.get("ativo", True))
        motor_eda = str(config_eda.get("motor", "sweetviz"))
        motor_tabular_eda = str(config_eda.get("motor_tabular", "pandas"))
        caminho_eda_json = str(config_eda.get("caminho_relatorio_json", "storage-data/relatorio_eda.json"))
        caminho_eda_html = str(config_eda.get("caminho_relatorio_html", "storage-data/relatorio_eda.html"))

        if ativo_eda:
            eda = AnaliseExploratoria(repositorio)
            relatorio_eda = eda.executar_eda(coluna_alvo=alvo)
            eda.salvar_relatorio_json(relatorio_eda, caminho_eda_json)
            analises_eda = config_eda.get("analises", [])
            try:
                eda.gerar_relatorio_visual(
                    coluna_alvo=alvo,
                    motor=motor_eda,
                    caminho_saida=caminho_eda_html,
                    analises_configuradas=analises_eda if isinstance(analises_eda, list) else None,
                    motor_tabular=motor_tabular_eda,
                )
            except Exception:
                if modo_efetivo in ("apenas_eda", "eda"):
                    raise
                # Falha pontual na geração visual não interrompe o treinamento
                pass

        # Retorno imediato quando o modo for apenas análise exploratória (EDA)
        if modo_efetivo in ("apenas_eda", "eda"):
            return {
                "modo": "apenas_eda",
                "status": "sucesso",
                "linhas_processadas": int(len(dados_staging)),
                "linhas_staging": int(len(dados_staging)),
                "colunas_staging": [str(c) for c in dados_staging.columns],
                "motor_eda": motor_eda,
                "motor_tabular": motor_tabular_eda,
                "eda": {
                    "ativo": ativo_eda,
                    "motor": motor_eda,
                    "motor_tabular": motor_tabular_eda,
                    "relatorio_html": caminho_eda_html,
                    "relatorio_json": caminho_eda_json,
                },
            }

        # 6. Segregação rigorosa de Holdout (Anti-Leakage)
        config_val = config.get("validacao", {})
        fracao_holdout = float(config_val.get("holdout", 0.2))

        df_dev, df_holdout = train_test_split(
            dados_staging,
            test_size=fracao_holdout,
            random_state=semente
        )

        # 7. Seleção do Motor de Treino via PreparadorTreino
        (
            motor_treino,
            motor_nome,
            transformar_alvo,
            metodo_transformacao_alvo,
            transformar_atributos,
            metodo_transformacao_atributos,
            colunas_transf_attr,
        ) = self._preparador_treino.criar_motor(config, semente=semente)

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
            # 8 & 9. Validação Cruzada Pareada e Inferência Estatística via ExecutorAvaliacao
            config_estatistica = config.get("estatistica", {})
            avaliacao = self._executor_avaliacao.executar_avaliacao(
                motor_treino=motor_treino,
                df_dev=df_dev,
                alvo=alvo,
                config_estatistica=config_estatistica if isinstance(config_estatistica, dict) else None,
            )
            resultado_estatistico_dict = avaliacao.resumo_estatistico

            # 10. Ensemble ou Tuning de Hiperparâmetros via OtimizadorCampeao
            config_tuning = config.get("tuning", {})
            config_votacao = config.get("votacao", {})
            splits_folds = int(config.get("validacao", {}).get("cv_interna", {}).get("splits", 3))

            otimizacao = self._otimizador_campeao.otimizar_ou_combinar(
                motor_treino=motor_treino,
                df_dev=df_dev,
                dados_staging=dados_staging,
                alvo=alvo,
                modelo_campeao_nome=avaliacao.modelo_campeao,
                config_tuning=config_tuning if isinstance(config_tuning, dict) else {},
                config_votacao=config_votacao if isinstance(config_votacao, dict) else {},
                resultado_estatistico=avaliacao.resultado_estatistico,
                perdas_pareadas=avaliacao.perdas_pareadas,
                cv_folds=splits_folds,
            )
            modelo_final = otimizacao.modelo_final
            modelo_campeao_nome = otimizacao.modelo_campeao_nome
            resultado_tuning = otimizacao.resultado_tuning
            if "modelo_campeao" in resultado_estatistico_dict:
                resultado_estatistico_dict["modelo_campeao"] = modelo_campeao_nome

        # 11. Montagem do ModeloServico via MontadorServico
        config_regras = config.get("regras_negocio", {})
        servico_mlflow = self._montador_servico.montar_servico(
            modelo_final=modelo_final,
            config_regras=config_regras if isinstance(config_regras, dict) else None,
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
                    parametros=parametros_log,
                    artefatos_adicionais=[caminho_eda_html, caminho_eda_json],
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
            "eda": {
                "ativo": ativo_eda,
                "motor": motor_eda,
                "relatorio_html": caminho_eda_html,
                "relatorio_json": caminho_eda_json,
            },
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
        choices=["completo", "apenas_campeao", "apenas_eda"],
        default=None,
        help="Modo de execução: 'completo' (com CV e tuning), 'apenas_campeao' (treino direto sem CV) ou 'apenas_eda' (apenas análise exploratória)"
    )
    parser.add_argument(
        "--apenas-campeao",
        action="store_true",
        help="Atalho para executar em modo 'apenas_campeao' (pula validação cruzada pareada e tuning de parâmetros)"
    )
    parser.add_argument(
        "--apenas-eda",
        "--eda",
        dest="apenas_eda",
        action="store_true",
        help="Atalho para executar apenas a análise exploratória (EDA) e gerar relatórios visuais e estatísticos"
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

    modo_selecionado = (
        "apenas_eda"
        if args.apenas_eda
        else ("apenas_campeao" if args.apenas_campeao else args.modo)
    )

    print("=" * 70)
    if modo_selecionado == "apenas_eda":
        print(" INICIANDO ANÁLISE EXPLORATÓRIA (EDA) — IMÓVEIS RIBEIRÃO PRETO")
    else:
        print(" INICIANDO PIPELINE DE TREINAMENTO — IMÓVEIS RIBEIRÃO PRETO")
    print("=" * 70)
    print(f"Arquivo de configuração : {args.config}")
    print(f"Modo de execução        : {modo_selecionado or 'definido pelo YAML'}")
    if args.modelo and modo_selecionado != "apenas_eda":
        print(f"Modelo forçado          : {args.modelo}")
    print("=" * 70 + "\n")

    try:
        pipeline = MainPipeline(caminho_configuracao=args.config)
        resultado = pipeline.executar_pipeline(
            modo=modo_selecionado,
            modelo_campeao_forcado=args.modelo
        )
        print("\n" + "=" * 70)
        if modo_selecionado == "apenas_eda":
            print(" ANÁLISE EXPLORATÓRIA (EDA) CONCLUÍDA COM SUCESSO!")
        else:
            print(" TREINAMENTO E REGISTRO CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as erro:
        print(f"\n[ERRO CRÍTICO NO PIPELINE]: {erro}", file=sys.stderr)
        sys.exit(1)

