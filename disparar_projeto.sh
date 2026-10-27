#!/usr/bin/env bash
# ==============================================================================
# Script de Disparo do Projeto de Machine Learning — Apartamentos RP
# ==============================================================================
# Orquestra a infraestrutura Docker, execução do pipeline de ML, API FastAPI,
# testes automatizados e acesso à documentação.
# ==============================================================================

set -e

DIR_RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR_RAIZ"

# Cores ANSI para saída no terminal
C_AZUL='\033[1;34m'
C_CIANO='\033[1;36m'
C_VERDE='\033[1;32m'
C_AMARELO='\033[1;33m'
C_VERMELHO='\033[1;31m'
C_RESET='\033[0m'

imprimir_banner() {
    echo -e "${C_CIANO}"
    echo "================================================================================"
    echo "       SISTEMA DE PREVISÃO DE PREÇOS DE APARTAMENTOS — RIBEIRÃO PRETO/SP        "
    echo "          Pipeline de ML • PyCaret & Sklearn • MLflow • Grafana • FastAPI       "
    echo "================================================================================"
    echo -e "${C_RESET}"
}

verificar_ambiente() {
    if [ ! -d ".venv" ]; then
        echo -e "${C_VERMELHO}ERRO: Ambiente virtual .venv não encontrado em $DIR_RAIZ.${C_RESET}"
        echo "Por favor, crie o ambiente virtual com: python3.12 -m venv .venv && pip install -r ..."
        exit 1
    fi
    source .venv/bin/activate
    export PYTHONPATH="$DIR_RAIZ"
}

verificar_docker() {
    echo -e "${C_AZUL}Verificando infraestrutura Docker (PostgreSQL, MLflow, Prometheus, Grafana)...${C_RESET}"
    if ! docker info >/dev/null 2>&1; then
        echo -e "${C_AMARELO}AVISO: Docker daemon não está acessível ou sem permissão. Pulando verificação automática de containers.${C_RESET}"
        return
    fi

    # Verifica se os containers essenciais estão rodando
    if ! docker ps --format '{{.Names}}' | grep -q "mlflow"; then
        echo -e "${C_AMARELO}Subindo containers do Docker Compose em segundo plano...${C_RESET}"
        docker compose up -d
        echo -e "${C_VERDE}Containers inicializados. Aguardando prontidão dos serviços...${C_RESET}"
        sleep 3
    else
        echo -e "${C_VERDE}✓ Containers Docker já estão operacionais (MLflow na porta 5000, Grafana na porta 3000).${C_RESET}"
    fi
}

executar_pipeline() {
    verificar_ambiente
    verificar_docker
    echo -e "\n${C_CIANO}Iniciando execução do pipeline de Machine Learning (todas as etapas)...${C_RESET}"
    python app/executar_pipeline.py --etapa todas
}

executar_etapa() {
    verificar_ambiente
    verificar_docker
    ETAPA="$1"
    if [ -z "$ETAPA" ]; then
        echo -e "${C_VERMELHO}Informe o número da etapa (1 a 7). Exemplo: ./disparar_projeto.sh etapa 4${C_RESET}"
        exit 1
    fi
    echo -e "\n${C_CIANO}Iniciando Etapa $ETAPA...${C_RESET}"
    python app/executar_pipeline.py --etapa "$ETAPA"
}

treinar_campeao() {
    verificar_ambiente
    verificar_docker
    echo -e "\n${C_CIANO}====================================================================${C_RESET}"
    echo -e "${C_CIANO}  Treinamento & Promoção Exclusiva do Modelo Campeão (LightGBM)    ${C_RESET}"
    echo -e "${C_CIANO}====================================================================${C_RESET}"
    echo -e "${C_AZUL}Iniciando Etapa 6: Ajuste com Pré-processamento, Holdout e Regras...${C_RESET}\n"
    python app/executar_pipeline.py --etapa 6
    echo -e "\n${C_VERDE}✓ Modelo campeão treinado, avaliado no holdout e promovido com sucesso!${C_RESET}"
    echo -e "${C_VERDE}  Artefatos persistidos em: production_artifacts/campeao/${C_RESET}"
}

iniciar_api() {
    verificar_ambiente
    echo -e "\n${C_VERDE}====================================================================${C_RESET}"
    echo -e "${C_VERDE}  Iniciando API FastAPI com Modelo Campeão Promovido (Porta 8000)   ${C_RESET}"
    echo -e "${C_VERDE}  Documentação Swagger disponível em: http://localhost:8000/docs   ${C_RESET}"
    echo -e "${C_VERDE}====================================================================${C_RESET}\n"
    exec uvicorn app.servicos_implantacao.aplicacao_fastapi:app --host 0.0.0.0 --port 8000 --reload
}

executar_testes() {
    verificar_ambiente
    echo -e "\n${C_AZUL}Executando suíte completa de testes automatizados com Pytest...${C_RESET}"
    pytest tests/ -v
    echo -e "\n${C_AZUL}Executando verificação de tipagem estrita com Pyright...${C_RESET}"
    pyright app/ tests/
    echo -e "\n${C_VERDE}✓ Todos os testes e validações de tipagem passaram com sucesso!${C_RESET}"
}

mostrar_docs() {
    echo -e "\n${C_CIANO}====================================================================${C_RESET}"
    echo -e "${C_CIANO}  Documentação Técnica Interativa do Projeto                       ${C_RESET}"
    echo -e "${C_CIANO}====================================================================${C_RESET}"
    echo -e "Arquivo HTML na raiz: ${C_VERDE}file://$DIR_RAIZ/documentacao_projeto.html${C_RESET}"
    echo -e "Arquivo HTML em docs: ${C_VERDE}file://$DIR_RAIZ/docs/index.html${C_RESET}"
    echo ""
    echo "Diagramas incluídos na documentação:"
    echo "  1. Diagrama UML de Classes (GoF Factory, Strategy, Repository, AstValidator)"
    echo "  2. Diagrama de Atividades (Fluxo operacional de ponta a ponta com decisões)"
    echo "  3. Diagrama de Fluxo & Sequência (Linha do tempo e comunicação entre sistemas)"
    echo ""
    if command -v xdg-open >/dev/null 2>&1; then
        echo -e "${C_AZUL}Tentando abrir no navegador padrão...${C_RESET}"
        xdg-open "$DIR_RAIZ/documentacao_projeto.html" >/dev/null 2>&1 || true
    fi
}

mostrar_ajuda() {
    echo -e "Uso: ${C_VERDE}./disparar_projeto.sh [COMANDO]${C_RESET}\n"
    echo "Comandos disponíveis:"
    echo -e "  ${C_CIANO}pipeline${C_RESET}     (padrão) Executa o pipeline de ML completo (Etapas 1 a 7)"
    echo -e "  ${C_CIANO}etapa <N>${C_RESET}    Executa uma etapa específica de 1 a 7 (ex: etapa 4)"
    echo -e "  ${C_CIANO}api${C_RESET}          Inicia o servidor de inferência FastAPI (porta 8000)"
    echo -e "  ${C_CIANO}testes${C_RESET}       Executa a suíte de 134 testes no Pytest e Pyright estrito"
    echo -e "  ${C_CIANO}docker${C_RESET}       Sobe ou verifica os containers (MLflow, Postgres, Prometheus, Grafana)"
    echo -e "  ${C_CIANO}docs${C_RESET}         Exibe o link e abre a documentação interativa em HTML"
    echo -e "  ${C_CIANO}ajuda${C_RESET}        Exibe este menu de ajuda"
    echo ""
}

# ------------------------------------------------------------------------------
# Ponto de entrada
# ------------------------------------------------------------------------------
imprimir_banner

case "${1:-pipeline}" in
    pipeline)
        executar_pipeline
        ;;
    etapa)
        executar_etapa "$2"
        ;;
    api)
        iniciar_api
        ;;
    testes)
        executar_testes
        ;;
    docker)
        verificar_docker
        ;;
    docs)
        mostrar_docs
        ;;
    ajuda|-h|--help)
        mostrar_ajuda
        ;;
    *)
        echo -e "${C_VERMELHO}Comando desconhecido: '$1'${C_RESET}\n"
        mostrar_ajuda
        exit 1
        ;;
esac
