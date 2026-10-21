#!/usr/bin/env bash
# ==============================================================================
# Script de Execução do Pipeline de Treinamento, Publicação e Serving no MLflow
# Projeto: Machine Learning Apartamentos Ribeirão Preto
# ==============================================================================
# Uso básico:
#   ./executar_treinamento.sh
#
# Exemplos com flags:
#   # Treinar direto o modelo campeão já salvo no MLflow e subir a API (sem CV e sem Tuning):
#   ./executar_treinamento.sh --apenas-campeao --subir-api
#
#   # Apenas treinar e registrar o campeão no MLflow rapidamente:
#   ./executar_treinamento.sh --apenas-campeao
#
#   # Forçar um modelo específico para treinar diretamente e servir:
#   ./executar_treinamento.sh --apenas-campeao --modelo lightgbm --subir-api
#
#   # Executar pipeline completo (CV pareada + Friedman + Tuning + Registro):
#   ./executar_treinamento.sh --modo completo
# ==============================================================================

set -euo pipefail

# ==============================================================================
# ⚙️ CONFIGURAÇÕES PRINCIPAIS (Altere diretamente aqui ou utilize flags de CLI)
# ==============================================================================
# MODO_EXECUCAO:
#   "completo"        -> Ingestão + Staging + EDA + CV Pareada + Friedman + Tuning + Registro MLflow
#   "apenas_campeao"  -> Pula CV e Tuning; treina direto o modelo campeão salvo no MLflow e empacota no ModeloServico
MODO_EXECUCAO="completo"

# SUBIR_API_SERVING:
#   true  -> Inicia a API do MLflow Serving ao término do treinamento
#   false -> Apenas executa o treinamento e publicação no MLflow
SUBIR_API_SERVING=false

# Porta e Host de Binding para a API do MLflow Serving
PORTA_SERVING=8080
HOST_SERVING="0.0.0.0"

# MODELO_CAMPEAO_ESPECIFICO (opcional):
#   Se preenchido (ex: "random_forest", "lightgbm"), força o treinamento deste modelo.
#   Se vazio (""), consulta automaticamente o modelo campeão salvo no MLflow Registry ou YAML.
MODELO_CAMPEAO_ESPECIFICO=""

# Caminho padrão do arquivo de configuração do projeto
CONFIG_FILE="configuracoes/projeto.yaml"

# ==============================================================================
# Cores ANSI para saída formatada no terminal
# ==============================================================================
CLR_RESET="\033[0m"
CLR_INFO="\033[1;34m"
CLR_SUCCESS="\033[1;32m"
CLR_WARN="\033[1;33m"
CLR_ERR="\033[1;31m"
CLR_BOLD="\033[1m"
CLR_CYAN="\033[1;36m"

# Diretório raiz do projeto
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

exibir_ajuda() {
    echo -e "${CLR_BOLD}Uso:${CLR_RESET} $0 [OPÇÕES]"
    echo ""
    echo -e "${CLR_BOLD}Opções:${CLR_RESET}"
    echo "  --apenas-campeao, --somente-campeao   Executa em modo rápido: treina direto o campeão salvo no MLflow (pula CV e Tuning)"
    echo "  --subir-api                           Inicializa a API do MLflow Serving após o treinamento"
    echo "  --modo <completo|apenas_campeao>      Define explicitamente o modo de execução"
    echo "  --modelo <nome>                       Força o modelo campeão a treinar (ex: random_forest, lightgbm, xgboost)"
    echo "  --porta <porta>                       Define a porta da API do MLflow Serving (padrão: 8080)"
    echo "  --host <host>                         Define o host de binding da API (padrão: 0.0.0.0)"
    echo "  --config, -c <caminho>                Define caminho do arquivo YAML de configuração"
    echo "  --ajuda, -h                           Exibe esta mensagem de ajuda"
    echo ""
    echo -e "${CLR_BOLD}Exemplos:${CLR_RESET}"
    echo "  $0 --apenas-campeao --subir-api"
    echo "  $0 --apenas-campeao --modelo random_forest"
    echo "  $0 --config configuracoes/projeto.yaml --subir-api"
    exit 0
}

# ==============================================================================
# Leitura de Argumentos de Linha de Comando (Sobrescrevem variáveis do topo)
# ==============================================================================
while [ "$#" -gt 0 ]; do
    case "$1" in
        --apenas-campeao|--somente-campeao|--pular-cv-tuning)
            MODO_EXECUCAO="apenas_campeao"
            shift
            ;;
        --subir-api|--servir)
            SUBIR_API_SERVING=true
            shift
            ;;
        --modo|-m)
            MODO_EXECUCAO="$2"
            shift 2
            ;;
        --modelo)
            MODELO_CAMPEAO_ESPECIFICO="$2"
            shift 2
            ;;
        --porta|-p)
            PORTA_SERVING="$2"
            shift 2
            ;;
        --host)
            HOST_SERVING="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --ajuda|-h|--help)
            exibir_ajuda
            ;;
        *)
            echo -e "${CLR_ERR}[ERRO] Opção desconhecida: $1${CLR_RESET}" >&2
            echo "Use $0 --ajuda para ver as opções disponíveis."
            exit 1
            ;;
    esac
done

echo -e "${CLR_INFO}====================================================================${CLR_RESET}"
echo -e "${CLR_BOLD} 🚀 PIPELINE DE MACHINE LEARNING & MLFLOW SERVING — IMÓVEIS RP${CLR_RESET}"
echo -e "${CLR_INFO}====================================================================${CLR_RESET}"

# 1. Carrega variáveis de ambiente de .env se existir
if [ -f ".env" ]; then
    echo -e "${CLR_INFO}[INFO]${CLR_RESET} Carregando variáveis de ambiente do arquivo .env..."
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# 2. Localiza interpretador Python no ambiente virtual
PYTHON_BIN=""
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo -e "${CLR_ERR}[ERRO] Interpretador Python não encontrado. Certifique-se de que o ambiente virtual (.venv) está configurado.${CLR_RESET}" >&2
    exit 1
fi

echo -e "${CLR_INFO}[INFO]${CLR_RESET} Interpretador Python : ${CLR_BOLD}${PYTHON_BIN}${CLR_RESET}"
echo -e "${CLR_INFO}[INFO]${CLR_RESET} Configuração YAML    : ${CLR_BOLD}${CONFIG_FILE}${CLR_RESET}"
echo -e "${CLR_INFO}[INFO]${CLR_RESET} Modo de Execução     : ${CLR_CYAN}${MODO_EXECUCAO}${CLR_RESET}"
if [ -n "${MODELO_CAMPEAO_ESPECIFICO}" ]; then
    echo -e "${CLR_INFO}[INFO]${CLR_RESET} Modelo Forçado       : ${CLR_CYAN}${MODELO_CAMPEAO_ESPECIFICO}${CLR_RESET}"
fi
echo -e "${CLR_INFO}[INFO]${CLR_RESET} Subir API Serving    : ${CLR_BOLD}${SUBIR_API_SERVING}${CLR_RESET} (Porta: ${PORTA_SERVING})"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${CLR_ERR}[ERRO] Arquivo de configuração não encontrado em: '${CONFIG_FILE}'${CLR_RESET}" >&2
    exit 1
fi

# 3. Monta parâmetros para chamada do módulo Python
ARGS_PYTHON=( "--config" "${CONFIG_FILE}" )
if [ "${MODO_EXECUCAO}" = "apenas_campeao" ]; then
    ARGS_PYTHON+=( "--apenas-campeao" )
fi
if [ -n "${MODELO_CAMPEAO_ESPECIFICO}" ]; then
    ARGS_PYTHON+=( "--modelo" "${MODELO_CAMPEAO_ESPECIFICO}" )
fi

echo ""
if [ "${MODO_EXECUCAO}" = "apenas_campeao" ]; then
    echo -e "${CLR_WARN}⚡ MODO RÁPIDO SELECIONADO:${CLR_RESET} Pulando Validação Cruzada Pareada e Tuning de Parâmetros."
    echo -e "${CLR_INFO}⚡ Treinando diretamente o modelo campeão salvo no MLflow com dados completos de staging e regras dinâmicas.${CLR_RESET}"
else
    echo -e "${CLR_INFO}[INFO]${CLR_RESET} Executando pipeline completo (Ingestão -> Staging -> EDA -> CV Pareada -> Friedman/Nemenyi -> Tuning -> Registro MLflow)..."
fi
echo ""

# 4. Executa o pipeline de produto via módulo Python oficial
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
${PYTHON_BIN} -m app.orquestracao_ml.main_pipeline "${ARGS_PYTHON[@]}"
STATUS_EXEC=$?

if [ ${STATUS_EXEC} -ne 0 ]; then
    echo ""
    echo -e "${CLR_ERR}====================================================================${CLR_RESET}"
    echo -e "${CLR_ERR} ❌ Ocorreu uma falha na execução do pipeline (código ${STATUS_EXEC}).${CLR_RESET}"
    echo -e "${CLR_ERR}====================================================================${CLR_RESET}"
    exit ${STATUS_EXEC}
fi

echo ""
echo -e "${CLR_SUCCESS}====================================================================${CLR_RESET}"
echo -e "${CLR_SUCCESS} ✅ Treinamento e Registro no MLflow concluídos com sucesso!${CLR_RESET}"
echo -e "${CLR_SUCCESS}====================================================================${CLR_RESET}"

# ==============================================================================
# 5. Inicialização da API do MLflow Serving (caso SUBIR_API_SERVING=true)
# ==============================================================================
if [ "${SUBIR_API_SERVING}" = true ]; then
    echo ""
    echo -e "${CLR_INFO}====================================================================${CLR_RESET}"
    echo -e "${CLR_BOLD} 🌐 INICIALIZANDO API DO MLFLOW SERVING (Porta ${PORTA_SERVING})${CLR_RESET}"
    echo -e "${CLR_INFO}====================================================================${CLR_RESET}"

    # Verifica se o container docker mlflow-serving já está ativo
    CONTAINER_SERVING_ID=$(docker ps -q -f name=mlflow-serving 2>/dev/null || true)
    if [ -n "${CONTAINER_SERVING_ID}" ]; then
        echo -e "${CLR_INFO}[INFO]${CLR_RESET} Container Docker 'mlflow-serving' detectado em execução na porta ${PORTA_SERVING}."
        echo -e "${CLR_INFO}[INFO]${CLR_RESET} Reiniciando container para carregar a nova versão registrada do modelo..."
        docker restart mlflow-serving >/dev/null 2>&1 || true
        echo -e "${CLR_SUCCESS} ✅ Container 'mlflow-serving' recarregado com o novo modelo campeão!${CLR_RESET}"
        echo -e "${CLR_CYAN} 📍 Endpoint de Inferência : http://localhost:${PORTA_SERVING}/invocations${CLR_RESET}"
        echo -e "${CLR_CYAN} 📍 Endpoint de Healthcheck: http://localhost:${PORTA_SERVING}/health${CLR_RESET}"
        echo -e "${CLR_INFO}====================================================================${CLR_RESET}"
    else
        echo -e "${CLR_INFO}[INFO]${CLR_RESET} Inicializando servidor nativo MLflow Models Serve..."
        echo -e "${CLR_CYAN} 📍 Endpoint de Inferência : http://localhost:${PORTA_SERVING}/invocations${CLR_RESET}"
        echo -e "${CLR_CYAN} 📍 Pressione Ctrl+C para encerrar o servidor.${CLR_RESET}"
        echo -e "${CLR_INFO}====================================================================${CLR_RESET}"
        ${PYTHON_BIN} -m mlflow models serve \
            -m "models:/previsao_preco_imoveis/latest" \
            -p "${PORTA_SERVING}" \
            -h "${HOST_SERVING}" \
            --no-conda
    fi
fi

exit 0
