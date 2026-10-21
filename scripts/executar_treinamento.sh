#!/usr/bin/env bash
# ==============================================================================
# Script de Execução do Pipeline de Treinamento e Avaliação Imobiliária
# Projeto: Machine Learning Apartamentos Ribeirão Preto
# ==============================================================================
# Uso:
#   ./executar_treinamento.sh
#   ./executar_treinamento.sh --config caminho/para/custom_config.yaml
# ==============================================================================

set -euo pipefail

# Cores ANSI para saída no terminal
CLR_RESET="\033[0m"
CLR_INFO="\033[1;34m"
CLR_SUCCESS="\033[1;32m"
CLR_WARN="\033[1;33m"
CLR_ERR="\033[1;31m"
CLR_BOLD="\033[1m"

# Diretório raiz do projeto (onde o script reside)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo -e "${CLR_INFO}====================================================================${CLR_RESET}"
echo -e "${CLR_BOLD} 🚀 INICIALIZADOR DE TREINAMENTO — IMÓVEIS RIBEIRÃO PRETO${CLR_RESET}"
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

echo -e "${CLR_INFO}[INFO]${CLR_RESET} Utilizando Python: ${CLR_BOLD}${PYTHON_BIN}${CLR_RESET}"
${PYTHON_BIN} --version

# 3. Define arquivo de configuração padrão ou argumento fornecido
CONFIG_FILE="configuracoes/projeto.yaml"
if [ "$#" -gt 0 ]; then
    if [ "$1" = "--config" ] || [ "$1" = "-c" ]; then
        if [ -n "${2:-}" ]; then
            CONFIG_FILE="$2"
        else
            echo -e "${CLR_ERR}[ERRO] Flag --config fornecida sem o caminho do arquivo.${CLR_RESET}" >&2
            exit 1
        fi
    else
        CONFIG_FILE="$1"
    fi
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${CLR_ERR}[ERRO] Arquivo de configuração não encontrado em: '${CONFIG_FILE}'${CLR_RESET}" >&2
    exit 1
fi

echo -e "${CLR_INFO}[INFO]${CLR_RESET} Arquivo de configuração selecionado: ${CLR_BOLD}${CONFIG_FILE}${CLR_RESET}"
echo -e "${CLR_INFO}[INFO]${CLR_RESET} Iniciando orquestração do pipeline (Ingestão -> Staging -> EDA -> CV Pareada -> Friedman/Nemenyi -> Tuning/Ensemble -> MLflow Serving)..."
echo ""

# 4. Executa o pipeline de produto via módulo Python oficial
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
${PYTHON_BIN} -m app.orquestracao_ml.main_pipeline --config "${CONFIG_FILE}"

STATUS_EXEC=$?

if [ ${STATUS_EXEC} -eq 0 ]; then
    echo ""
    echo -e "${CLR_SUCCESS}====================================================================${CLR_RESET}"
    echo -e "${CLR_SUCCESS} ✅ Pipeline finalizado com sucesso!${CLR_RESET}"
    echo -e "${CLR_SUCCESS}====================================================================${CLR_RESET}"
else
    echo ""
    echo -e "${CLR_ERR}====================================================================${CLR_RESET}"
    echo -e "${CLR_ERR} ❌ Ocorreu uma falha na execução do treinamento (código ${STATUS_EXEC}).${CLR_RESET}"
    echo -e "${CLR_ERR}====================================================================${CLR_RESET}"
fi

exit ${STATUS_EXEC}
