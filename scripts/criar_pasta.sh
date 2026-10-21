#!/bin/sh

set -e

echo "Criando diretórios de dados e persistência do projeto..."

# Diretórios de dados para os containers Docker
for pasta in \
  "./pgdata" \
  "./storage-data" \
  "./grafana_data" \
  "./prometheus_data" \
  "./observabilidade_data" \
  "./app_build" \
  "./configuracoes" \
  "./src"
do
  mkdir -p "$pasta"
  chmod -R 777 "$pasta" 2>/dev/null || true
done

echo "Diretórios criados com sucesso!"