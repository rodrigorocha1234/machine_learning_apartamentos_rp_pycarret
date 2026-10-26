#!/bin/bash

# Confirmar com o usuário
echo "Este script irá remover todos os containers, imagens, volumes e redes do Docker!"
read -p "Tem certeza que deseja continuar? (s/n): " resposta

if [ "$resposta" = "s" ] || [ "$resposta" = "S" ]; then
    echo "Iniciando a limpeza do Docker..."

    docker stop $(docker ps -q)

    docker rm $(docker ps -aq)

    docker rmi $(docker images -q) -f

    docker volume rm $(docker volume ls -q)

    docker network prune -f

    docker system prune -a --volumes -f

    echo "Limpeza do Docker concluída!"

else
    echo "Operação cancelada."
fi