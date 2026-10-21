# Integração de Infraestrutura e Serving Nativo MLflow
**Papel Responsável:** `@devops`  
**Data:** 2026-10-20  
**Status:** VALIDADO E CONFIGURADO  

---

## 1. Stack Existente no Docker Compose

O projeto reutiliza integralmente a infraestrutura existente definida em `docker-compose.yaml`:
- **PostgreSQL 15 (`mlflow-postgres`):** Backend Store na porta 5432.
- **RustFS / S3 (`storage`):** Armazenamento de artefatos na porta 9000/9001.
- **MLflow Server (`mlflow-server`):** Tracking e Registry na porta 5000.
- **Prometheus (`prometheus`):** Coleta de métricas na porta 9090.
- **Grafana (`grafana`):** Painéis e dashboards na porta 3000.

---

## 2. Padrão de Deployment de Inferência

- O modelo final é empacotado como artefato `mlflow.pyfunc` versionado.
- Endpoint de inferência oficial: `http://localhost:5000/invocations` (via MLflow Models Serve nativo).
- Proibição de rotas FastAPI paralelas conforme a regra `arquitetura.md`.
- Política estrita: nenhum deployment ou alteração de ambiente produtivo é realizado sem autorização prévia explícita do usuário.
