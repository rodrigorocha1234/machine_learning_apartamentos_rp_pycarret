# Arquitetura e responsabilidades
**Fluxo:** fonte XLSX → AdaptadorFonte → MotorTabular (pandas/polars) → RegrasAST → Quality Gate → RepositorioStaging (SQLite) → EDA → separação treino/holdout → MotorTreino (PyCaret **ou** sklearn) → tuning + CV sem vazamento → Friedman/Nemenyi → Ensemble opcional → finalização → MLflow Registry/Serving + regras YAML → Grafana.
**Ports Protocol:** FonteDados.carregar_dados; MotorTabela.filtrar_dados/transformar_dados; RepositorioDados.gravar_dados/ler_dados; MotorAprendizado.ajustar_parametros/avaliar_modelos/treinar_final; RegraNegocio.aplicar_regra.
**Composição:** MainPipeline agrega fonte, motor tabular, repositório, avaliação, seleção e registro. Injeção de dependência para testes. Não acoplar PyCaret e sklearn ao mesmo módulo de implementação.
**Persistência:** SQLite é staging local autorizado no escopo explicitamente solicitado; modelos, parâmetros e experimentos ficam no MLflow; integrar aos serviços do Compose existente, sem criar stack paralelo.
**Nota:** `finalize_model` do PyCaret em API pública é usado apenas no motor PyCaret, com cuidado para excluir holdout retido de avaliação até o gate adequado.
