#!/usr/bin/env python3
"""Script para executar tuning e validação cruzada com PyCaret nativo.

Baseado na proposta do usuário para previsão de preços de imóveis.
"""
from pathlib import Path
from app.modelagem_ml.experimento_pycaret import ExperimentoPyCaret

def main() -> None:
    caminho_dados: Path = Path("dados/bairro_final_v3_engineered.xlsx")
    if not caminho_dados.exists():
        caminho_dados = Path("docs/bairro_final_v3_engineered.xlsx")

    print(f"-> Carregando dados a partir de: {caminho_dados}")
    experimento = ExperimentoPyCaret(semente=42)

    df_limpo = experimento.preparar_dados(
        caminho_dados=caminho_dados,
        colunas_remover=["Código", "Apartamento", "valor_m2"],
        alvo="Valor_da_Venda",
    )
    print(f"-> Dimensão dos dados após remoção de colunas de vazamento: {df_limpo.shape}")

    # Renomear para 'preco' para refletir exatamente a nomenclatura solicitada pelo usuário
    if "Valor_da_Venda" in df_limpo.columns:
        df_limpo = df_limpo.rename(columns={"Valor_da_Venda": "preco"})

    # Separar holdout de 20% previamente (garantia de ausência de vazamento de dados)
    df_treino = df_limpo.sample(frac=0.8, random_state=42)
    df_holdout = df_limpo.drop(df_treino.index)
    print(f"-> Partição de treino (df_treino): {df_treino.shape} | Holdout intocado: {df_holdout.shape}")

    print("-> Executando setup no PyCaret com parâmetros lidos do YAML (imputação, folds, pré-processamento)...")
    experimento.iniciar_setup(
        df_treino=df_treino,
        alvo="preco",
    )

    print("-> Comparando modelos com validação cruzada (ordenado por MAE, top 3)...")
    melhores = experimento.comparar_modelos(ordenar_por="MAE", n_selecionar=3)

    print("\n=== Comparação dos modelos ===")
    tabela_comp = experimento.obter_tabela_comparacao()
    print(tabela_comp)

    print("\n-> Ajustando hiperparâmetros dos modelos com registro por modelo no MLflow (tune_model)...")
    nomes_modelos = ["Random_Forest", "LightGBM", "Extra_Trees"]
    modelo_campeao_tunado = None
    for i, mod in enumerate(melhores[:2]):
        nome_mod = nomes_modelos[i] if i < len(nomes_modelos) else f"Modelo_{i+1}"
        print(f"\n   -> [Tuning MLflow] Otimizando e registrando no MLflow: {nome_mod}...")
        mod_tunado = experimento.ajustar_hiperparametros(
            mod,
            nome_modelo=nome_mod,
            metrica_otimizar="MAE",
            n_iteracoes=15,
            n_folds=5,
            salvar_mlflow=True,
        )
        if modelo_campeao_tunado is None:
            modelo_campeao_tunado = mod_tunado
        print(f"   -> [Tuning MLflow] Resultados de {nome_mod} persistidos com sucesso no MLflow!")

    print("\n=== Resultados do tuning (último modelo) ===")
    tabela_tuning = experimento.obter_tabela_tuning()
    print(tabela_tuning)

    print("\n-> Executando validação cruzada posterior sobre dados de treino...")
    resumo_cv = experimento.executar_validacao_posterior(modelo_campeao_tunado, n_splits=5)

    print("\n=== Validação cruzada posterior ===")
    for metrica, valores in resumo_cv.items():
        media = valores["media"]
        desvio = valores["desvio"]
        print(f"{metrica}: média={media:.4f}, desvio={desvio:.4f}")

if __name__ == "__main__":
    main()
