"""Script de execução para gerar explicações com SHAP."""

from pathlib import Path
from src.explain import generate_all_explanations
from src.ensemble import load_and_preprocess


def main() -> None:
    """Executa geração de explicações."""
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "creditcard.csv"
    models_path = project_root / "models"
    output_path = project_root / "outputs" / "explanations"

    # Carregar dados de teste
    _, X_test, _, _ = load_and_preprocess(data_path)

    # Gerar todas as explicações
    generate_all_explanations(X_test, models_path, output_path)


if __name__ == "__main__":
    main()
