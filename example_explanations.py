"""Exemplo: Consumindo as explicações geradas com SHAP."""

from pathlib import Path
from PIL import Image
import pandas as pd


def load_explanation_images() -> dict[str, Image.Image]:
    """Carrega todas as imagens de explicabilidade.

    Returns:
        Dicionário com nome do arquivo como chave e Image como valor.
    """
    explanations_path = Path("outputs") / "explanations"

    images = {}
    for img_file in explanations_path.glob("*.png"):
        images[img_file.stem] = Image.open(img_file)

    print(f"✅ Carregadas {len(images)} imagens de explicabilidade")
    return images


def load_autoencoder_analysis() -> pd.DataFrame:
    """Carrega análise de erros do Autoencoder.

    Returns:
        DataFrame com análise de features.
    """
    csv_path = Path("outputs") / "explanations" / "autoencoder_reconstruction_errors.csv"
    df = pd.read_csv(csv_path)
    print(f"✅ Carregada análise com {len(df)} features")
    return df


def main() -> None:
    """Exemplo de uso das explicações."""
    explanations_path = Path("outputs") / "explanations"

    if not explanations_path.exists():
        print("❌ Pasta de explicações não encontrada.")
        print("Execute: python -m src.explain")
        return

    # Carregar imagens
    images = load_explanation_images()

    print("\n" + "=" * 60)
    print("EXPLICAÇÕES DISPONÍVEIS")
    print("=" * 60)
    print("\n🔹 RANDOM FOREST:")
    for key in images:
        if "random_forest" in key:
            print(f"  - {key}")

    print("\n🔹 XGBOOST:")
    for key in images:
        if "xgboost" in key:
            print(f"  - {key}")

    print("\n🔹 WATERFALL (Exemplos por transação):")
    for key in images:
        if "waterfall" in key:
            print(f"  - {key}")

    print("\n🔹 AUTOENCODER:")
    for key in images:
        if "autoencoder" in key:
            print(f"  - {key}")

    # Carregar análise do Autoencoder
    ae_analysis = load_autoencoder_analysis()
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES - MAIOR ERRO DE RECONSTRUÇÃO")
    print("=" * 60)
    print(ae_analysis.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("📌 Como usar no dashboard:")
    print("=" * 60)
    print("""
from PIL import Image
from pathlib import Path

# Carregar uma imagem
img = Image.open("outputs/explanations/shap_summary_random_forest.png")

# Exibir no Streamlit
import streamlit as st
st.image(img, caption="SHAP Summary - Random Forest")

# Ou carregar todas
explanations_path = Path("outputs/explanations")
for img_file in sorted(explanations_path.glob("*.png")):
    st.image(str(img_file))
    """)


if __name__ == "__main__":
    main()
