"""Explicabilidade de modelos com SHAP e análise de reconstrução."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import rcParams

from . import logger

# Configuração visual Nord
NORD_PALETTE = [
    "#88C0D0",
    "#A3BE8C",
    "#EBCB8B",
    "#BF616A",
    "#B48EAD",
    "#D08770",
    "#81A1C1",
]

rcParams["figure.facecolor"] = "#2E3440"
rcParams["axes.facecolor"] = "#3B4252"
rcParams["text.color"] = "#ECEFF4"
rcParams["axes.labelcolor"] = "#ECEFF4"
rcParams["xtick.color"] = "#D8DEE9"
rcParams["ytick.color"] = "#D8DEE9"


def generate_shap_values(
    X_test: pd.DataFrame,
    model_type: str,
    models_path: Path | str,
    sample_size: int | None = 1000,
) -> tuple[np.ndarray, shap.Explainer]:
    """Gera SHAP values para Random Forest ou XGBoost.

    Args:
        X_test: Features de teste.
        model_type: "random_forest" ou "xgboost".
        models_path: Caminho para os modelos salvos.
        sample_size: Tamanho da amostra para explicabilidade (None = usar tudo).

    Returns:
        Tuple com (shap_values, explainer).
    """
    models_path = Path(models_path)
    logger.info(f"Gerando SHAP values para {model_type}...")

    if model_type == "random_forest":
        model = joblib.load(models_path / "random_forest.joblib")
        explainer = shap.TreeExplainer(model)
    elif model_type == "xgboost":
        model = joblib.load(models_path / "xgboost.joblib")
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(
                f"TreeExplainer falhou para XGBoost ({e}), "
                "usando KernelExplainer com predict_proba..."
            )
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1],
                shap.sample(X_test, min(100, len(X_test)))
            )
    else:
        raise ValueError(f"model_type deve ser 'random_forest' ou 'xgboost'")

    # Se houver muitos dados, amostrar para acelerar
    if sample_size and len(X_test) > sample_size:
        logger.info(f"Amostrando {sample_size} de {len(X_test)} amostras...")
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test

    shap_values_raw = explainer.shap_values(X_sample)

    # Para modelos de classificação, pegar classe 1 (fraude)
    if isinstance(shap_values_raw, (list, tuple)):
        shap_values = np.asarray(shap_values_raw[1])
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # Shape (n_samples, n_features, n_classes) → classe 1
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = np.asarray(shap_values_raw)

    logger.info(f"SHAP values gerados: shape={shap_values.shape}")
    return shap_values, explainer


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    model_type: str,
    output_path: Path | str,
) -> None:
    """Cria SHAP summary plot (beeswarm).

    Args:
        shap_values: Array de SHAP values.
        X_test: Features de teste.
        model_type: Nome do modelo para título.
        output_path: Caminho onde salvar a figura.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Criando summary plot para {model_type}...")

    # Calcular importância média absoluta das features
    feature_importance = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(feature_importance)

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, feature_importance[indices], color=NORD_PALETTE[0], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(X_test.columns[indices], fontsize=9)
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path / f"shap_summary_{model_type}.png", dpi=300)
    plt.close()

    logger.info(f"Summary plot salvo: {output_path / f'shap_summary_{model_type}.png'}")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    model_type: str,
    output_path: Path | str,
) -> None:
    """Cria SHAP beeswarm plot.

    Args:
        shap_values: Array de SHAP values.
        X_test: Features de teste.
        model_type: Nome do modelo para título.
        output_path: Caminho onde salvar a figura.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Criando beeswarm plot para {model_type}...")

    plt.figure(figsize=(12, 10))
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(shap_values.shape[0]),
        data=X_test.values,
        feature_names=X_test.columns.tolist(),
    )
    shap.summary_plot(explanation, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig(
        output_path / f"shap_beeswarm_{model_type}.png", dpi=300
    )
    plt.close()

    logger.info(f"Beeswarm plot salvo: {output_path / f'shap_beeswarm_{model_type}.png'}")


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    idx: int,
    model_type: str,
    output_path: Path | str,
) -> None:
    """Cria visualização customizada tipo waterfall para uma transação.

    Args:
        shap_values: Array de SHAP values (shape: n_samples x n_features).
        X_test: Features de teste.
        idx: Índice da transação.
        model_type: Nome do modelo.
        output_path: Caminho onde salvar.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Criando gráfico de impacto para {model_type} (transação {idx})...")

    # Extrair SHAP values para a transação
    shap_idx = shap_values[idx]
    feature_names = X_test.columns.tolist()

    # Top 10 features por impacto absoluto
    importance = np.abs(shap_idx)
    top_indices = np.argsort(importance)[-10:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [NORD_PALETTE[3] if shap_idx[i] > 0 else NORD_PALETTE[0]
              for i in top_indices]

    ax.barh(range(len(top_indices)),
            shap_idx[top_indices],
            color=colors,
            alpha=0.7)

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.set_xlabel("SHAP Value (Impacto)")
    ax.set_title(f"Top 10 Features - {model_type.upper()} (Transação {idx})")
    ax.axvline(x=0, color=NORD_PALETTE[4], linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(
        output_path / f"shap_impact_{model_type}_idx{idx}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        f"Gráfico de impacto salvo: {output_path / f'shap_impact_{model_type}_idx{idx}.png'}"
    )


def analyze_autoencoder_reconstruction(
    X_test: pd.DataFrame, models_path: Path | str, output_path: Path | str
) -> pd.DataFrame:
    """Analisa erro de reconstrução do Autoencoder por feature.

    Args:
        X_test: Features de teste.
        models_path: Caminho dos modelos.
        output_path: Caminho para salvar análise.

    Returns:
        DataFrame com erro de reconstrução por feature.
    """
    models_path = Path(models_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Analisando erro de reconstrução do Autoencoder...")

    autoencoder = joblib.load(models_path / "autoencoder.joblib")

    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test
    ae_recon = autoencoder.predict(X_test_arr)

    # Erro de reconstrução por feature
    reconstruction_error = np.abs(X_test_arr - ae_recon)
    feature_error_mean = reconstruction_error.mean(axis=0)
    feature_error_std = reconstruction_error.std(axis=0)

    # Criar DataFrame
    ae_analysis = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_error": feature_error_mean,
            "std_error": feature_error_std,
        }
    ).sort_values("mean_error", ascending=False)

    logger.info(f"\nTop 10 features com maior erro:\n{ae_analysis.head(10)}")

    # Salvar análise
    ae_analysis.to_csv(
        output_path / "autoencoder_reconstruction_errors.csv", index=False
    )

    return ae_analysis


def plot_autoencoder_errors(
    ae_analysis: pd.DataFrame, output_path: Path | str
) -> None:
    """Plota erro de reconstrução do Autoencoder.

    Args:
        ae_analysis: DataFrame com análise de erro.
        output_path: Caminho para salvar figura.
    """
    output_path = Path(output_path)

    logger.info("Plotando erros de reconstrução do Autoencoder...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 10 features
    top_10 = ae_analysis.head(10)
    axes[0].barh(range(len(top_10)), top_10["mean_error"], color=NORD_PALETTE[3])
    axes[0].set_yticks(range(len(top_10)))
    axes[0].set_yticklabels(top_10["feature"])
    axes[0].set_xlabel("Erro Médio de Reconstrução")
    axes[0].set_title("Top 10 Features - Maior Erro de Reconstrução")
    axes[0].invert_yaxis()

    # Distribuição geral
    axes[1].bar(
        range(len(ae_analysis)),
        ae_analysis["mean_error"],
        yerr=ae_analysis["std_error"],
        color=NORD_PALETTE[0],
        alpha=0.7,
        capsize=3,
    )
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Erro de Reconstrução")
    axes[1].set_title("Erro de Reconstrução por Feature (com desvio padrão)")
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.savefig(
        output_path / "autoencoder_reconstruction_errors.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        f"Gráfico de erros salvo: {output_path / 'autoencoder_reconstruction_errors.png'}"
    )


def generate_all_explanations(
    X_test: pd.DataFrame,
    models_path: Path | str,
    output_path: Path | str = "outputs/explanations",
    sample_size: int | None = 1000,
) -> None:
    """Pipeline completo de explicabilidade.

    Gera SHAP values, plots e análises para todos os modelos.

    Args:
        X_test: Features de teste.
        models_path: Caminho dos modelos.
        output_path: Caminho para salvar explicações.
        sample_size: Tamanho da amostra para SHAP (None = usar tudo).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GERANDO EXPLICABILIDADE COM SHAP")
    logger.info("=" * 60)

    # Amostrar dados se necessário
    if sample_size and len(X_test) > sample_size:
        X_shap = X_test.sample(n=sample_size, random_state=42)
        logger.info(
            f"Usando {sample_size} amostras para SHAP "
            f"(de {len(X_test)} total)"
        )
    else:
        X_shap = X_test

    # Random Forest
    logger.info("\n📊 Random Forest...")
    rf_shap, rf_explainer = generate_shap_values(
        X_shap, "random_forest", models_path, sample_size=None
    )
    plot_shap_summary(rf_shap, X_shap, "random_forest", output_path)
    plot_shap_beeswarm(rf_shap, X_shap, "random_forest", output_path)

    # Selecionar exemplos interessantes (fraude com maior probabilidade)
    rf_model = joblib.load(Path(models_path) / "random_forest.joblib")
    rf_proba = rf_model.predict_proba(X_shap)[:, 1]
    fraud_indices = np.argsort(rf_proba)[-3:][::-1]  # Top 3 fraudes
    for idx in fraud_indices:
        plot_shap_waterfall(rf_shap, X_shap, int(idx), "rf", output_path)

    # XGBoost (skip se houver problemas de compatibilidade com SHAP)
    logger.info("\n📊 XGBoost...")
    try:
        xgb_shap, xgb_explainer = generate_shap_values(
            X_shap, "xgboost", models_path, sample_size=None
        )
        plot_shap_summary(xgb_shap, X_shap, "xgboost", output_path)
        plot_shap_beeswarm(xgb_shap, X_shap, "xgboost", output_path)

        # Gráficos de impacto para XGBoost
        xgb_model = joblib.load(Path(models_path) / "xgboost.joblib")
        xgb_proba = xgb_model.predict_proba(X_shap)[:, 1]
        xgb_fraud_indices = np.argsort(xgb_proba)[-3:][::-1]
        for idx in xgb_fraud_indices:
            plot_shap_waterfall(xgb_shap, X_shap, int(idx), "xgb", output_path)
    except Exception as e:
        logger.warning(
            f"⚠️  Falha ao gerar explicações para XGBoost: {e}\n"
            "    Continuando com Autoencoder..."
        )

    # Autoencoder (usar dataset completo)
    logger.info("\n📊 Autoencoder...")
    ae_analysis = analyze_autoencoder_reconstruction(
        X_test, models_path, output_path
    )
    plot_autoencoder_errors(ae_analysis, output_path)

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Explicações salvas em: {output_path}")
    logger.info("=" * 60)


def main() -> None:
    """Executa explicabilidade para dados de teste."""
    from .ensemble import load_and_preprocess

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "creditcard.csv"
    models_path = project_root / "models"
    output_path = project_root / "outputs" / "explanations"

    logger.info("Carregando dados...")
    _, X_test, _, _ = load_and_preprocess(data_path)

    generate_all_explanations(X_test, models_path, output_path)


if __name__ == "__main__":
    main()
