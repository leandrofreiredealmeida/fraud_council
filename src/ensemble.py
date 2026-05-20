"""Meta-modelo ensemble que aprende a pesar as predições dos especialistas."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess(data_path: Path | str) -> tuple:
    """Carrega e faz pré-processamento dos dados.

    Retorna X_train, X_test, y_train, y_test (after normalization).
    Salva o scaler em models/scaler.joblib para uso em inferência.
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalização de Time e Amount
    scaler = StandardScaler()
    cols_to_scale = ["Time", "Amount"]

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Salvar scaler para inferência
    models_path = data_path.parent.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_path / "scaler.joblib")

    return X_train, X_test, y_train, y_test


def generate_meta_features(
    X_test: pd.DataFrame, models_path: Path | str
) -> np.ndarray:
    """Gera matriz de meta-features: scores de cada especialista.

    Retorna array de shape (n_samples, 3) com [rf_score, xgb_score, ae_score].
    """
    models_path = Path(models_path)

    # Carregar modelos
    rf_model = joblib.load(models_path / "random_forest.joblib")
    xgb_model = joblib.load(models_path / "xgboost.joblib")
    autoencoder = joblib.load(models_path / "autoencoder.joblib")
    ae_threshold = np.load(models_path / "autoencoder_threshold.npy")

    # Random Forest score (probabilidade classe 1)
    rf_scores = rf_model.predict_proba(X_test)[:, 1]

    # XGBoost score (probabilidade classe 1)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

    # Autoencoder score (MSE / anomaly score)
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test
    ae_recon = autoencoder.predict(X_test_arr)
    ae_scores = np.mean((X_test_arr - ae_recon) ** 2, axis=1)

    # Normalizar autoencoder score para [0, 1]
    # Usar sigmoid para converter MSE em probabilidade
    ae_scores_normalized = 1 / (1 + np.exp(-10 * (ae_scores - ae_threshold)))

    # Montar matriz de meta-features
    meta_features = np.column_stack([rf_scores, xgb_scores, ae_scores_normalized])

    return meta_features


def train_ensemble(
    meta_features: np.ndarray, y_test: np.ndarray, models_path: Path | str
) -> LogisticRegression:
    """Treina a regressão logística sobre as meta-features.

    Args:
        meta_features: Array (n_samples, 3) com scores dos especialistas.
        y_test: Array (n_samples,) com labels verdadeiros.
        models_path: Caminho onde salvar o meta-modelo.

    Returns:
        Meta-modelo treinado.
    """
    models_path = Path(models_path)

    # Treinar regressão logística leve
    meta_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    meta_model.fit(meta_features, y_test)

    # Avaliação
    y_pred = meta_model.predict(meta_features)
    y_proba = meta_model.predict_proba(meta_features)[:, 1]

    # Métricas
    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("META-MODELO (REGRESSÃO LOGÍSTICA)")
    print("=" * 60)
    print(f"\nPesos dos especialistas:")
    print(f"  Random Forest  : {meta_model.coef_[0, 0]:.4f}")
    print(f"  XGBoost        : {meta_model.coef_[0, 1]:.4f}")
    print(f"  Autoencoder    : {meta_model.coef_[0, 2]:.4f}")
    print(f"  Intercepto     : {meta_model.intercept_[0]:.4f}")

    print("\nMétricas de Avaliação:")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Legítima", "Fraude"]
        )
    )

    # Salvar meta-modelo
    joblib.dump(meta_model, models_path / "ensemble_meta_model.joblib")
    print(f"\nMeta-modelo salvo: {models_path / 'ensemble_meta_model.joblib'}")

    return meta_model


def main() -> None:
    """Executa todo o pipeline do meta-modelo."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "creditcard.csv"
    models_path = project_root / "models"

    print("Carregando e pré-processando dados...")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    print("Gerando meta-features dos especialistas...")
    meta_features = generate_meta_features(X_test, models_path)

    print("Treinando meta-modelo...")
    train_ensemble(meta_features, y_test, models_path)

    print("\n✅ Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
