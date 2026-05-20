"""Dashboard Streamlit - The Council"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path

# Suprimir warning de feature names do sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib import rcParams
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.ensemble import generate_meta_features, load_and_preprocess

# Configuração visual
st.set_page_config(page_title="The Council", layout="wide", initial_sidebar_state="expanded")

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


@st.cache_resource
def load_models() -> dict:
    """Carrega todos os modelos e artefatos necessários."""
    models_path = Path("models")
    return {
        "rf": joblib.load(models_path / "random_forest.joblib"),
        "xgb": joblib.load(models_path / "xgboost.joblib"),
        "ae": joblib.load(models_path / "autoencoder.joblib"),
        "meta": joblib.load(models_path / "ensemble_meta_model.joblib"),
        "scaler": joblib.load(models_path / "scaler.joblib"),
        "ae_threshold": np.load(models_path / "autoencoder_threshold.npy"),
    }


@st.cache_data
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega dados de teste pré-salvos ou reconstrói a partir do CSV original."""
    x_path = Path("models") / "X_test.parquet"
    y_path = Path("models") / "y_test.parquet"
    if x_path.exists() and y_path.exists():
        X_test = pd.read_parquet(x_path)
        y_test = pd.read_parquet(y_path)["Class"]
        return X_test, y_test
    _, X_test, _, y_test = load_and_preprocess(Path("data") / "creditcard.csv")
    return X_test, y_test


@st.cache_data
def compute_metrics(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de cada modelo e ensemble."""
    models = load_models()
    meta_features = generate_meta_features(X_test, "models")

    # Predições
    rf_proba = models["rf"].predict_proba(X_test)[:, 1]
    xgb_proba = models["xgb"].predict_proba(X_test)[:, 1]
    ensemble_proba = models["meta"].predict_proba(meta_features)[:, 1]

    # Threshold padrão de 0.5
    rf_pred = (rf_proba >= 0.5).astype(int)
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    # Métricas
    metrics = {}
    for name, pred, proba in [
        ("Random Forest", rf_pred, rf_proba),
        ("XGBoost", xgb_pred, xgb_proba),
        ("Ensemble", ensemble_pred, ensemble_proba),
    ]:
        metrics[name] = {
            "auc": float(roc_auc_score(y_test, proba)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
        }

    return metrics


def tab1_visao_geral():
    """Aba 1: Visão Geral do Ensemble"""
    st.header("📊 Visão Geral")

    X_test, y_test = load_test_data()
    metrics = compute_metrics(X_test, y_test)

    # Cards de métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC", f"{metrics['Ensemble']['auc']:.4f}")
    with col2:
        st.metric("F1-Score", f"{metrics['Ensemble']['f1']:.4f}")
    with col3:
        st.metric("Precision", f"{metrics['Ensemble']['precision']:.4f}")
    with col4:
        st.metric("Recall", f"{metrics['Ensemble']['recall']:.4f}")

    # Gráfico de desbalanceamento
    st.subheader("Desbalanceamento de Classes")
    class_counts = y_test.value_counts()
    class_labels = ["🔴 Fraude" if x == 1 else "🟢 Legítima" for x in class_counts.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [NORD_PALETTE[3] if x == 1 else NORD_PALETTE[0] for x in class_counts.index]
    ax.barh(class_labels, class_counts.values, color=colors, alpha=0.7)
    ax.set_xlabel("Quantidade")
    ax.set_title("Distribuicao de Classes no Dataset de Teste")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.pyplot(fig, width="stretch")

    # Tabela comparativa
    st.subheader("Comparativo de Performance")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, width="stretch")


def _render_council_verdict(X_sample: pd.DataFrame, models: dict) -> None:
    """Exibe os votos dos especialistas e o veredito final do meta-modelo."""
    meta_features_sample = generate_meta_features(X_sample, "models")

    rf_score = float(models["rf"].predict_proba(X_sample)[0, 1])
    xgb_score = float(models["xgb"].predict_proba(X_sample)[0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ae_recon = models["ae"].predict(X_sample)
    ae_mse = np.mean((X_sample.values - ae_recon) ** 2)
    ae_score = float(1 / (1 + np.exp(-10 * (ae_mse - models["ae_threshold"]))))

    ensemble_score = float(models["meta"].predict_proba(meta_features_sample)[0, 1])

    st.subheader("Votos dos Especialistas")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌲 Random Forest", f"{rf_score:.3f}")
        st.progress(min(rf_score, 1.0))

    with col2:
        st.metric("⚡ XGBoost", f"{xgb_score:.3f}")
        st.progress(min(xgb_score, 1.0))

    with col3:
        st.metric("🔧 Autoencoder", f"{ae_score:.3f}")
        st.progress(min(ae_score, 1.0))

    st.subheader("Veredito Final")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Score de Confiança:** {ensemble_score:.4f}")
        st.progress(min(ensemble_score, 1.0))

    with col2:
        if ensemble_score >= 0.5:
            st.markdown("## 🔴 FRAUDE")
        else:
            st.markdown("## 🟢 LEGÍTIMA")


def _render_manual_input() -> pd.DataFrame:
    """Formulário de entrada manual de transação.

    Retorna DataFrame de uma linha com as 30 features na ordem correta,
    já com Time e Amount em escala bruta (o scaler é aplicado depois).
    """
    st.info(
        "V1–V28 são componentes anônimos de PCA extraídos dos dados originais. "
        "O valor **0.0** representa o comportamento médio de uma transação legítima."
    )

    col_time, col_amount = st.columns(2)
    with col_time:
        picked_time = st.time_input(
            "Horário da transação",
            value=datetime.time(12, 0),
        )
        time_val = picked_time.hour * 3600 + picked_time.minute * 60
        st.caption(f"Equivalente a {time_val:,} segundos no dataset")
    with col_amount:
        amount_val = st.number_input(
            "Valor da transação",
            min_value=0.0,
            max_value=25691.16,
            value=100.0,
            step=0.01,
            format="%.2f",
        )

    v_values: dict[str, float] = {}
    with st.expander("Ajustar componentes V1–V28", expanded=False):
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.slider(
                    f"V{i}",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.01,
                    key=f"manual_v{i}",
                )

    row = {"Time": float(time_val), "Amount": float(amount_val)}
    row.update(v_values)

    feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return pd.DataFrame([row])[feature_order]


def _preprocess_manual_transaction(
    raw_df: pd.DataFrame, scaler: object
) -> pd.DataFrame:
    """Aplica o scaler em Time e Amount mantendo as features V inalteradas."""
    df = raw_df.copy()
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    return df


def tab2_council_em_acao():
    """Aba 2: The Council em Ação"""
    st.header("⚖️ The Council em Ação")

    models = load_models()
    X_test, y_test = load_test_data()

    mode = st.radio(
        "Modo de análise:",
        ["Transação do dataset", "Transação manual"],
        horizontal=True,
    )
    st.session_state["analysis_mode"] = (
        "dataset" if mode == "Transação do dataset" else "manual"
    )

    st.divider()

    if mode == "Transação do dataset":
        max_idx = len(X_test) - 1
        selected_idx = st.number_input(
            f"Selecione a transação (0 a {max_idx}):",
            min_value=0,
            max_value=max_idx,
            step=1,
            key="selected_idx",
        )

        true_label = y_test.iloc[selected_idx]
        true_label_str = "🟢 Legítima" if true_label == 0 else "🔴 Fraude"
        st.write(f"**Label real:** {true_label_str}")

        X_sample = X_test.iloc[[selected_idx]]
        _render_council_verdict(X_sample, models)

    else:
        raw_df = _render_manual_input()
        if st.button("Analisar transação", type="primary"):
            X_sample = _preprocess_manual_transaction(raw_df, models["scaler"])
            _render_council_verdict(X_sample, models)


def tab3_explicabilidade():
    """Aba 3: Explicabilidade"""
    st.header("🔍 Explicabilidade")

    if st.session_state.get("analysis_mode") == "manual":
        st.info(
            "A análise SHAP por transação está disponível apenas para transações do dataset. "
            "Volte à aba **The Council em Ação** e selecione o modo **Transação do dataset**."
        )

    models = load_models()
    X_test, y_test = load_test_data()
    selected_idx = st.session_state.get("selected_idx", 0)

    # Selectbox para escolher modelo SHAP
    modelo_shap = st.selectbox(
        "Selecione o modelo para visualizar SHAP summary:",
        ["Random Forest", "XGBoost"],
    )

    # Exibir summary plot
    st.subheader(f"SHAP Summary Plot - {modelo_shap}")
    img_path = f"outputs/explanations/shap_summary_{modelo_shap.lower().replace(' ', '_')}.png"
    try:
        st.image(img_path, width="stretch")
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {img_path}")

    # Waterfall plot da transação selecionada
    st.subheader(f"Impacto das Features (Transação #{selected_idx})")

    X_sample = X_test.iloc[[selected_idx]]

    if modelo_shap == "Random Forest":
        # Gerar SHAP values para RF (rápido com TreeExplainer)
        explainer = shap.TreeExplainer(models["rf"])
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, (list, tuple)):
            shap_values = np.asarray(shap_values[1])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = np.asarray(shap_values)

        # Extrair SHAP values para a transação
        shap_idx = shap_values[0]
        feature_names = X_test.columns.tolist()

        # Top 10 features
        importance = np.abs(shap_idx)
        top_indices = np.argsort(importance)[-10:][::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            NORD_PALETTE[3] if shap_idx[i] > 0 else NORD_PALETTE[0]
            for i in top_indices
        ]
        ax.barh(range(len(top_indices)), shap_idx[top_indices], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([feature_names[i] for i in top_indices])
        ax.set_xlabel("SHAP Value (Impacto)")
        ax.set_title(f"Top 10 Features - {modelo_shap} - Transacao #{selected_idx}")
        ax.axvline(x=0, color=NORD_PALETTE[4], linestyle="--", linewidth=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.pyplot(fig, width="stretch")

        # Tabela top 5 features
        st.subheader("Top 5 Features")
        top5_indices = top_indices[:5]
        top5_data = pd.DataFrame(
            {
                "Feature": [feature_names[i] for i in top5_indices],
                "SHAP Value": [float(shap_idx[i]) for i in top5_indices],
                "Direção": [
                    "▲ Aumenta suspeita" if shap_idx[i] > 0 else "▼ Reduz suspeita"
                    for i in top5_indices
                ],
            }
        )
        st.dataframe(top5_data, width="stretch")

    else:
        # Listar gráficos disponíveis para XGBoost
        import glob
        xgb_plots = sorted(glob.glob("outputs/explanations/shap_impact_xgb_idx*.png"))

        if xgb_plots:
            # Extrair índices dos nomes dos arquivos
            indices = [int(f.split("idx")[1].split(".png")[0]) for f in xgb_plots]
            selected_plot_idx = st.selectbox(
                "Selecione uma transação com gráfico pré-calculado:",
                indices,
                format_func=lambda x: f"Transação #{x}"
            )

            # Encontrar o arquivo correspondente
            plot_file = next(f for f in xgb_plots if f"idx{selected_plot_idx}.png" in f)
            st.image(plot_file, width="stretch")
        else:
            st.warning(
                "Nenhum gráfico SHAP pré-calculado encontrado para XGBoost. "
                "Execute `python -m src.explain` para gerar os gráficos."
            )


def main():
    """Função principal do dashboard."""
    st.title("🏛️ The Fraud Council")
    st.write(
        "Dashboard interativo do ensemble de detecção de fraude com 3 especialistas."
    )

    # Inicializar session_state
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0

    # Criar abas
    tab1, tab2, tab3 = st.tabs(
        ["📊 Visão Geral", "⚖️ The Council em Ação", "🔍 Explicabilidade"]
    )

    with tab1:
        tab1_visao_geral()

    with tab2:
        tab2_council_em_acao()

    with tab3:
        tab3_explicabilidade()


if __name__ == "__main__":
    main()
