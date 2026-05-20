Este é um projeto de detecção de fraude de transação de cartão de crédito que utiliza um conselho de modelos de ML para chegar ao veredito.

O conjunto de dados foi obtido a partir do famoso [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Durante o projeto retirei o Isolation Forest por causa do recall de apenas 47% mesmo após o finetune (valor muito baixo e inaceitável). Com isso, decidi manter apenas 3 especialistas.

# Treinamento dos modelos especialistas

**Ensemble Final: RF + XGB + AE**
Retirando Isolation Forest (PR-AUC=0.17), o conselho fica:


| Especialista | Recall | Precision | PR-AUC | Papel                   |
| ------------ | ------ | --------- | ------ | ----------------------- |
| RF           | 0.84   | 0.83      | 0.8728 | Generalista confiável   |
| XGB          | 0.87   | 0.24      | 0.8424 | Sensível (captura tudo) |
| AE           | 0.85   | 0.12      | 0.5517 | Comportamental (desvio) |

Força do ensemble:

- Recall complementar: todos capturam 84–87% das fraudes
- Precisões diferentes: RF é confiável (83%), XGB sensível (24%), AE detecta desvio (12%)
- Votos divergentes geram debate SHAP rico: por que AE flagou algo que RF não flagou?

# Construção do meta-modelo

## Meta-features geradas
A matriz foi montada com os 3 scores dos especialistas no conjunto de validação (56,962 transações):

Random Forest: probabilidade da classe 1
XGBoost: probabilidade da classe 1
Autoencoder: anomaly score normalizado via sigmoid

## Meta-modelo Treinado
Tipo: Regressão Logística (leve, rápida, interpretável)
Entrada: 3 features (scores dos especialistas)
Saída: Probabilidade final de fraude [0, 1]

## Pesos Aprendidos

Random Forest  :  4.1800  ← maior confiança
XGBoost        :  3.7330
Autoencoder    :  2.7280
Intercepto     : -8.4824

## Métricas de Avaliação


| Métrica   | Valor  |
| --------- | ------ |
| AUC-ROC   | 0.9786 |
| F1-Score  | 0.8541 |
| Precision | 0.9080 |
| Recall    | 0.8061 |


## Desempenho Comparativo
O ensemble superou os especialistas individuais:

- **F1 original**: máx 0.83 (Random Forest)
- **F1 ensemble**: 0.85 (+2%)
- **Precision**: 0.91 (melhor na redução de falsos positivos)

# Explicabilidade com SHAP

## O que é SHAP?

SHAP (SHapley Additive exPlanations) é um método baseado em Shapley values (teoria dos jogos) que explica a contribuição de cada feature para uma predição.
O TreeExplainer foi otimizado para modelos de árvore (50-100x mais rápido que LIME)

## Explicabilidade

**Módulo:** `src/explain.py` (406 linhas)

### Funcionalidades Implementadas

**Para Random Forest e XGBoost:**
- `TreeExplainer`: explica decisões dos modelos
- `Summary Plot` (bar): importância média de features
- `Beeswarm Plot` (scatter): dispersão de impacto SHAP
- `Impact Plots`: top 10 features por transação

**Para Autoencoder:**
- Análise de erro de reconstrução (MSE) por feature
- Identifica features com comportamento anômalo
- Top 10 features com maior dificuldade de aprendizado

### Otimizações

- **Amostragem**: 1000 amostras (1.75% do dataset) mantendo representatividade
- **Performance**: reduz tempo de ~20 min → ~1.5 min

### Saídas Geradas

```
outputs/explanations/
├── shap_summary_random_forest.png
├── shap_beeswarm_random_forest.png
├── shap_impact_rf_idx*.png (3 transações com maior fraude)
├── autoencoder_reconstruction_errors.png
└── autoencoder_reconstruction_errors.csv
```

### Como Usar

```bash
# Gerar explicações (1.5 min com amostragem)
python -m src.explain

# Ou via script
python run_explanations.py
```

**Como módulo:**
```python
from src.explain import generate_all_explanations
from src.ensemble import load_and_preprocess

_, X_test, _, _ = load_and_preprocess("data/creditcard.csv")
generate_all_explanations(X_test, "models", "outputs/explanations")
```

# Dashboard Streamlit

**Arquivo:** `app.py`

Interface interativa para explorar o ensemble em tempo real, organizada em três abas:

**Visão Geral**
- Cards com as métricas principais do ensemble (AUC, F1, Precision, Recall)
- Gráfico de desbalanceamento de classes no conjunto de teste
- Tabela comparativa de performance entre RF, XGBoost e Ensemble

**The Council em Ação**
- Seleção de qualquer transação do conjunto de teste por índice ou inserção manual de features via formulário
- Exibe o label real e os scores individuais de cada especialista (RF, XGBoost, Autoencoder) com barras de progresso
- Veredito final do meta-modelo com score de confiança

**Explicabilidade**
- SHAP Summary Plot pré-calculado para RF ou XGBoost
- Waterfall plot interativo das top 10 features para a transação selecionada (gerado em tempo real para RF via `TreeExplainer`)
- Tabela com as top 5 features e direção de impacto (aumenta/reduz suspeita)
- Para XGBoost, exibe gráficos pré-calculados de transações de alto risco

## Como executar

```bash
streamlit run app.py
```


