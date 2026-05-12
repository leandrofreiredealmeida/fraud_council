Este é um projeto de detecção de fraude de transação de cartão de crédito que utiliza um conselho de modelos de ML para chegar ao veredito.

O conjunto de dados foi obtido a partir do famoso [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Durante o projeto retirei o Isolation Forest por causa do recall de apenas 47% mesmo após o finetune (valor muito baixo e inaceitável). Com isso, decidi manter apenas 3 especialistas.

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