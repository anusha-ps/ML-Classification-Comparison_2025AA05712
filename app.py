import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Breast Cancer Classification Model Evaluation App",  page_icon="🎗️", layout="wide")

st.title("🎗️Breast Cancer Classification Model Evaluation")
st.write(
    "Upload a test dataset and evaluate different trained classification models."
)

# -----------------------------
# Load Models & Scaler
# -----------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

models = load_models()
scaler = load_scaler()

#Step 1: Download Sample Test Data
st.subheader("⬇️ Download Sample Test Dataset")

try:
    sample_df = pd.read_csv("data/sample_test_data.csv")
    st.download_button(
        label="Download Sample Test Data",
        data=sample_df.to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )
except:
    st.info("Sample test dataset not found.")



# -----------------------------
# Upload Dataset
# -----------------------------
st.subheader("📂 Step 1: Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df_test = pd.read_csv(uploaded_file)

    if "diagnosis" not in df_test.columns:
        st.error("Uploaded file must contain 'diagnosis' column.")
    else:

        # Separate features and target
        X_test = df_test.drop("diagnosis", axis=1)
        y_test = df_test["diagnosis"].map({'M': 1, 'B': 0})

        # -----------------------------
        # Model Selection
        # -----------------------------
        st.subheader("🤖 Step 2: Select Model")

        model_name = st.selectbox(
            "Choose a Model",
            list(models.keys())
        )

        # -----------------------------
        # Predict Button
        # -----------------------------
        if st.button("Predict⚡"):

            model = models[model_name]

            # Apply same scaler used during training
            X_test_scaled = scaler.transform(X_test)

            y_pred = model.predict(X_test_scaled)

            try:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            except:
                y_prob = y_pred

            # -----------------------------
            # Evaluation Metrics
            # -----------------------------
            st.subheader("📈Evaluation Metrics")

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

            metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            # st.table(metrics_df)

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            col2.metric("AUC", f"{metrics['AUC']:.3f}")
            col3.metric("Precision", f"{metrics['Precision']:.3f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{metrics['Recall']:.3f}")
            col5.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
            col6.metric("MCC", f"{metrics['MCC']:.3f}")




            # -----------------------------
            # Confusion Matrix
            # -----------------------------
            st.subheader("📊Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(4, 3), dpi = 500)
            heatmap = sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar_kws={"shrink": 0.9},
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                annot_kws= {"size":10},
                ax=ax
            )

            ax.set_xlabel("Predicted Label", fontsize = 10)
            ax.set_ylabel("True Label", fontsize = 10)

            ax.tick_params(axis='both', labelsize=10)

            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)

            plt.tight_layout()

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.pyplot(fig, use_container_width=False)


            # # -----------------------------
            # # Classification Report
            # # -----------------------------
            # st.subheader("Classification Report")

            # report_dict = classification_report(
            #     y_test,
            #     y_pred,
            #     target_names=["Benign (0)", "Malignant (1)"],
            #     output_dict=True
            # )

            # report_df = pd.DataFrame(report_dict).transpose()

            # # Round values for cleaner display
            # report_df = report_df.round(3)

            # # Styled display
            # st.dataframe(
            #     report_df.style.background_gradient(cmap="Blues"),
            #     use_container_width=True
            # )

            # -----------------------------
            # Classification Report
            # -----------------------------
            st.subheader("🧾Classification Report")

            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=["Benign (0)", "Malignant (1)"],
                output_dict=True
            )

            report_df = pd.DataFrame(report_dict).transpose().round(3)

            # Custom styling for header only
            styled_report = report_df.style.set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#7db0cc"),   # light blue
                        ("font-weight", "bold"),
                        ("color", "black"),
                        ("text-align", "center")
                    ]
                }
            ]).set_properties(**{
                "text-align": "center"
            })

            st.dataframe(styled_report, use_container_width=True)

