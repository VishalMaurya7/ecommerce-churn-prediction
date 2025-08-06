import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: try importing SHAP
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

st.set_page_config(page_title="E-commerce Churn Predictor", layout="wide")
st.title("🛍️ E-commerce Customer Churn Prediction")

# --- Load model and data ---
try:
    model = joblib.load("notebooks/outputs/churn_model.pkl")
    X = pd.read_pickle("notebooks/outputs/X.pkl")
    y = pd.read_pickle("notebooks/outputs/y.pkl")
except Exception as e:
    st.error(f"❌ Could not load model or data: {e}")
    st.stop()

# --- Preprocess ---
if 'customer_unique_id' in X.columns:
    X_display = X.copy()
    X = X.drop(columns=['customer_unique_id'])
else:
    X_display = X.copy()

# --- Sidebar: Select Customer ---
st.sidebar.header("🔍 Select a Customer")
index = st.sidebar.number_input("Customer Index", min_value=0, max_value=len(X)-1, value=0)

# Get selected customer data
sample = X.iloc[[index]]
sample_display = X_display.iloc[[index]]

# --- Predict ---
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

# --- Show Prediction ---
st.subheader("🔮 Prediction Result")
st.write(f"**Prediction:** {'🔴 Churn' if prediction else '🟢 Not Churn'}")
st.write(f"**Churn Probability:** `{probability:.2f}`")

# --- Feature Importance Plot ---
st.subheader("📊 Top 10 Most Important Features (Global)")
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig1, ax = plt.subplots()
sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', ax=ax)
st.pyplot(fig1)

# --- Customer Info ---
with st.expander("📋 View Customer Data"):
    st.dataframe(sample_display.T)

# Upload external data
st.sidebar.markdown("### 📤 Upload External Data (optional)")
uploaded_file = st.sidebar.file_uploader("Upload a .csv file", type=["csv"])

if uploaded_file is not None:
    try:
        external_df = pd.read_csv(uploaded_file)

        # -- Apply same preprocessing here --
        external_df = external_df.drop(columns=["customer_unique_id"], errors="ignore")

        # Optional: reorder columns to match training data
        external_df = external_df[X.columns]

        st.success("✅ External data loaded!")

        # Predict
        external_preds = model.predict(external_df)
        external_probs = model.predict_proba(external_df)[:, 1]

        st.markdown("### 🔮 External Data Predictions")
        results = external_df.copy()
        results["Prediction"] = ["Churn" if p == 1 else "Not Churn" for p in external_preds]
        results["Churn Probability"] = external_probs.round(2)

        st.dataframe(results)

    except Exception as e:
        st.error(f"⚠️ Failed to process uploaded data: {e}")

        
missing = set(X.columns) - set(external_df.columns)
extra = set(external_df.columns) - set(X.columns)
if missing:
    st.warning(f"Missing columns: {missing}")
if extra:
    st.warning(f"Extra columns: {extra}")

