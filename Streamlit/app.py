import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import time
import requests
import io

st.set_page_config(
    page_title="Credit Risk — Black & Gold",
    layout="wide",
    initial_sidebar_state="expanded"
)

TOP_17 = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "ANNUITY_CREDIT_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_INCOME_RATIO",
    "YEARS_EMPLOYED",
    "APPLICATION_CREDIT_RATIO_MEAN",
    "APPLICATION_CREDIT_RATIO_MAX",
    "APPLICATION_CREDIT_RATIO_STD",
    "CREDIT_GOODS_RATIO_MEAN",
    "CREDIT_GOODS_RATIO_MAX",
    "CREDIT_GOODS_RATIO_STD",
    "APPLICATION_CREDIT_RATIO_MIN",
    "CREDIT_GOODS_RATIO_MIN",
    "SOCIAL_CIRCLE_AVG",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CREDIT_3M_TO_LAST_RATIO_MEAN",
]

MODEL_INFO = {
    'framework': 'LightGBM',
    'objective': 'binary',
    'learning_rate': 0.006,
    'num_leaves': 31,
    'max_depth': 7,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 200,
    'reg_alpha': 5.0,
    'reg_lambda': 10.0,
    'class_weight': 'balanced',
    'n_estimators': 3000,
}

@st.cache_data
def focal_sigmoid(p: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    return np.power(p, gamma)

GAMMA = 2.0  # fixed per request
THRESHOLD = 0.4571  # fixed threshold per request

API_BASE = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_BASE}/predict"
PREDICT_CSV_ENDPOINT = f"{API_BASE}/predict_csv"

# optional sample paths (from this session)
SAMPLE_PATH_10 = "/mnt/data/sample_10_random.csv"
SAMPLE_PATH_5 = "/mnt/data/sample_features_only.csv"

@st.cache_resource
def load_model(path: str = "models/credit_model.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m


model = load_model()

if 'show_single' not in st.session_state:
    st.session_state['show_single'] = False
if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None
if 'show_single_A' not in st.session_state:
    st.session_state['show_single_A'] = False
if 'show_single_B' not in st.session_state:
    st.session_state['show_single_B'] = False

st.markdown(
    """
    <style>
    /* Target main action buttons by aria-label (Streamlit usually sets it equal to the label) */
    button[aria-label="Predict this applicant"],
    button[aria-label="Export inputs CSV"],
    button[aria-label="Run predictions on uploaded CSV"] {
        background: linear-gradient(90deg,#ff6b6b,#ff3b3b) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
    }
    button[aria-label="Predict this applicant"]:hover,
    button[aria-label="Export inputs CSV"]:hover,
    button[aria-label="Run predictions on uploaded CSV"]:hover {
        filter: brightness(1.12);
        transform: translateY(-2px);
    }

    /* Slightly more compact dataframe font */
    [data-testid="stDataFrame"] .row-widget.stDataFrame {font-size:12px;}

    /* Compact the metric blocks a bit */
    .metric-label, .metric-value {
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
menu = st.sidebar.radio("", ["Predict", "About"], index=0)

default_values = {
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "ANNUITY_CREDIT_RATIO": 0.05,
    "ANNUITY_INCOME_RATIO": 0.12,
    "CREDIT_INCOME_RATIO": 2.0,
    "YEARS_EMPLOYED": 5.0,
    "APPLICATION_CREDIT_RATIO_MEAN": 1.0,
    "APPLICATION_CREDIT_RATIO_MAX": 1.0,
    "APPLICATION_CREDIT_RATIO_STD": 0.05,
    "CREDIT_GOODS_RATIO_MEAN": 1.0,
    "CREDIT_GOODS_RATIO_MAX": 1.15,
    "CREDIT_GOODS_RATIO_STD": 0.05,
    "APPLICATION_CREDIT_RATIO_MIN": 1.0,
    "CREDIT_GOODS_RATIO_MIN": 1.0,
    "SOCIAL_CIRCLE_AVG": 2.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1,
    "CREDIT_3M_TO_LAST_RATIO_MEAN": 1.0,
}

def predict_single_via_api_or_local(features_list):
    # features_list: list of 17 values in TOP_17 order
    # Try API first
    try:
        resp = requests.post(PREDICT_ENDPOINT, json={"features": features_list}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # API expected response: {"prediction": float}
            if isinstance(data, dict) and "prediction" in data:
                return float(data["prediction"])
            # or fallback if API returns {"predictions":[...]} for a single-row form
            if isinstance(data, dict) and "predictions" in data and len(data["predictions"]) > 0:
                return float(data["predictions"][0])
    except Exception:
        pass

    # fallback to local model if available
    try:
        if model is not None:
            df_tmp = pd.DataFrame([features_list], columns=TOP_17)
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(df_tmp)[0, 1])
            else:
                out = model.predict(df_tmp)
                return float(np.asarray(out).ravel()[0])
    except Exception:
        pass

    raise RuntimeError("Both API and local model unavailable.")

if menu == "Predict":
    st.title("Predict — Single Applicant")

    # top tabs: Inputs A | Inputs B | Upload CSV
    tab_inputs_a, tab_inputs_b, tab_upload = st.tabs(["Inputs A", "Inputs B", "Upload CSV"])

    # ---------- Inputs A ----------
    with tab_inputs_a:
        st.markdown("#### Section A — personal & credit signals")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            ext2 = st.slider(
                "EXT_SOURCE_2",
                0.0, 1.0,
                float(default_values['EXT_SOURCE_2']),
                step=0.01
            )
            an_credit = st.number_input(
                "ANNUITY_CREDIT_RATIO",
                min_value=0.0, max_value=1.0,
                value=float(default_values['ANNUITY_CREDIT_RATIO']),
                step=0.001,
                format="%.3f"
            )
            cred_income = st.number_input(
                "CREDIT_INCOME_RATIO",
                min_value=0.0, max_value=50.0,
                value=float(default_values['CREDIT_INCOME_RATIO']),
                step=0.01,
                format="%.2f"
            )
        with col2:
            ext3 = st.slider(
                "EXT_SOURCE_3",
                0.0, 1.0,
                float(default_values['EXT_SOURCE_3']),
                step=0.01
            )
            an_income = st.number_input(
                "ANNUITY_INCOME_RATIO",
                min_value=0.0, max_value=1.0,
                value=float(default_values['ANNUITY_INCOME_RATIO']),
                step=0.001,
                format="%.3f"
            )
            years_emp = st.number_input(
                "YEARS_EMPLOYED",
                min_value=0.0, max_value=60.0,
                value=float(default_values['YEARS_EMPLOYED']),
                step=0.1,
                format="%.1f"
            )
        with col3:
            social = st.number_input(
                "SOCIAL_CIRCLE_AVG",
                min_value=0.0, max_value=1000.0,
                value=float(default_values['SOCIAL_CIRCLE_AVG']),
                step=0.25,
                format="%.2f"
            )
            amt_req = st.slider(
                "AMT_REQ_CREDIT_BUREAU_YEAR",
                0, 30,
                int(default_values['AMT_REQ_CREDIT_BUREAU_YEAR'])
            )
            cred_3m = st.number_input(
                "CREDIT_3M_TO_LAST_RATIO_MEAN",
                min_value=0.0, max_value=50.0,
                value=float(default_values['CREDIT_3M_TO_LAST_RATIO_MEAN']),
                step=0.01,
                format="%.2f"
            )

    # ---------- Inputs B ----------
    with tab_inputs_b:
        st.markdown("#### Section B — application aggregates & ratios")
        col4, col5 = st.columns([1, 1])
        with col4:
            app_mean = st.number_input(
                "APPLICATION_CREDIT_RATIO_MEAN",
                min_value=0.0, max_value=10.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_MEAN']),
                step=0.01,
                format="%.2f"
            )
            app_max = st.number_input(
                "APPLICATION_CREDIT_RATIO_MAX",
                min_value=0.0, max_value=20.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_MAX']),
                step=0.01,
                format="%.2f"
            )
            app_std = st.number_input(
                "APPLICATION_CREDIT_RATIO_STD",
                min_value=0.0, max_value=10.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_STD']),
                step=0.01,
                format="%.3f"
            )
        with col5:
            cg_mean = st.number_input(
                "CREDIT_GOODS_RATIO_MEAN",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_MEAN']),
                step=0.01,
                format="%.2f"
            )
            cg_max = st.number_input(
                "CREDIT_GOODS_RATIO_MAX",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_MAX']),
                step=0.01,
                format="%.2f"
            )
            cg_std = st.number_input(
                "CREDIT_GOODS_RATIO_STD",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_STD']),
                step=0.001,
                format="%.3f"
            )

        col6, col7 = st.columns([1, 1])
        with col6:
            app_min = st.number_input(
                "APPLICATION_CREDIT_RATIO_MIN",
                min_value=0.0, max_value=10.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_MIN']),
                step=0.01,
                format="%.2f"
            )
        with col7:
            cg_min = st.number_input(
                "CREDIT_GOODS_RATIO_MIN",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_MIN']),
                step=0.01,
                format="%.2f"
            )

    with tab_upload:
        st.markdown("#### Upload CSV")
        uploaded = st.file_uploader("", type=['csv'], key='uploader_tab')

        # Hide single prediction outputs when user opens Upload tab
        st.session_state['show_single'] = False
        st.session_state['show_single_A'] = False
        st.session_state['show_single_B'] = False

        if uploaded is not None:
            try:
                # ensure we read from start and store DataFrame for reuse
                uploaded.seek(0)
                batch_df_preview = pd.read_csv(uploaded)
                uploaded.seek(0)
                st.session_state['uploaded_df'] = batch_df_preview

                missing = [c for c in TOP_17 if c not in batch_df_preview.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns: {missing}")
                else:
                    st.success(f"CSV validated — {len(batch_df_preview)} rows")
                    st.dataframe(batch_df_preview[TOP_17].head(10), use_container_width=True, height=250)

                    # -------- Run predictions on uploaded CSV (only here) --------
                    st.markdown("### Predictions")
                    if st.button("Run predictions on uploaded CSV", key='run_batch_from_tab'):
                        if model is None and not PREDICT_CSV_ENDPOINT:
                            st.error("Model not found and no API configured.")
                        else:
                            try:
                                # prepare CSV bytes (only TOP_17 columns)
                                csv_bytes = batch_df_preview[TOP_17].to_csv(index=False).encode('utf-8')

                                # call FastAPI batch endpoint
                                files = {"file": ("uploaded.csv", csv_bytes, "text/csv")}
                                resp = requests.post(PREDICT_CSV_ENDPOINT, files=files, timeout=60)

                                if resp.status_code != 200:
                                    st.error(f"API error: {resp.status_code} — {resp.text}")
                                else:
                                    data = resp.json()
                                    preds = data.get("predictions", [])
                                    if len(preds) != len(batch_df_preview):
                                        st.warning("API returned different number of predictions than rows. Showing what we have.")
                                    # add raw probability and focal, then label using focal threshold
                                    batch_df_preview['raw_probability'] = np.nan
                                    batch_df_preview.loc[:len(preds)-1, 'raw_probability'] = preds
                                    batch_df_preview['focal_probability'] = focal_sigmoid(batch_df_preview['raw_probability'].fillna(0).to_numpy(), gamma=GAMMA)
                                    batch_df_preview['prediction'] = np.where(batch_df_preview['focal_probability'] >= THRESHOLD, "RISKY", "LOW RISK")

                                    st.success("Batch predictions completed (via API).")
                                    st.dataframe(batch_df_preview[['prediction', 'raw_probability']].head(20), use_container_width=True)

                                    st.download_button(
                                        "Download predictions CSV",
                                        batch_df_preview.to_csv(index=False).encode('utf-8'),
                                        file_name='batch_predictions.csv',
                                        mime='text/csv',
                                        key='download_batch'
                                    )
                            except Exception as e:
                                st.error(f"Batch prediction failed: {e}")

            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")

    # ---------- Build single-row DF (order critical) ----------
    df_single = pd.DataFrame([{
        'EXT_SOURCE_2': ext2,
        'EXT_SOURCE_3': ext3,
        'ANNUITY_CREDIT_RATIO': an_credit,
        'ANNUITY_INCOME_RATIO': an_income,
        'CREDIT_INCOME_RATIO': cred_income,
        'YEARS_EMPLOYED': years_emp,
        'APPLICATION_CREDIT_RATIO_MEAN': app_mean,
        'APPLICATION_CREDIT_RATIO_MAX': app_max,
        'APPLICATION_CREDIT_RATIO_STD': app_std,
        'CREDIT_GOODS_RATIO_MEAN': cg_mean,
        'CREDIT_GOODS_RATIO_MAX': cg_max,
        'CREDIT_GOODS_RATIO_STD': cg_std,
        'APPLICATION_CREDIT_RATIO_MIN': app_min,
        'CREDIT_GOODS_RATIO_MIN': cg_min,
        'SOCIAL_CIRCLE_AVG': social,
        'AMT_REQ_CREDIT_BUREAU_YEAR': amt_req,
        'CREDIT_3M_TO_LAST_RATIO_MEAN': cred_3m,
    }])[TOP_17]

    # ========= Single applicant preview + buttons داخل Inputs A =========
    with tab_inputs_a:
        st.markdown("### Inputs preview")
        st.dataframe(df_single.T.rename(columns={0: 'value'}), use_container_width=True, height=200)

        st.markdown("---")
        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            predict_btn_a = st.button("Predict this applicant", key='single_predict_A')
            if predict_btn_a:
                # ✅ لما تضغط من A خلّي النتيجة تظهر في A و B
                st.session_state['show_single_A'] = True
                st.session_state['show_single_B'] = True
        with bcol2:
            csv_bytes_a = df_single.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export inputs CSV",
                csv_bytes_a,
                file_name='single_input.csv',
                mime='text/csv',
                key='download_single_A'
            )

        st.markdown("### Predictions")
        if st.session_state.get('show_single_A', False):
            if model is None and not PREDICT_ENDPOINT:
                st.error("Model not found and no API configured.")
            else:
                progress_a = st.progress(0)
                for i in range(0, 100, 10):
                    time.sleep(0.02)
                    progress_a.progress(i + 10)

                # prepare features in order
                features_list = df_single.iloc[0][TOP_17].tolist()
                try:
                    raw_a = predict_single_via_api_or_local(features_list)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    raw_a = None

                if raw_a is not None:
                    focal_a = focal_sigmoid(np.array([raw_a]), gamma=GAMMA)[0]
                    st.metric("Raw probability", f"{raw_a * 100:.2f}%")
                    label_a = "RISKY" if focal_a >= THRESHOLD else "LOW RISK"
                    color_a = "#ff5b5b" if label_a == "RISKY" else "#9AD3BC"
                    st.markdown(
                        f"""
                        <div style='padding:18px;border-radius:12px;
                             background:linear-gradient(90deg,#0b0b0b,#121212);
                             border:2px solid rgba(255,91,91,0.12);'>
                            <h1 style='color:{color_a};text-align:center'>{label_a}</h1>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander("Explain (brief)", expanded=False):
                        st.write(
                            "- Raw probability from the model.\n"
                            "- Focal transform applied (`p**gamma`) with gamma fixed to 2.\n"
                            "- Threshold on focal decides label."
                        )

    # ========= Single applicant preview + buttons داخل Inputs B =========
    with tab_inputs_b:
        st.markdown("### Inputs preview")
        st.dataframe(df_single.T.rename(columns={0: 'value'}), use_container_width=True, height=200)

        st.markdown("---")
        bcol1b, bcol2b = st.columns([1, 1])
        with bcol1b:
            predict_btn_b = st.button("Predict this applicant", key='single_predict_B')
            if predict_btn_b:
                st.session_state['show_single_B'] = True
                st.session_state['show_single_A'] = True
        with bcol2b:
            csv_bytes_b = df_single.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export inputs CSV",
                csv_bytes_b,
                file_name='single_input.csv',
                mime='text/csv',
                key='download_single_B'
            )

        st.markdown("### Predictions")
        if st.session_state.get('show_single_B', False):
            if model is None and not PREDICT_ENDPOINT:
                st.error("Model not found and no API configured.")
            else:
                progress_b = st.progress(0)
                for i in range(0, 100, 10):
                    time.sleep(0.02)
                    progress_b.progress(i + 10)

                # prepare features in order
                features_list_b = df_single.iloc[0][TOP_17].tolist()
                try:
                    raw_b = predict_single_via_api_or_local(features_list_b)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    raw_b = None

                if raw_b is not None:
                    focal_b = focal_sigmoid(np.array([raw_b]), gamma=GAMMA)[0]
                    label_b = "RISKY" if focal_b >= THRESHOLD else "LOW RISK"
                    color_b = "#ff5b5b" if label_b == "RISKY" else "#9AD3BC"
                    st.markdown(
                        f"""
                        <div style='padding:18px;border-radius:12px;
                             background:linear-gradient(90deg,#0b0b0b,#121212);
                             border:2px solid rgba(255,91,91,0.12);'>
                            <h1 style='color:{color_b};text-align:center'>{label_b}</h1>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander("Explain (brief)", expanded=False):
                        st.write(
                            "- Raw probability from the model.\n"
                            "- Focal transform applied (`p**gamma`) with gamma fixed to 2.\n"
                            "- Threshold on focal decides label."
                        )

# -----------------------------
# About page — minimal display
# -----------------------------
else:
    st.title("About the Model")

    st.markdown(
        """
        ### **Loan Defaulter Prediction Model**
        This model predicts the likelihood of loan default using financial, behavioral, and historical credit indicators.

        ---
        """
    )

    st.subheader("Model Overview")
    st.markdown(
        """
        - **Type:** Binary Classification (Good vs Risky Applicant)  
        - **Algorithm:** LightGBM Gradient Boosted Trees  
        - **Focal Probability Adjustment:** Enabled with γ = 2  
        - **Calibrated Output:** Post-processing to improve decision confidence  
        """
    )

    st.subheader("Why this model?")
    st.markdown(
        """
        • Designed for **credit approval & risk assessment**  
        • Handles **imbalanced data** using focal transformation  
        • Prioritizes **high recall on risky applicants**  
        • Supports **real-time single predictions + batch predictions**
        """
    )

    st.subheader("Notes")
    st.markdown(
        """
        - This deployed model is optimized for inference only  
        - Make sure input features match training schema  
        - Threshold can be tuned based on internal policy
        """
    )

