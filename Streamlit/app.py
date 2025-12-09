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

API_BASE = os.getenv("API_URL", "https://loan-prediction-production-96aa.up.railway.app")
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



st.markdown("""
<style>
/* Hide Streamlit keyboard shortcut hint */
[data-testid="stKeyboardCommand"] {
    display: none !important;
}

/* Hide tooltips that show text on hover */
.stTooltip {
    display: none !important;
}

/* Hide ANY orphan floating text in sidebar */
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div[role="tooltip"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Inject enhanced CSS + ambient orbs (visual improvements only)
# ----------------------

st.markdown(
    """
    <style>

    :root{
        --bg1:#020617;
        --bg2:#020014;
        --accent:#fbbf24;
        --accent-soft:#f97316;
        --accent-soft2:#38bdf8;
        --panel:#020617;
        --panel-soft:#020617;
        --text:#f9fafb;
        --muted:#9ca3af;
    }

    /* ================================
       GLOBAL APP BACKGROUND (ANIMATED)
       ================================ */
    body{
        background: radial-gradient(circle at 0% 0%, #1e293b 0, #020617 45%),
                    radial-gradient(circle at 100% 100%, #020617 0, #000 60%);
        color: var(--text);
    }

    [data-testid="stAppViewContainer"]{
        background: radial-gradient(circle at 0% 0%, #111827 0, #020617 40%),
                    radial-gradient(circle at 100% 100%, #020617 0, #000 55%);
        animation:bgDrift 32s ease-in-out infinite alternate;
    }

    @keyframes bgDrift{
        0%   { background-position:0% 0%,100% 100%; }
        50%  { background-position:20% 10%,80% 90%; }
        100% { background-position:40% 0%,60% 100%; }
    }

    /* ================================
       MAIN PAGE PANEL / FIX OVERFLOW
       ================================ */
    [data-testid="stAppViewContainer"] > .main{
        padding-top:1.5rem;
        padding-bottom:1.5rem;
    }

    .block-container{
        max-width:1180px;
        margin:auto;
        padding:1.5rem 2.2rem 2.4rem;
        border-radius:26px;
        background:radial-gradient(circle at top left,rgba(15,23,42,0.92),rgba(2,6,23,0.98));
        border:1px solid rgba(148,163,184,0.32);
        box-shadow:
            0 30px 90px rgba(0,0,0,0.85),
            inset 0 0 60px rgba(15,23,42,0.9);
        backdrop-filter:blur(24px);

        /* ===== FIX PAGE NOT COMPLETING ===== */
        overflow:visible !important;

        position:relative;
    }

    /* ORBITAL HALO EFFECTS (LIGHT BLOBS) */
    .block-container::before{
        content:"";
        position:absolute;
        width:520px;
        height:520px;
        border-radius:50%;
        background:radial-gradient(circle,rgba(56,189,248,0.20),transparent 65%);
        top:-220px;
        right:-120px;
        filter:blur(3px);
        opacity:0.75;
        mix-blend-mode:screen;
        animation:haloOrbit 26s ease-in-out infinite alternate;
        pointer-events:none;
        z-index:-1;
    }

    .block-container::after{
        content:"";
        position:absolute;
        width:420px;
        height:420px;
        border-radius:50%;
        background:radial-gradient(circle,rgba(248,250,252,0.08),transparent 60%);
        bottom:-180px;
        left:-80px;
        filter:blur(4px);
        opacity:0.5;
        mix-blend-mode:screen;
        animation:haloOrbit2 32s ease-in-out infinite alternate;
        pointer-events:none;
        z-index:-1;
    }

    @keyframes haloOrbit{
        0%{transform:translate3d(0,0,0);}
        50%{transform:translate3d(-30px,20px,0);}
        100%{transform:translate3d(10px,-10px,0);}
    }
    @keyframes haloOrbit2{
        0%{transform:translate3d(0,0,0) scale(1);}
        50%{transform:translate3d(25px,-15px,0) scale(1.05);}
        100%{transform:translate3d(-15px,10px,0) scale(1.02);}
    }

    /* =======================================
       FLOATING ORBS (ANIMATION OPTIMIZED)
       ======================================= */
    .ambient-orb{
        position:fixed;
        border-radius:999px;
        filter:blur(25px);
        opacity:0.55;
        mix-blend-mode:screen;
        pointer-events:none;
        z-index:-2;
    }

    /* Smaller orbs for performance */
    .orb-1{
        width:200px;height:200px;
        background:radial-gradient(circle,rgba(56,189,248,0.32),transparent 60%);
        top:6%;left:4%;
        animation:orbFloat1 38s ease-in-out infinite alternate;
    }
    .orb-2{
        width:170px;height:170px;
        background:radial-gradient(circle,rgba(234,179,8,0.28),transparent 60%);
        bottom:10%;right:6%;
        animation:orbFloat2 42s ease-in-out infinite alternate;
    }
    .orb-3{
        width:220px;height:220px;
        background:radial-gradient(circle,rgba(147,51,234,0.24),transparent 60%);
        top:45%;right:35%;
        animation:orbFloat3 48s ease-in-out infinite alternate;
    }

    /* smooth animations */
    @keyframes orbFloat1{
        0%{transform:translate3d(0,0,0);}
        50%{transform:translate3d(30px,20px,0);}
        100%{transform:translate3d(-10px,10px,0);}
    }
    @keyframes orbFloat2{
        0%{transform:translate3d(0,0,0);}
        50%{transform:translate3d(-25px,-15px,0);}
        100%{transform:translate3d(15px,-5px,0);}
    }
    @keyframes orbFloat3{
        0%{transform:translate3d(0,0,0);}
        50%{transform:translate3d(20px,-20px,0);}
        100%{transform:translate3d(-15px,15px,0);}
    }

    /* ================================
       FIX HEADER hiding behind orbs
       ================================ */
    [data-testid="stHeader"]{
        z-index:9999 !important;
        position:relative !important;
    }

    /* ================================
       SIDEBAR
       ================================ */
    [data-testid="stSidebar"]{
        background:linear-gradient(180deg,#020617,#020617);
        border-right:1px solid rgba(55,65,81,0.8);
        box-shadow:20px 0 45px rgba(0,0,0,0.85);
    }
    [data-testid="stSidebar"] *{
        color:#e5e7eb !important;
        font-family:"Inter",system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
    }
    [data-testid="stSidebar"] h1{
        font-size:1.1rem;
        letter-spacing:.12em;
        text-transform:uppercase;
        margin-bottom:.75rem;
    }

    /* RADIO BUTTONS */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label{
        font-size:.95rem;
        font-weight:500;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label{
        padding:.35rem .4rem;
        border-radius:999px;
        transition:background .2s ease, transform .16s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover{
        background:rgba(148,163,184,0.18);
        transform:translateX(2px);
    }

    /* ================================
       TABS
       ================================ */
    .stTabs [data-baseweb="tab-list"]{
        border-bottom:1px solid rgba(148,163,184,0.4);
        margin-bottom:.35rem;
    }
    .stTabs [data-baseweb="tab"]{
        font-size:.95rem;
        font-weight:500;
        padding-bottom:.55rem;
        color:rgba(148,163,184,0.8);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"]{
        color:#f9fafb;
        border-bottom:2px solid var(--accent-soft);
    }

    /* ================================
       SLIDERS
       ================================ */
    .stSlider > div > div > div{
        background:rgba(31,41,55,0.85);
        border-radius:999px;
    }
    .stSlider [role="slider"]{
        background:linear-gradient(135deg,#f97316,#facc15) !important;
        box-shadow:0 0 0 4px rgba(248,181,0,0.32);
    }

    /* ================================
       INPUTS
       ================================ */
    input[type="number"],
    input[type="text"],
    input[type="email"],
    input[type="password"]{
        background:rgba(15,23,42,0.95);
        border-radius:12px;
        border:1px solid rgba(148,163,184,0.45);
        color:#e5e7eb;
    }
    input[type="number"]:focus,
    input[type="text"]:focus,
    input[type="email"]:focus,
    input[type="password"]:focus{
        border-color:var(--accent-soft2);
        box-shadow:0 0 0 1px rgba(56,189,248,0.6);
    }

    /* ================================
       BUTTONS
       ================================ */
    button[kind="primary"],
    .stDownloadButton button{
        background:linear-gradient(135deg,#f97316,#facc15) !important;
        color:#111827 !important;
        border-radius:999px !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 18px 45px rgba(248,181,0,0.38) !important;
        padding:.55rem 1.5rem !important;
        transition:transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
    }
    button[kind="primary"]:hover,
    .stDownloadButton button:hover{
        transform:translateY(-2px);
        filter:brightness(1.05);
        box-shadow:0 26px 60px rgba(248,181,0,0.5) !important;
    }

    /* ================================
       DATAFRAME
       ================================ */
    [data-testid="stDataFrame"]{
        background:rgba(15,23,42,0.9);
        border-radius:16px;
        border:1px solid rgba(148,163,184,0.45);
        box-shadow:0 18px 50px rgba(15,23,42,0.85);
    }

    /* ================================
       EXPANDER
       ================================ */
    details{
        background:rgba(15,23,42,0.92);
        border-radius:14px;
        border:1px solid rgba(148,163,184,0.55);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ambient orbs (HTML)
st.markdown(
    """
    <div class="ambient-orb orb-1"></div>
    <div class="ambient-orb orb-2"></div>
    <div class="ambient-orb orb-3"></div>
    """,
    unsafe_allow_html=True
)

# ====== keep existing app-level quick CSS patch (original buttons targeted)
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
                "External credit score 2",
                0.0, 1.0,
                float(default_values['EXT_SOURCE_2']),
                step=0.01
            )
            an_credit = st.number_input(
                "Installment-to-loan ratio",
                min_value=0.0, max_value=1.0,
                value=float(default_values['ANNUITY_CREDIT_RATIO']),
                step=0.001,
                format="%.3f"
            )
            cred_income = st.number_input(
                "Loan-to-income ratio",
                min_value=0.0, max_value=50.0,
                value=float(default_values['CREDIT_INCOME_RATIO']),
                step=0.01,
                format="%.2f"
            )
        with col2:
            ext3 = st.slider(
                "External credit score 3",
                0.0, 1.0,
                float(default_values['EXT_SOURCE_3']),
                step=0.01
            )
            an_income = st.number_input(
                "Installment-to-income ratio",
                min_value=0.0, max_value=1.0,
                value=float(default_values['ANNUITY_INCOME_RATIO']),
                step=0.001,
                format="%.3f"
            )
            years_emp = st.number_input(
                "Years employed",
                min_value=0.0, max_value=60.0,
                value=float(default_values['YEARS_EMPLOYED']),
                step=0.1,
                format="%.1f"
            )
        with col3:
            social = st.number_input(
                "Average social-circle impact",
                min_value=0.0, max_value=1000.0,
                value=float(default_values['SOCIAL_CIRCLE_AVG']),
                step=0.25,
                format="%.2f"
            )
            amt_req = st.slider(
                "Credit bureau requests in last year",
                0, 30,
                int(default_values['AMT_REQ_CREDIT_BUREAU_YEAR'])
            )
            cred_3m = st.number_input(
                "Avg 3-months / last loan ratio",
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
                "Average application loan ratio",
                min_value=0.0, max_value=10.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_MEAN']),
                step=0.01,
                format="%.2f"
            )
            app_max = st.number_input(
                "Max application loan ratio",
                min_value=0.0, max_value=20.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_MAX']),
                step=0.01,
                format="%.2f"
            )
            app_std = st.number_input(
                "Application loan ratio variability",
                min_value=0.0, max_value=10.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_STD']),
                step=0.01,
                format="%.3f"
            )
        with col5:
            cg_mean = st.number_input(
                "Average goods-financing ratio",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_MEAN']),
                step=0.01,
                format="%.2f"
            )
            cg_max = st.number_input(
                "Max goods-financing ratio",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_MAX']),
                step=0.01,
                format="%.2f"
            )
            cg_std = st.number_input(
                "Goods-financing ratio variability",
                min_value=0.0, max_value=10.0,
                value=float(default_values['CREDIT_GOODS_RATIO_STD']),
                step=0.001,
                format="%.3f"
            )

        col6, col7 = st.columns([1, 1])
        with col6:
            app_min = st.number_input(
                "Min application loan ratio",
                min_value=0.0, max_value=10.0,
                value=float(default_values['APPLICATION_CREDIT_RATIO_MIN']),
                step=0.01,
                format="%.2f"
            )
        with col7:
            cg_min = st.number_input(
                "Min goods-financing ratio",
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
                    st.metric("Raw Probability", f"{raw_b * 100:.2f}%")
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
