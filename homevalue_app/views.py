import streamlit as st

from .constants import ABOUT_IMG_1, ABOUT_IMG_2, LOGO_PATH, MODEL_PATH
from .modeling import (
    load_artifact,
    normalize_artifact,
    render_importance_section,
    render_prediction_section,
    render_trend_section,
    render_what_if_section,
)


def render_brand() -> None:
    c1, c2 = st.columns([1, 8])
    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=70)
        else:
            st.markdown("## 🏠")
    with c2:
        st.markdown("## HomeValue AI")
        st.caption("AI Real Estate Price Intelligence System")


def render_about() -> None:
    st.markdown(
        """
        <div class="hero">
            <h3 style="margin:0;">About This App</h3>
            <p style="margin:6px 0 0 0;">
                HomeValue AI predicts house prices, supports what-if simulation, and provides trend and feature insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if ABOUT_IMG_1.exists():
            st.image(str(ABOUT_IMG_1), use_container_width=True)
        st.markdown(
            """
            <div class="card">
                <b>Price Prediction</b><br/>
                Predict estimated market value from property attributes like area, bedrooms, bathrooms, age, and location.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        if ABOUT_IMG_2.exists():
            st.image(str(ABOUT_IMG_2), use_container_width=True)
        st.markdown(
            """
            <div class="card">
                <b>Decision Support</b><br/>
                Explore feature impact, compare scenarios, and view projection trends for better investment planning.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
            <b>Tech Stack:</b> Python, scikit-learn, pandas, Streamlit, joblib<br/>
            <b>Model:</b> Random Forest Regressor with preprocessing pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="card">
                <b>Purpose</b><br/>
                Helps buyers and investors estimate property value quickly from core home features.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <b>What It Shows</b><br/>
                Prediction result, what-if simulation, projected trend, and feature importance insights.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
                <b>How It Works</b><br/>
                Data is preprocessed, model is trained in <code id = "code">train.py</code>, then loaded by <code id = "code">app.py</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <h3 style="margin:0;">AI Real Estate Price Predictor</h3>
            <p style="margin:6px 0 0 0;">
                Predict price, run what-if simulations, check trend projections, and inspect feature importance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("model.pkl was not found.")
        st.code("python train.py --data-path data/housing.csv --model-path model.pkl")
        return

    artifact = load_artifact(MODEL_PATH)
    model, feature_columns, feature_schema, metrics, external_test_metrics = normalize_artifact(
        artifact
    )
    if not feature_columns:
        st.error("No feature columns found in model artifact.")
        return

    tab_predict, tab_what_if, tab_trend, tab_explain = st.tabs(
        ["Prediction", "What-If Simulator", "Price Trend", "Feature Importance"]
    )

    with tab_predict:
        render_prediction_section(
            model,
            feature_columns,
            feature_schema,
            metrics,
            external_test_metrics,
        )
    with tab_what_if:
        render_what_if_section(model, feature_columns, feature_schema)
    with tab_trend:
        render_trend_section()
    with tab_explain:
        render_importance_section(model)
