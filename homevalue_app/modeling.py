from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def default_schema(columns: list[str]) -> dict[str, dict]:
    return {
        c: {"type": "numeric", "min": 0.0, "max": 10000.0, "default": 1000.0}
        for c in columns
    }


@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)


def normalize_artifact(artifact):
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        columns = artifact.get("feature_columns", [])
        schema = artifact.get("feature_schema", {})
        metrics = artifact.get("metrics", {})
        external = artifact.get("external_test_metrics")
        if not columns:
            columns = list(schema.keys())
        if not schema:
            schema = default_schema(columns)
        return model, columns, schema, metrics, external

    fallback_columns = ["area", "bedrooms", "bathrooms", "floors", "age"]
    fallback_schema = {
        "area": {"type": "numeric", "min": 500.0, "max": 10000.0, "default": 1500.0},
        "bedrooms": {"type": "numeric", "min": 1.0, "max": 8.0, "default": 3.0},
        "bathrooms": {"type": "numeric", "min": 1.0, "max": 6.0, "default": 2.0},
        "floors": {"type": "numeric", "min": 1.0, "max": 4.0, "default": 2.0},
        "age": {"type": "numeric", "min": 0.0, "max": 70.0, "default": 15.0},
    }
    return artifact, fallback_columns, fallback_schema, {}, None


def get_feature_importance_frame(model) -> pd.DataFrame | None:
    try:
        estimator = model.named_steps["model"]
        preprocessor = model.named_steps["preprocess"]
        return (
            pd.DataFrame(
                {
                    "Feature": preprocessor.get_feature_names_out(),
                    "Importance": estimator.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return None


def is_integer_feature(feature: str, meta: dict[str, object]) -> bool:
    if bool(meta.get("is_integer", False)):
        return True
    n = feature.lower()
    return ("bedroom" in n) or ("bathroom" in n)


def render_inputs(columns: list[str], schema: dict[str, dict]) -> dict[str, object]:
    values: dict[str, object] = {}
    c1, c2 = st.columns(2)
    for i, feat in enumerate(columns):
        meta = schema.get(feat, {"type": "numeric", "min": 0.0, "max": 10000.0, "default": 0.0})
        label = feat.replace("_", " ").title()
        col = c1 if i % 2 == 0 else c2

        with col:
            if meta.get("type") == "categorical":
                options = meta.get("options", ["Unknown"]) or ["Unknown"]
                default = meta.get("default", options[0])
                idx = options.index(default) if default in options else 0
                values[feat] = st.selectbox(label, options=options, index=idx, key=f"input_{feat}")
            else:
                mn = float(meta.get("min", 0.0))
                mx = float(meta.get("max", mn + 1.0))
                dv = float(meta.get("default", (mn + mx) / 2))
                if mx <= mn:
                    mx = mn + 1.0

                if is_integer_feature(feat, meta):
                    mn_i = int(round(mn))
                    mx_i = int(round(mx))
                    dv_i = int(round(dv))
                    if mx_i <= mn_i:
                        mx_i = mn_i + 1
                    dv_i = min(max(dv_i, mn_i), mx_i)
                    values[feat] = st.number_input(
                        label,
                        min_value=mn_i,
                        max_value=mx_i,
                        value=dv_i,
                        step=1,
                        key=f"input_{feat}",
                    )
                else:
                    step = 1.0 if (mx - mn) >= 10 else 0.1
                    values[feat] = st.number_input(
                        label,
                        min_value=mn,
                        max_value=mx,
                        value=dv,
                        step=step,
                        key=f"input_{feat}",
                    )
    return values


def render_prediction_section(
    model,
    feature_columns: list[str],
    feature_schema: dict[str, dict],
    metrics: dict,
    external_test_metrics: dict | None,
) -> None:
    st.subheader("Enter Property Details")
    inputs = render_inputs(feature_columns, feature_schema)

    if st.button("Predict Price", type="primary"):
        pred = float(model.predict(pd.DataFrame([inputs], columns=feature_columns))[0])
        st.session_state["last_input_values"] = inputs
        st.session_state["last_prediction"] = pred
        st.success(f"Estimated Price: {pred:,.2f}")
        if metrics:
            st.write(f"Model RMSE: {metrics.get('rmse', float('nan')):,.2f}")
            st.write(f"Model R2: {metrics.get('r2', float('nan')):.4f}")
        if external_test_metrics:
            st.write("External Test RMSE:", f"{external_test_metrics.get('rmse', float('nan')):,.2f}")
            st.write("External Test R2:", f"{external_test_metrics.get('r2', float('nan')):.4f}")


def render_what_if_section(model, feature_columns: list[str], feature_schema: dict[str, dict]) -> None:
    st.subheader("Scenario Simulation")
    if "last_input_values" not in st.session_state:
        st.info("Run one prediction first to enable simulation.")
        return

    base = dict(st.session_state["last_input_values"])
    numeric_features = [
        f for f in feature_columns if feature_schema.get(f, {}).get("type") != "categorical"
    ]
    if not numeric_features:
        st.info("No numeric features available for what-if analysis.")
        return

    feature = st.selectbox("Feature to vary", options=numeric_features)
    meta = feature_schema.get(feature, {})
    mn = float(meta.get("min", 0.0))
    mx = float(meta.get("max", 1.0))
    if mx <= mn:
        mx = mn + 1.0
    grid = np.linspace(mn, mx, 30)

    rows = []
    for value in grid:
        row = dict(base)
        row[feature] = float(value)
        rows.append(row)

    sim_df = pd.DataFrame(rows, columns=feature_columns)
    sim_pred = model.predict(sim_df)
    st.line_chart(pd.DataFrame({feature: grid, "Predicted Price": sim_pred}).set_index(feature))


def render_trend_section() -> None:
    st.subheader("Projected Price Trend")
    if "last_prediction" not in st.session_state:
        st.info("Run one prediction first to create a trend projection.")
        return

    base_price = float(st.session_state["last_prediction"])
    growth = st.slider("Expected annual growth (%)", -10.0, 20.0, 4.0, 0.5)
    years = st.slider("Forecast horizon (years)", 1, 15, 5)
    x = list(range(years + 1))
    y = [base_price * ((1 + growth / 100) ** year) for year in x]
    st.line_chart(pd.DataFrame({"Year": x, "Projected Price": y}).set_index("Year"))


def render_importance_section(model) -> None:
    st.subheader("Model Feature Importance")
    fi = get_feature_importance_frame(model)
    if fi is None or fi.empty:
        st.info("Feature importance is unavailable for this model artifact.")
        return

    top = fi.head(min(20, len(fi)))
    st.bar_chart(top.set_index("Feature"))
    st.dataframe(top, use_container_width=True)
