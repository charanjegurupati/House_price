from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model.pkl")


def default_schema(columns: list[str]) -> dict[str, dict]:
    schema: dict[str, dict] = {}
    for column in columns:
        schema[column] = {
            "type": "numeric",
            "min": 0.0,
            "max": 10000.0,
            "default": 1000.0,
        }
    return schema


@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)


def normalize_artifact(artifact):
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        columns = artifact.get("feature_columns", [])
        schema = artifact.get("feature_schema", {})
        metrics = artifact.get("metrics", {})
        external_test_metrics = artifact.get("external_test_metrics")

        if not columns:
            columns = list(schema.keys())
        if not schema:
            schema = default_schema(columns)

        return model, columns, schema, metrics, external_test_metrics

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
        importances = estimator.feature_importances_
        names = preprocessor.get_feature_names_out()
        frame = (
            pd.DataFrame({"Feature": names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        return frame
    except Exception:
        return None


def is_integer_feature(feature: str, meta: dict[str, object]) -> bool:
    if bool(meta.get("is_integer", False)):
        return True

    name = feature.lower()
    # Force common count-style inputs to integer in UI.
    return ("bedroom" in name) or ("bathroom" in name)


def render_inputs(feature_columns: list[str], schema: dict[str, dict]) -> dict[str, object]:
    user_values: dict[str, object] = {}
    left_col, right_col = st.columns(2)

    for idx, feature in enumerate(feature_columns):
        meta = schema.get(
            feature, {"type": "numeric", "min": 0.0, "max": 10000.0, "default": 0.0}
        )
        label = feature.replace("_", " ").title()
        target_col = left_col if idx % 2 == 0 else right_col

        with target_col:
            if meta.get("type") == "categorical":
                options = meta.get("options", ["Unknown"])
                if not options:
                    options = ["Unknown"]
                default = meta.get("default", options[0])
                default_index = options.index(default) if default in options else 0
                user_values[feature] = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    key=f"input_{feature}",
                )
            else:
                min_value = float(meta.get("min", 0.0))
                max_value = float(meta.get("max", min_value + 1.0))
                default_value = float(meta.get("default", (min_value + max_value) / 2))
                if max_value <= min_value:
                    max_value = min_value + 1.0

                if is_integer_feature(feature, meta):
                    min_int = int(round(min_value))
                    max_int = int(round(max_value))
                    default_int = int(round(default_value))
                    if max_int <= min_int:
                        max_int = min_int + 1
                    default_int = min(max(default_int, min_int), max_int)

                    user_values[feature] = st.number_input(
                        label,
                        min_value=min_int,
                        max_value=max_int,
                        value=default_int,
                        step=1,
                        key=f"input_{feature}",
                    )
                else:
                    step = 1.0 if (max_value - min_value) >= 10 else 0.1
                    user_values[feature] = st.number_input(
                        label,
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value,
                        step=step,
                        key=f"input_{feature}",
                    )

    return user_values


st.set_page_config(page_title="AI Real Estate Price Predictor", layout="wide")
st.title("AI Real Estate Price Predictor")
st.caption("Prediction, what-if simulation, trend projection, and model explainability.")

st.sidebar.title("About")
st.sidebar.info(
    "This app uses a trained Random Forest regression pipeline. "
    "Train once with train.py, then deploy app.py with model.pkl."
)

if not MODEL_PATH.exists():
    st.error("model.pkl was not found.")
    st.code("python train.py --data-path data/housing.csv --model-path model.pkl")
    st.stop()

artifact = load_artifact(MODEL_PATH)
model, feature_columns, feature_schema, metrics, external_test_metrics = normalize_artifact(
    artifact
)

if not feature_columns:
    st.error("No feature columns found in model artifact.")
    st.stop()

tab_predict, tab_what_if, tab_trend, tab_explain = st.tabs(
    ["Prediction", "What-If Simulator", "Price Trend", "Feature Importance"]
)

with tab_predict:
    st.subheader("Enter Property Details")
    input_values = render_inputs(feature_columns, feature_schema)

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([input_values], columns=feature_columns)
        prediction = float(model.predict(input_df)[0])

        st.session_state["last_input_values"] = input_values
        st.session_state["last_prediction"] = prediction

        st.success(f"Estimated Price: {prediction:,.2f}")
        if metrics:
            st.write(f"Model RMSE: {metrics.get('rmse', float('nan')):,.2f}")
            st.write(f"Model R2: {metrics.get('r2', float('nan')):.4f}")
        if external_test_metrics:
            st.write("External Test RMSE:", f"{external_test_metrics.get('rmse', float('nan')):,.2f}")
            st.write("External Test R2:", f"{external_test_metrics.get('r2', float('nan')):.4f}")

with tab_what_if:
    st.subheader("Scenario Simulation")
    if "last_input_values" not in st.session_state:
        st.info("Run one prediction first to enable simulation.")
    else:
        base_input = dict(st.session_state["last_input_values"])
        numeric_features = [
            f for f in feature_columns if feature_schema.get(f, {}).get("type") != "categorical"
        ]

        if not numeric_features:
            st.info("No numeric features available for what-if analysis.")
        else:
            selected_feature = st.selectbox("Feature to vary", options=numeric_features)
            selected_meta = feature_schema.get(selected_feature, {})
            min_value = float(selected_meta.get("min", 0.0))
            max_value = float(selected_meta.get("max", min_value + 1.0))
            if max_value <= min_value:
                max_value = min_value + 1.0

            grid = np.linspace(min_value, max_value, 30)
            rows = []
            for value in grid:
                row = dict(base_input)
                row[selected_feature] = float(value)
                rows.append(row)

            sim_df = pd.DataFrame(rows, columns=feature_columns)
            sim_pred = model.predict(sim_df)
            chart_df = pd.DataFrame(
                {selected_feature: grid, "Predicted Price": sim_pred}
            ).set_index(selected_feature)

            st.line_chart(chart_df)
            st.caption("Other inputs are kept constant from the last prediction.")

with tab_trend:
    st.subheader("Projected Price Trend")
    if "last_prediction" not in st.session_state:
        st.info("Run one prediction first to create a trend projection.")
    else:
        base_price = float(st.session_state["last_prediction"])
        annual_growth = st.slider("Expected annual growth (%)", -10.0, 20.0, 4.0, 0.5)
        horizon = st.slider("Forecast horizon (years)", 1, 15, 5)

        years = list(range(horizon + 1))
        projected = [base_price * ((1 + annual_growth / 100) ** year) for year in years]
        trend_df = pd.DataFrame({"Year": years, "Projected Price": projected}).set_index("Year")
        st.line_chart(trend_df)
        st.caption("This is a scenario-based projection, not a time-series model forecast.")

with tab_explain:
    st.subheader("Model Feature Importance")
    importance_df = get_feature_importance_frame(model)

    if importance_df is None or importance_df.empty:
        st.info("Feature importance is unavailable for this model artifact.")
    else:
        top_n = min(20, len(importance_df))
        plot_df = importance_df.head(top_n).set_index("Feature")
        st.bar_chart(plot_df)
        st.dataframe(importance_df.head(top_n), use_container_width=True)
