from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_feature_schema(features: pd.DataFrame) -> dict[str, dict]:
    schema: dict[str, dict] = {}

    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.empty:
                col_min, col_max, col_default = 0.0, 1.0, 0.0
                is_integer = False
            else:
                col_min = float(clean.min())
                col_max = float(clean.max())
                col_default = float(clean.median())
                clean_as_float = clean.to_numpy(dtype=float)
                is_integer = bool(
                    np.all(np.isclose(clean_as_float, np.round(clean_as_float)))
                )

            if col_min == col_max:
                col_max = col_min + 1.0

            schema[column] = {
                "type": "numeric",
                "min": round(col_min, 4),
                "max": round(col_max, 4),
                "default": round(col_default, 4),
                "is_integer": is_integer,
            }
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            schema[column] = {
                "type": "categorical",
                "options": options[:100],
                "default": options[0],
            }

    return schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save house price model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/housing.csv",
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="price",
        help="Target column name.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model.pkl",
        help="Path to save model artifact.",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="",
        help="Optional CSV path for external test evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees in random forest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place your CSV there or pass --data-path."
        )

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.target]).copy()
    if df.empty:
        raise ValueError("No rows available after dropping missing target values.")

    x = df.drop(columns=[args.target])
    y = df[args.target]

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        num_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable input features found.")

    preprocessor = ColumnTransformer(transformers=transformers)
    regressor = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    external_test_metrics: dict[str, float | int | str] | None = None
    if args.test_data_path:
        test_data_path = Path(args.test_data_path)
        if not test_data_path.exists():
            print(
                f"Warning: external test file not found at {test_data_path}. "
                "Skipping external test evaluation."
            )
        else:
            ext_df = pd.read_csv(test_data_path)
            if args.target not in ext_df.columns:
                print(
                    f"Warning: external test data is missing target '{args.target}'. "
                    "Skipping external test evaluation."
                )
            else:
                ext_df = ext_df.dropna(subset=[args.target]).copy()
                if ext_df.empty:
                    print(
                        "Warning: no rows available in external test data after "
                        "dropping missing target values."
                    )
                else:
                    ext_x = ext_df.drop(columns=[args.target]).copy()
                    expected_columns = x.columns.tolist()

                    missing_cols = [c for c in expected_columns if c not in ext_x.columns]
                    extra_cols = [c for c in ext_x.columns if c not in expected_columns]

                    for col in missing_cols:
                        ext_x[col] = np.nan
                    if extra_cols:
                        ext_x = ext_x.drop(columns=extra_cols)

                    ext_x = ext_x[expected_columns]
                    ext_y = ext_df[args.target]
                    ext_preds = model.predict(ext_x)

                    ext_rmse = float(np.sqrt(mean_squared_error(ext_y, ext_preds)))
                    ext_r2 = float(r2_score(ext_y, ext_preds))
                    external_test_metrics = {
                        "rows": int(len(ext_df)),
                        "rmse": ext_rmse,
                        "r2": ext_r2,
                        "path": str(test_data_path),
                    }

                    if missing_cols:
                        print(
                            "External test data had missing feature columns filled with NaN: "
                            + ", ".join(missing_cols)
                        )
                    if extra_cols:
                        print(
                            "External test data had extra columns ignored: "
                            + ", ".join(extra_cols)
                        )

    artifact = {
        "model": model,
        "feature_columns": x.columns.tolist(),
        "feature_schema": build_feature_schema(x),
        "target": args.target,
        "metrics": {"rmse": rmse, "r2": r2},
        "external_test_metrics": external_test_metrics,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print("Training complete.")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(x.columns)}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    if external_test_metrics is not None:
        print("External Test Evaluation:")
        print(f"Rows: {external_test_metrics['rows']}")
        print(f"RMSE: {external_test_metrics['rmse']:.2f}")
        print(f"R2: {external_test_metrics['r2']:.4f}")
        print(f"Source: {external_test_metrics['path']}")
    print(f"Model artifact saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
