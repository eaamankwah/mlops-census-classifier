# Script to train machine learning model.
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    compute_slice_metrics,
)

# Paths (relative to project root so they work both locally and on Heroku)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(_DATA_DIR, "census.csv")
DATA_PATH = os.path.join(_DATA_DIR, "census_clean.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")
SLICE_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "slice_output.txt"
)


def clean_data():
    """
    Strip leading/trailing whitespace from all fields in census.csv
    and write the result to census_clean.csv.

    The raw file uses ', ' (comma-space) as its delimiter style, producing
    values like ' State-gov' instead of 'State-gov'. If the encoder is fit
    on un-stripped values, categories like 'Private' and ' Private' would
    be treated as distinct, corrupting predictions at inference time.

    This function is idempotent — safe to call on every training run.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Copy census.csv into starter/data/ before running training."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from all string-valued columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    df.to_csv(DATA_PATH, index=False)
    print(f"Data cleaned: {len(df):,} rows written to {DATA_PATH}")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def run_training():
    # Step 1 — clean raw data (idempotent — safe to run every time)
    clean_data()

    # Step 2 — load cleaned data and split
    data = pd.read_csv(DATA_PATH)

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # Process test data (inference mode — reuse encoder/lb)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate on test set
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Overall Test Metrics — Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {fbeta:.4f}")

    # Save model artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(model, MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    with open(LB_PATH, "wb") as f:
        pickle.dump(lb, f)
    print(f"Model saved to {MODEL_PATH}")

    # Compute and save slice metrics for all categorical features
    with open(SLICE_OUTPUT_PATH, "w") as out:
        for feature in cat_features:
            results = compute_slice_metrics(
                test, feature, cat_features, "salary", model, encoder, lb
            )
            out.write(f"\n{'='*60}\n")
            out.write(f"Slice metrics for feature: {feature}\n")
            out.write(f"{'='*60}\n")
            for r in results:
                line = (
                    f"  [{r['value']}]  n={r['count']}  "
                    f"Precision={r['precision']:.4f}  "
                    f"Recall={r['recall']:.4f}  "
                    f"F1={r['fbeta']:.4f}\n"
                )
                out.write(line)
                print(line, end="")
    print(f"\nSlice output saved to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    run_training()
