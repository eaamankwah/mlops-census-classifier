import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """Save a trained machine learning model to disk.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    path : str
        Path to save the model.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Load a trained machine learning model from disk.

    Inputs
    ------
    path : str
        Path to the saved model.
    Returns
    -------
    model : RandomForestClassifier
        Loaded machine learning model.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def compute_slice_metrics(df, feature, cat_features, label, model, encoder, lb):
    """
    Compute model performance metrics for slices of a categorical feature.

    For each unique value of `feature`, the function filters the dataframe to
    that slice, processes it, runs inference, and computes precision/recall/F1.

    Inputs
    ------
    df : pd.DataFrame
        Full dataset (test split recommended).
    feature : str
        Name of the categorical feature to slice on.
    cat_features : list[str]
        List of categorical feature names used during training.
    label : str
        Name of the target label column.
    model : RandomForestClassifier
        Trained model.
    encoder : OneHotEncoder
        Fitted encoder from training.
    lb : LabelBinarizer
        Fitted label binarizer from training.

    Returns
    -------
    results : list[dict]
        List of dicts with keys: feature, value, count, precision, recall, fbeta.
    """
    from starter.ml.data import process_data

    results = []
    for value in df[feature].unique():
        df_slice = df[df[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=cat_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        results.append({
            "feature": feature,
            "value": value,
            "count": len(df_slice),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fbeta": round(fbeta, 4),
        })
    return results
