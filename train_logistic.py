"""Train the logistic-regression baseline; use validation only, not test sets."""
import json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
from project_config import DATA_PATH, RESULTS_DIR, CLASSES
from data_utils import load_radioml, build_arrays


def main():
    data = load_radioml()
    X, y, snr = build_arrays(data)
    splits = np.load(RESULTS_DIR / "fixed_splits.npz")
    tr, va = splits["train"], splits["validation"]
    X_flat = X.reshape(len(X), -1)
    X_train, X_val = X_flat[tr], X_flat[va]
    y_train, y_val = y[tr], y[va]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val = scaler.transform(X_val).astype(np.float32, copy=False)

    model = LogisticRegression(
        solver="saga", C=1.0, max_iter=200, tol=1e-3,
        random_state=678, verbose=0
    )
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    pred = model.predict(X_val)
    report = {
        "model": "LogisticRegression(saga, C=1.0, max_iter=200, tol=1e-3)",
        "features": "flattened I/Q (256 features)",
        "scaling": "StandardScaler fitted on training set only",
        "train_observations": int(len(tr)),
        "validation_observations": int(len(va)),
        "validation_accuracy": float(accuracy_score(y_val, pred)),
        "validation_macro_f1": float(f1_score(y_val, pred, labels=list(CLASSES), average="macro")),
        "iterations_per_class": model.n_iter_.tolist(),
        "training_seconds": float(elapsed),
        "warnings": [str(w.message) for w in caught],
        "test_sets_used_for_tuning": False,
    }
    dump({"scaler": scaler, "model": model, "classes": list(CLASSES)}, RESULTS_DIR / "logistic_baseline.joblib", compress=3)
    (RESULTS_DIR / "logistic_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
