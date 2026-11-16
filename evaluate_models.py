"""One-time final evaluation on the two locked test partitions."""
import json, csv
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from joblib import load
import matplotlib.pyplot as plt
from project_config import RESULTS_DIR, CLASSES
from data_utils import load_radioml, build_arrays
from train_cnn import SmallAMC


def evaluate_scores(y_true, y_pred):
    return {
        "observations": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASSES), average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=list(CLASSES), output_dict=True, zero_division=0),
    }


def save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted modulation")
    ax.set_ylabel("True modulation")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i,j] > 0.55 else "black")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / filename, dpi=180); plt.close(fig)


def main():
    torch.set_num_threads(4)
    data = load_radioml()
    X, y, snr = build_arrays(data)
    split_file = np.load(RESULTS_DIR / "fixed_splits.npz")
    label_to_id = {c: i for i, c in enumerate(CLASSES)}

    # Load the frozen models selected using training/validation only.
    log_pack = load(RESULTS_DIR / "logistic_baseline.joblib")
    log_scaler, log_model = log_pack["scaler"], log_pack["model"]
    cnn_ckpt = torch.load(RESULTS_DIR / "cnn_best.pt", map_location="cpu", weights_only=True)
    cnn = SmallAMC(len(CLASSES))
    cnn.load_state_dict(cnn_ckpt["model_state_dict"])
    cnn.eval()
    cnn_scale = float(cnn_ckpt["rms_scale"])

    result = {"test_access_note": "Both locked test partitions were evaluated after model selection; no hyperparameters were changed using test results.", "cnn_best_epoch": int(cnn_ckpt["best_epoch"]), "cnn_validation_macro_f1": float(cnn_ckpt["validation_macro_f1"]), "models": {}}
    all_per_snr = []

    for split_name in ("familiar_test", "unseen_snr_test"):
        idx = split_file[split_name]
        X_sub, y_sub, snr_sub = X[idx], y[idx], snr[idx]
        y_idx = np.asarray([label_to_id[c] for c in y_sub], dtype=np.int64)

        # Logistic regression: same 256 flattened I/Q features; scaler was fit on training only.
        xf = X_sub.reshape(len(X_sub), -1)
        pred_log = log_model.predict(log_scaler.transform(xf))
        result["models"].setdefault("logistic_regression", {})[split_name] = evaluate_scores(y_sub, pred_log)
        save_cm(y_sub, pred_log, f"Logistic regression — {split_name.replace('_',' ')}", f"cm_logistic_{split_name}.png")

        # CNN: apply the single training-set RMS scale; no fitting on test data.
        x_tensor = torch.from_numpy((X_sub.copy() / cnn_scale).astype(np.float32))
        preds = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), 2048):
                logits = cnn(x_tensor[start:start+2048])
                preds.extend(logits.argmax(1).numpy().tolist())
        pred_cnn = np.asarray([CLASSES[i] for i in preds])
        result["models"].setdefault("1d_cnn", {})[split_name] = evaluate_scores(y_sub, pred_cnn)
        save_cm(y_sub, pred_cnn, f"1D CNN — {split_name.replace('_',' ')}", f"cm_cnn_{split_name}.png")

        for level in sorted(np.unique(snr_sub)):
            mask = snr_sub == level
            all_per_snr.append({
                "split": split_name, "snr_db": int(level),
                "n": int(mask.sum()),
                "logistic_accuracy": float(accuracy_score(y_sub[mask], pred_log[mask])),
                "logistic_macro_f1": float(f1_score(y_sub[mask], pred_log[mask], labels=list(CLASSES), average="macro")),
                "cnn_accuracy": float(accuracy_score(y_sub[mask], pred_cnn[mask])),
                "cnn_macro_f1": float(f1_score(y_sub[mask], pred_cnn[mask], labels=list(CLASSES), average="macro")),
            })

    (RESULTS_DIR / "test_results.json").write_text(json.dumps(result, indent=2))
    with (RESULTS_DIR / "per_snr_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_snr[0]))
        writer.writeheader(); writer.writerows(all_per_snr)

    # Compact performance-vs-SNR figure across both predeclared test partitions.
    all_per_snr.sort(key=lambda r: r["snr_db"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([r["snr_db"] for r in all_per_snr], [r["logistic_accuracy"] for r in all_per_snr], marker="o", label="Logistic regression")
    ax.plot([r["snr_db"] for r in all_per_snr], [r["cnn_accuracy"] for r in all_per_snr], marker="s", label="1D CNN")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Test accuracy across SNR levels")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_accuracy_vs_snr.png", dpi=180); plt.close(fig)

    print("FINAL TEST RESULTS")
    for model_name, parts in result["models"].items():
        print(f"\n{model_name}")
        for part, scores in parts.items():
            print(f"  {part}: n={scores['observations']}, accuracy={scores['accuracy']:.4f}, macro-F1={scores['macro_f1']:.4f}, balanced-accuracy={scores['balanced_accuracy']:.4f}")
    print("\nSaved final metrics, per-SNR CSV, confusion matrices, and SNR performance plot.")

if __name__ == "__main__":
    main()
