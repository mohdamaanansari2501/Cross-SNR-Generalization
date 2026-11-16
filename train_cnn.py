"""Train a CPU-friendly 1D CNN; checkpoint and early-stop using validation only."""
import json, random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from project_config import RESULTS_DIR, CLASSES, SEED
from data_utils import load_radioml, build_arrays


class SmallAMC(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    data = load_radioml()
    X, y, _ = build_arrays(data)
    splits = np.load(RESULTS_DIR / "fixed_splits.npz")
    tr, va = splits["train"], splits["validation"]
    label_to_id = {c: i for i, c in enumerate(CLASSES)}
    y_id = np.asarray([label_to_id[v] for v in y], dtype=np.int64)

    X_train = X[tr].copy()
    X_val = X[va].copy()
    rms = float(np.sqrt(np.mean(X_train ** 2)))
    X_train /= rms
    X_val /= rms

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_id[tr]))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_id[va]))
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

    model = SmallAMC(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    max_epochs, patience = 10, 4
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_n += len(yb)

        model.eval()
        val_true, val_pred = [], []
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_pred.extend(logits.argmax(dim=1).numpy().tolist())
                val_true.extend(yb.numpy().tolist())
                val_loss_sum += float(loss.item()) * len(yb)
                val_n += len(yb)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        row = {"epoch": epoch, "train_loss": train_loss_sum/train_n,
               "validation_loss": val_loss_sum/val_n,
               "validation_accuracy": float(val_acc),
               "validation_macro_f1": float(val_f1)}
        history.append(row)
        print(f"Epoch {epoch:02d}/{max_epochs} | train_loss={row['train_loss']:.4f} | val_loss={row['validation_loss']:.4f} | val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}", flush=True)

        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "rms_scale": rms,
                        "classes": list(CLASSES), "best_epoch": best_epoch,
                        "validation_macro_f1": best_f1, "validation_accuracy": float(val_acc)}, RESULTS_DIR / "cnn_best.pt")
        else:
            stale += 1

        progress = {"model": "CPU-friendly SmallAMC 1D CNN", "epochs_completed": epoch,
                    "best_epoch": best_epoch, "best_validation_macro_f1": float(best_f1),
                    "test_sets_used_for_tuning": False, "history": history}
        (RESULTS_DIR / "cnn_validation_progress.json").write_text(json.dumps(progress, indent=2))
        if stale >= patience:
            print("Early stopping based on validation macro-F1.", flush=True)
            break

    elapsed = time.perf_counter() - start
    report = {
        "model": "CPU-friendly SmallAMC 1D CNN (channels 2-16-32-32; global average pooling)",
        "optimizer": "Adam(lr=0.002, weight_decay=0.0001)",
        "batch_size": 1024, "max_epochs": max_epochs, "patience": patience,
        "best_epoch": best_epoch, "best_validation_macro_f1": float(best_f1),
        "training_seconds": float(elapsed), "training_rms_scale": rms,
        "test_sets_used_for_tuning": False, "history": history,
    }
    (RESULTS_DIR / "cnn_validation.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["epoch"] for r in history], [r["validation_accuracy"] for r in history], label="Validation accuracy")
    ax.plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history], label="Validation macro-F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("1D CNN validation performance")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cnn_validation_curve.png", dpi=180); plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k != "history"}, indent=2))

if __name__ == "__main__":
    main()
