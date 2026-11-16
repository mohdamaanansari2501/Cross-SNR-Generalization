"""Training-size sensitivity experiment; validation only, no test-set access."""
import argparse, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from project_config import RESULTS_DIR, CLASSES, SEED
from data_utils import load_radioml, build_arrays
from train_cnn import SmallAMC


def stratified_fraction(train_idx, y, snr, fraction):
    rng = np.random.default_rng(SEED)
    chosen = []
    for cls in CLASSES:
        for level in sorted(np.unique(snr[train_idx]).tolist()):
            cell = train_idx[(y[train_idx] == cls) & (snr[train_idx] == level)]
            cell = rng.permutation(cell)
            chosen.extend(cell[:int(round(len(cell) * fraction))].tolist())
    return np.asarray(chosen, dtype=np.int64)


def run(fraction):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    data = load_radioml(); X, y, snr = build_arrays(data)
    splits = np.load(RESULTS_DIR / "fixed_splits.npz")
    train_idx = stratified_fraction(splits["train"], y, snr, fraction)
    val_idx = splits["validation"]
    class_to_id = {c:i for i,c in enumerate(CLASSES)}
    y_id = np.asarray([class_to_id[c] for c in y], dtype=np.int64)

    X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
    rms = float(np.sqrt(np.mean(X_train ** 2)))
    X_train /= rms; X_val /= rms
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_id[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_id[val_idx]))
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

    model = SmallAMC(len(CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    max_epochs, patience = 10, 3
    tag = f"{int(fraction*100):02d}"
    checkpoint_path = RESULTS_DIR / f"cnn_fraction_{tag}.pt"
    report_path = RESULTS_DIR / f"cnn_fraction_{tag}.json"

    for epoch in range(1, max_epochs+1):
        model.train(); loss_sum=0.0; n=0
        for xb,yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb); loss=loss_fn(logits,yb); loss.backward(); optimizer.step()
            loss_sum += float(loss.item())*len(yb); n += len(yb)
        model.eval(); truth=[]; pred=[]; val_loss=0.0; vn=0
        with torch.no_grad():
            for xb,yb in val_loader:
                logits=model(xb); loss=loss_fn(logits,yb)
                truth.extend(yb.numpy().tolist()); pred.extend(logits.argmax(1).numpy().tolist())
                val_loss += float(loss.item())*len(yb); vn += len(yb)
        acc=accuracy_score(truth,pred); f1=f1_score(truth,pred,average='macro')
        row={"epoch":epoch,"train_loss":loss_sum/n,"validation_loss":val_loss/vn,
             "validation_accuracy":float(acc),"validation_macro_f1":float(f1)}
        history.append(row)
        if f1 > best_f1 + 1e-5:
            best_f1,best_epoch,stale=f1,epoch,0
            torch.save({"model_state_dict":model.state_dict(),"rms_scale":rms,
                        "classes":list(CLASSES),"best_epoch":best_epoch,
                        "validation_macro_f1":best_f1,"validation_accuracy":float(acc),
                        "train_fraction":fraction,"train_count":int(len(train_idx))},checkpoint_path)
        else:
            stale += 1
        report={"train_fraction":fraction,"train_observations":int(len(train_idx)),
                "validation_observations":int(len(val_idx)),"best_epoch":best_epoch,
                "best_validation_macro_f1":float(best_f1),"history":history,
                "test_sets_used_for_tuning":False}
        report_path.write_text(json.dumps(report,indent=2))
        print(f"{int(fraction*100):02d}% | epoch {epoch:02d} | val_acc={acc:.4f} | val_macro_f1={f1:.4f} | best={best_f1:.4f}",flush=True)
        if stale >= patience:
            print("Early stopping.",flush=True); break
    print(f"Saved {checkpoint_path} and {report_path}")

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fraction',type=float,required=True,choices=[0.25,0.5])
    args=parser.parse_args(); run(args.fraction)
