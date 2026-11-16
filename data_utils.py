"""Loading, validation, and deterministic splitting for RadioML 2016.10A."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from project_config import CLASSES, DATA_PATH, UNSEEN_SNRS, SEED


def load_radioml(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {path}")
    # The uploaded archive was verified against the official Zenodo MD5 before extraction.
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    if not isinstance(data, dict):
        raise TypeError("Expected RadioML dictionary keyed by (modulation, SNR).")
    return data


def build_arrays(data):
    """Build deterministic arrays ordered by class then ascending SNR."""
    X_parts, y_parts, snr_parts = [], [], []
    snrs = sorted({int(k[1]) for k in data})
    for mod in CLASSES:
        for snr in snrs:
            key = (mod, snr)
            if key not in data:
                raise ValueError(f"Missing dataset cell: {key}")
            samples = np.asarray(data[key], dtype=np.float32)
            if samples.ndim != 3 or samples.shape[1:] != (2, 128):
                raise ValueError(f"Unexpected shape for {key}: {samples.shape}")
            X_parts.append(samples)
            y_parts.append(np.full(samples.shape[0], mod, dtype="U8"))
            snr_parts.append(np.full(samples.shape[0], snr, dtype=np.int16))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts)
    snr = np.concatenate(snr_parts)
    return X, y, snr


def make_splits(y, snr):
    """Split within familiar SNR cells; reserve all selected unseen SNR cells."""
    rng = np.random.default_rng(SEED)
    train, validation, familiar_test, unseen_test = [], [], [], []
    for mod in CLASSES:
        for level in sorted(np.unique(snr).tolist()):
            cell = np.flatnonzero((y == mod) & (snr == level))
            cell = rng.permutation(cell)
            if level in UNSEEN_SNRS:
                unseen_test.extend(cell.tolist())
                continue
            n = len(cell)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            train.extend(cell[:n_train].tolist())
            validation.extend(cell[n_train:n_train+n_val].tolist())
            familiar_test.extend(cell[n_train+n_val:].tolist())
    out = {
        "train": np.asarray(train, dtype=np.int64),
        "validation": np.asarray(validation, dtype=np.int64),
        "familiar_test": np.asarray(familiar_test, dtype=np.int64),
        "unseen_snr_test": np.asarray(unseen_test, dtype=np.int64),
    }
    # Guardrails: no overlap, and unseen SNRs cannot occur in train/validation.
    values = [set(v.tolist()) for v in out.values()]
    assert all(values[i].isdisjoint(values[j]) for i in range(4) for j in range(i+1, 4))
    assert set(np.unique(snr[out["train"]])).isdisjoint(UNSEEN_SNRS)
    assert set(np.unique(snr[out["validation"]])).isdisjoint(UNSEEN_SNRS)
    assert set(np.unique(snr[out["familiar_test"]])).isdisjoint(UNSEEN_SNRS)
    assert set(np.unique(snr[out["unseen_snr_test"]])) == set(UNSEEN_SNRS)
    return out
