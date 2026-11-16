"""Inspect RadioML 2016.10A and create the fixed ES678 split."""
import hashlib, json, tarfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from project_config import ROOT, DATA_PATH, RESULTS_DIR, CLASSES, UNSEEN_SNRS, SEED
from data_utils import load_radioml, build_arrays, make_splits

ARCHIVE = ROOT / "data" / "raw" / "RML2016.10a.tar.bz2"
EXPECTED_ZENODO_MD5 = "ffedb2c24e33b826d3d0df4aa8f32c1b"

def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Place the downloaded RML2016.10a.tar.bz2 archive in {ARCHIVE} before running this script."
        )
    actual_md5 = md5(ARCHIVE)
    if actual_md5 != EXPECTED_ZENODO_MD5:
        raise ValueError(f"Archive checksum mismatch: expected {EXPECTED_ZENODO_MD5}, got {actual_md5}")
    if not DATA_PATH.exists():
        with tarfile.open(ARCHIVE, mode="r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith((".pkl", ".pickle", ".dat"))]
            if len(members) != 1:
                raise ValueError(f"Expected one pickle file in archive; found {[m.name for m in members]}")
            source = tar.extractfile(members[0])
            if source is None:
                raise RuntimeError("Could not read pickle member from the verified archive")
            with source, DATA_PATH.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
    data = load_radioml()
    all_classes = sorted({k[0] for k in data})
    all_snrs = sorted({int(k[1]) for k in data})
    shapes = sorted({tuple(np.asarray(v).shape) for v in data.values()})
    dtypes = sorted({str(np.asarray(v).dtype) for v in data.values()})
    counts = {f"{k[0]}|{int(k[1])}": int(np.asarray(v).shape[0]) for k, v in data.items()}
    X, y, snr = build_arrays(data)
    finite = bool(np.isfinite(X).all())
    if not finite:
        raise ValueError("Non-finite values found in I/Q data.")
    if not set(CLASSES).issubset(all_classes):
        raise ValueError("One or more selected modulation classes are missing.")
    splits = make_splits(y, snr)
    np.savez_compressed(RESULTS_DIR / "fixed_splits.npz", **splits)

    split_counts = {}
    for name, idx in splits.items():
        split_counts[name] = {
            "observations": int(len(idx)),
            "snr_levels": sorted(map(int, np.unique(snr[idx]))),
            "class_counts": {c: int(np.sum(y[idx] == c)) for c in CLASSES},
        }

    summary = {
        "dataset_file": DATA_PATH.name,
        "archive_md5": md5(ARCHIVE) if ARCHIVE.exists() else None,
        "number_of_dictionary_keys": len(data),
        "all_modulation_classes": all_classes,
        "selected_digital_classes": list(CLASSES),
        "all_snr_levels_db": all_snrs,
        "observations_all_classes": int(sum(counts.values())),
        "observations_selected_classes": int(len(X)),
        "samples_per_class_snr_pair": sorted(set(counts.values())),
        "stored_array_shapes": [list(s) for s in shapes],
        "stored_dtypes": dtypes,
        "iq_shape_after_loading": list(X.shape[1:]),
        "all_iq_values_finite": finite,
        "split_seed": SEED,
        "unseen_snr_levels_db": list(UNSEEN_SNRS),
        "split_counts": split_counts,
        "notes": [
            "Only the eight digital modulation classes are retained; three analog classes are excluded.",
            "All examples at the six preselected unseen SNR levels are reserved for the unseen-SNR test.",
            "The remaining fourteen SNR levels are split 70/15/15 within each class-SNR cell.",
            "The dataset provides no explicit transmission/group identifiers, so the split controls class and SNR leakage but cannot enforce group-level independence beyond available metadata."
        ]
    }
    (RESULTS_DIR / "dataset_inspection.json").write_text(json.dumps(summary, indent=2))

    # SNR coverage plot for the selected 8-class subset.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = [int(np.sum(snr == s)) for s in all_snrs]
    bars = ax.bar(all_snrs, vals, width=1.4)
    for s, bar in zip(all_snrs, bars):
        if s in UNSEEN_SNRS:
            bar.set_hatch("///")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Number of selected observations")
    ax.set_title("RadioML 2016.10A: observations by SNR (8 digital classes)")
    ax.set_xticks(all_snrs)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "snr_coverage.png", dpi=180)
    plt.close(fig)

    print("DATASET VERIFIED")
    print(f"All classes in file: {all_classes}")
    print(f"Selected classes: {list(CLASSES)}")
    print(f"SNR levels: {all_snrs}")
    print(f"Selected observations: {len(X):,}")
    print(f"Example shape: {X.shape[1:]}; dtype: {X.dtype}; finite: {finite}")
    print("SPLIT COUNTS")
    for name, info in split_counts.items():
        print(f"{name:18s}: {info['observations']:>7,} | SNR={info['snr_levels']}")
    print(f"Saved: {RESULTS_DIR / 'dataset_inspection.json'}")
    print(f"Saved: {RESULTS_DIR / 'fixed_splits.npz'}")
    print(f"Saved: {RESULTS_DIR / 'snr_coverage.png'}")

if __name__ == "__main__":
    main()
