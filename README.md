# ES678 Project: Cross-SNR Generalization of Machine Learning Models for Automatic Modulation Classification

## Project status
The dataset has been verified, the fixed split has been created, both primary models have been trained, and the two locked test partitions have been evaluated. A preplanned CNN training-size sensitivity experiment has also been completed.

## Dataset
- Source: RadioML 2016.10A, optimized Zenodo distribution: https://doi.org/10.5281/zenodo.18397070
- Download: https://zenodo.org/records/18397070
- Download the `RML2016.10a.tar.bz2` file and place it in `data/raw/`.
- `prepare_data.py` checks the archive against the published MD5 `ffedb2c24e33b826d3d0df4aa8f32c1b` before extracting the pickle file. This verification is important because the data are pickle-serialized.
- The dataset contains 11 modulation types, 20 SNR levels (-20 to +18 dB in 2 dB steps), 1,000 samples per modulation/SNR pair, and arrays of shape `(1000, 2, 128)` with `float32` values.

The analysis retains eight digital classes: BPSK, QPSK, 8PSK, PAM4, QAM16, QAM64, CPFSK, and GFSK. The three analog classes are excluded, leaving 160,000 samples.

## Fixed split
The six SNR levels `[-20, -12, -4, 4, 12, 18] dB` are withheld in their entirety for the unseen-SNR test. The remaining fourteen SNR levels are divided within each modulation/SNR cell into 70% training, 15% validation, and 15% familiar-SNR test. The random seed is 678.

| Partition | Samples |
|---|---:|
| Training | 78,400 |
| Validation | 16,800 |
| Familiar-SNR test | 16,800 |
| Unseen-SNR test | 48,000 |

The split is stored in `results/fixed_splits.npz`. The deterministic sample order is created by `data_utils.build_arrays`.

## Run order
Use Python 3.10 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python prepare_data.py
python train_logistic.py
python train_cnn.py
python evaluate_models.py
python train_cnn_fraction.py --fraction 0.25
python train_cnn_fraction.py --fraction 0.5
python evaluate_cnn_fractions.py
```

The test scripts are for final reporting only. Do not use test results to change model settings, preprocessing, training duration, or the split.

### Important interpretation
- The CNN is substantially stronger than the flattened-I/Q logistic regression baseline.
- CNN accuracy is strongly SNR-dependent: it is close to chance at the most challenging very-low-SNR levels and rises to about 85% at several moderate/high SNR levels.
- The aggregate familiar-SNR and unseen-SNR partitions have different SNR compositions. Therefore, do not interpret their aggregate accuracy difference as a pure causal effect of withholding SNR. Use per-SNR results and comparisons with neighboring seen SNR levels.
- The held-out SNRs include interior values (interpolation across SNR) and endpoints (-20 and +18 dB) outside the training range (extrapolation).


## Reproducibility
- Configuration: `project_config.py`
- Data loading and deterministic split logic: `data_utils.py`
- Dataset inspection and split creation: `prepare_data.py`
- Model scripts: `train_logistic.py`, `train_cnn.py`, `train_cnn_fraction.py`
- Locked test evaluation: `evaluate_models.py`, `evaluate_cnn_fractions.py`
- Results and diagnostic plots: `results/`

All fitted preprocessing is learned from training data only. The validation set is used for checkpoint selection. The final test partitions are not used for tuning.

## AI Usage

- ChatGPT has been used for assistance here.