# ES678 experiment log

## Dataset verification
The uploaded archive was checked against the published Zenodo MD5 checksum: `ffedb2c24e33b826d3d0df4aa8f32c1b`. The checksum matched. The archive contained `RML2016.10a_dict_optimized.pkl` (225,292,173 bytes). Inspection found 220 keys, corresponding to 11 modulations x 20 SNR levels; every key contained 1,000 samples of shape `(1000, 2, 128)` and dtype `float32`. All loaded values were finite.

## Subset and partitions
Eight digital modulation classes; 160,000 samples. Held-out SNRs: -20, -12, -4, 4, 12, 18 dB. Remaining class/SNR cells split 70/15/15 using seed 678. Exact partition sizes: 78,400 / 16,800 / 16,800 / 48,000.

## Model configuration
- Logistic regression: flattened I/Q, StandardScaler fitted on training subset only, scikit-learn LogisticRegression with saga solver, C=1.0, max_iter=200, tol=1e-3.
- CNN: 1D conv channels 2->16->32->32, kernel sizes 7/5/3, batch normalization, ReLU, max pooling, adaptive global average pooling, 8-way linear head. Adam, learning rate 0.002, weight decay 0.0001, batch size 1024, 10 epochs. RMS normalization scalar computed on training data only. Best checkpoint selected by validation macro-F1.
- Training-size study: CNN repeated using stratified 25% and 50% subsets of the training partition; validation checkpoint selection; fixed architecture and optimizer.

## Final results
See `results/test_results.json`, `results/per_snr_metrics.csv`, `results/cnn_training_size_test.json`, and `results/project_results_summary.json`.

## Important interpretation notes
The two aggregate test partitions have different SNR compositions. Main conclusions should be drawn from per-SNR performance and from comparisons of each held-out SNR with the nearest seen SNR level(s), not only from the aggregate familiar/unseen accuracy difference. Very-low-SNR performance is poor even when SNR values are familiar. The dataset is synthetic, has known historical errata according to DeepSig, and has no explicit group/transmission IDs.

## AI-use disclosure
AI assistance was used to help plan experiments, draft and debug code, and organize analysis. The student must independently inspect the source code, verify all reported values, understand the split and metrics, and disclose AI use according to the course instructions.

## Graduate-depth analyses
- CNN training-size sensitivity: 25%, 50%, and 100% of the permitted training partition (19,600 / 39,200 / 78,400 examples), with subsets stratified by modulation x SNR cell. Checkpoints were selected using validation macro-F1. Unseen-SNR test accuracy: 53.52% / 53.62% / 57.72%, respectively.
- Nearest-SNR comparison: held-out SNR levels were compared with the closest seen level(s). Accuracy differences (held-out minus neighbor mean) were -0.34, -2.15, -1.23, -0.05, -0.18, and +0.33 percentage points for -20, -12, -4, +4, +12, and +18 dB, respectively.
- Failure audit: on the unseen-SNR test, the most frequent normalized errors were QAM16 -> QAM64 (34.47% of QAM16 examples), QAM64 -> QAM16 (25.03%), and QPSK -> 8PSK (14.52%). See `results/top_confusions_unseen_snr.csv` and the normalized confusion-matrix plot.
