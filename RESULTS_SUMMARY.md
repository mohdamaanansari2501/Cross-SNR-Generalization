# ES678 results summary

## Main comparison
The logistic regression baseline obtained 13.40% accuracy and 13.37% macro-F1 on the familiar-SNR test partition, and 13.36% accuracy and 13.31% macro-F1 on the unseen-SNR partition. With eight equally represented classes, uniform random guessing has expected accuracy 12.5%; the baseline is only slightly above this reference.

The compact 1D CNN obtained 58.52% accuracy / 58.06% macro-F1 on the familiar-SNR test and 57.72% accuracy / 57.18% macro-F1 on the unseen-SNR test. Its validation accuracy and macro-F1 were 58.39% and 57.93%; the best checkpoint was selected at epoch 10 by validation macro-F1.

## SNR-dependent behavior
The CNN's accuracy is not uniform across SNR. On the unseen SNR test, accuracy was 12.16% at -20 dB, 16.76% at -12 dB, 61.28% at -4 dB, 85.11% at 4 dB, 85.53% at 12 dB, and 85.50% at 18 dB. This suggests that very low SNR remains the dominant challenge, while performance is strong at moderate and high SNR.

To avoid confusing SNR difficulty with unseen-level generalization, each held-out level was compared with its nearest seen SNR level(s). The CNN differences were -0.34 percentage points (-20 vs -18 dB), -2.15 pp (-12 vs mean of -14/-10 dB), -1.23 pp (-4 vs mean of -6/-2 dB), -0.05 pp (4 vs mean of 2/6 dB), -0.18 pp (12 vs mean of 10/14 dB), and +0.33 pp (18 vs 16 dB). These local comparisons are descriptive rather than a causal estimate.

## Training-data sensitivity
The CNN trained on 19,600 examples (25%) reached 53.52% accuracy and 53.36% macro-F1 on unseen SNR. With 39,200 examples (50%), it reached 53.62% accuracy and 53.97% macro-F1. With all 78,400 training examples, it reached 57.72% accuracy and 57.18% macro-F1. In this run, the increase from 50% to 100% training data yielded a clearer improvement than the increase from 25% to 50%.

## Error analysis
On unseen SNR test data, the most frequent normalized confusion was QAM16 classified as QAM64 (34.47% of true QAM16 examples), followed by QAM64 classified as QAM16 (25.03%). QPSK classified as 8PSK occurred for 14.52% of true QPSK examples. These confusions are plausible given the related constellation structures, but the paper should phrase this as an observed error pattern, not a proven causal explanation.

## Interpretation cautions
The aggregate familiar-SNR and unseen-SNR test sets contain different SNR distributions. The overall accuracy difference therefore should not be presented as a pure effect of SNR holdout. Report per-SNR performance and local neighbor comparisons. The RadioML 2016.10A benchmark is synthetic and historical; DeepSig warns of known errata and recommends real over-the-air or newly generated data for strong practical claims. Also, no explicit transmission/group IDs are supplied, so this study controls sample and SNR-level leakage but cannot claim group-level independence not supported by the metadata.
