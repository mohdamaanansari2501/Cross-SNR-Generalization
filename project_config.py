"""Fixed experimental configuration for the ES678 AMC project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "RML2016.10a_dict_optimized.pkl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Eight digital modulation classes; the three analog classes are excluded.
CLASSES = ("BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64", "CPFSK", "GFSK")
UNSEEN_SNRS = (-20, -12, -4, 4, 12, 18)
SEED = 678
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
