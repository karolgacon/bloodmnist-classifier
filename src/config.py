"""Central configuration for BloodMNIST Classifier."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

RANDOM_SEED = 42
NUM_CLASSES = 8

CLASS_NAMES_EN = {
    0: "basophil",
    1: "eosinophil",
    2: "erythroblast",
    3: "immature granulocytes",
    4: "lymphocyte",
    5: "monocyte",
    6: "neutrophil",
    7: "platelet",
}

CLASS_NAMES_PL = {
    0: "Bazofil",
    1: "Eozynofil",
    2: "Erytroblast",
    3: "Niedojrzałe granulocyty",
    4: "Limfocyt",
    5: "Monocyt",
    6: "Neutrofil",
    7: "Płytka krwi",
}

# Filled by O1 after EDA (see reports/results/dataset_stats.json)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

CNN_BATCH_SIZE = 64
CNN_EPOCHS = 50
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_EARLY_STOPPING_PATIENCE = 10

SKLEARN_PCA_COMPONENTS = 100
SKLEARN_CV_FOLDS = 3
SKLEARN_RANDOM_SEARCH_ITER = 12

MEDMNIST_BENCHMARK = {
    "resnet18_28": {"accuracy": 0.958, "auc_macro": 0.998},
}
