# Data contract — BloodMNIST

**Właściciel:** O1 (Mateusz) · **Odbiorcy:** O2 (Julia), O3 (Marcin), O4 (Karol)
**Status:** gotowy do akceptacji (D3) — zmiany API tylko po uzgodnieniu z O1.

## 1. Źródło danych

- Biblioteka `medmnist` (testowane na 3.0.2), klasa `BloodMNIST`, `as_rgb=True`, rozmiar 28×28.
- Plik `~/.medmnist/bloodmnist.npz` (~35 MB) pobiera się automatycznie przy pierwszym użyciu (`download=True` domyślnie we wszystkich funkcjach).
- Splity **wyłącznie** oficjalne: `"train"` / `"val"` / `"test"` z API MedMNIST.
- **Zakaz** własnego losowego podziału, łączenia splitów i mieszania testu do CV/tuningu.

| Split | N | Użycie |
|-------|---|--------|
| train | 11 959 | trening, CV, tuning (sklearn), statystyki normalizacji |
| val | 1 712 | early stopping i tuning CNN |
| test | 3 421 | **tylko** finalna ewaluacja (D20) |

Klasy (etykiety `int64`, 0–7) — nazwy w `src/config.py` (`CLASS_NAMES_PL`, `CLASS_NAMES_EN`):

| Label | Klasa | train | val | test |
|-------|-------|------:|----:|-----:|
| 0 | Bazofil | 852 | 122 | 244 |
| 1 | Eozynofil | 2181 | 312 | 624 |
| 2 | Erytroblast | 1085 | 155 | 311 |
| 3 | Niedojrzałe granulocyty | 2026 | 290 | 579 |
| 4 | Limfocyt | 849 | 122 | 243 |
| 5 | Monocyt | 993 | 143 | 284 |
| 6 | Neutrofil | 2330 | 333 | 666 |
| 7 | Płytka krwi | 1643 | 235 | 470 |

Niezbalansowanie umiarkowane (max/min ≈ 2,7 na train); proporcje klas praktycznie identyczne we wszystkich splitach (`reports/figures/eda_split_consistency.png`) → raportujemy metryki **macro**.

## 2. scikit-learn (O1 → O2)

```python
from src.data.load import load_numpy_split

X_train, y_train = load_numpy_split("train")   # (11959, 28, 28, 3) uint8, (11959,) int64
X_val, y_val = load_numpy_split("val")
```

- Obrazy `uint8` w zakresie 0–255, kolejność kanałów RGB, układ `(N, H, W, C)`.
- Flatten `(N, 2352)`, `StandardScaler` i opcjonalne PCA (`SKLEARN_PCA_COMPONENTS` w `src/config.py`) — w `src/features/tabular.py` (O2).
- Każdy `fit` (scaler, PCA, model) **tylko na train**; val/test wyłącznie `transform`. Najprościej: wszystko w jednym `Pipeline`, który dostaje `X_train`.

## 3. PyTorch (O1 → O3, O4)

```python
from src.data.datasets import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=64,      # domyślnie CNN_BATCH_SIZE z config
    num_workers=0,      # >0 opcjonalnie na Linuksie; reprodukowalne dzięki worker_init_fn
    download=True,
    augment=True,       # False → ablacja bez augmentacji (O3-04)
    seed=42,            # kolejność batchy train; domyślnie RANDOM_SEED
)
images, labels = next(iter(train_loader))  # (B, 3, 28, 28) float32, (B,) int64
```

- Tylko `train_loader` jest tasowany (`torch.Generator` z `seed`); val/test w stałej kolejności, `drop_last=False`.
- `pin_memory` włączane automatycznie, gdy dostępna jest CUDA.
- Etykiety gotowe do `nn.CrossEntropyLoss` — bez `.squeeze()`.
- Losowość augmentacji pochodzi z globalnego RNG torcha → w skrypcie treningowym ustaw `torch.manual_seed(RANDOM_SEED)`.

Dla O4 (Grad-CAM, galeria błędów) — dostęp po indeksie:

```python
from src.data.datasets import get_datasets, get_val_test_transform

train_ds, val_ds, test_ds = get_datasets()   # augment=True dotyczy tylko train_ds
x, y = test_ds[123]                          # x: (3, 28, 28) float32 znormalizowany, y: int
raw = test_ds.imgs[123]                      # (28, 28, 3) uint8 — do wyświetlania
```

Dla Streamlit (upload obrazu PIL): `get_val_test_transform()(pil_image.convert("RGB").resize((28, 28)))`.

### Transformy

| | train (`augment=True`) | train (`augment=False`), val, test |
|---|---|---|
| `RandomRotation(15, fill=ROTATION_FILL)` | ✓ | — |
| `RandomHorizontalFlip()` + `RandomVerticalFlip()` | ✓ | — |
| `ColorJitter(0.1, 0.1, 0.1, hue=0.02)` | ✓ | — |
| `ToTensor()` (→ [0, 1]) | ✓ | ✓ |
| `Normalize(NORMALIZE_MEAN, NORMALIZE_STD)` | ✓ | ✓ |

- **Normalizacja:** statystyki per kanał liczone na **train** (nie ImageNet) — `NORMALIZE_MEAN = (0.7943, 0.6597, 0.6962)`, `NORMALIZE_STD = (0.2156, 0.2416, 0.1179)`.
- **`ROTATION_FILL = (238, 208, 190)`:** mediana koloru brzegu obrazów train; rogi po obrocie wypełniane tłem rozmazu zamiast czerni (czerń nie występuje w danych → artefakt).
- Flipy bezpieczne — komórki nie mają orientacji. Mały `hue`, bo kolor barwienia odróżnia klasy (eozynofil vs bazofil).

## 4. Artefakty O1

| Plik | Opis |
|------|------|
| `reports/results/dataset_stats.json` | mean/std RGB (train), `background_rgb_uint8`, liczności klas per split |
| `reports/figures/eda_*.png` | 5 figur EDA do raportu |

Generowanie: `python scripts/run_eda.py [--lang pl|en]`. Wartości w `src/config.py` muszą zgadzać się z JSON — pilnuje tego `tests/test_data.py` (`python -m pytest tests/`).

## 5. Test set

- Używany **wyłącznie** w skryptach finalnej ewaluacji po zamrożeniu hiperparametrów (D17+), na test day (D20).
- O2: tuning i CV **tylko na train** (`StratifiedKFold`).
- O3: early stopping i tuning CNN **tylko na val**.
- `test_loader` / `load_numpy_split("test")` nie pojawiają się w kodzie treningu ani tuningu.
