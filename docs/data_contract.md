# Data contract — BloodMNIST (szkielet, O1 uzupełnia do D3)

## Źródło danych

- Biblioteka: `medmnist`, klasa `BloodMNIST`, `as_rgb=True`
- Splits: **wyłącznie** `"train"` | `"val"` | `"test"` z API MedMNIST
- **Zakaz** własnego losowego podziału lub mieszania testu do CV/tuningu

## PyTorch (O1 → O3)

- Funkcja docelowa: `get_dataloaders(batch_size, ...)` w `src/data/datasets.py`
- Batch: tensor obrazów `(B, 3, 28, 28)` float, label `(B,)` int64
- `train_transform`: augmentacja + normalizacja
- `val_test_transform`: tylko normalizacja (te same mean/std co train)

## scikit-learn (O1 → O2)

- Funkcja docelowa: `load_numpy_split(split)` w `src/data/load.py`
- Obrazy: `(N, 28, 28, 3)` uint8 → flatten `(N, 2352)` w `src/features/tabular.py`
- `StandardScaler`: **fit tylko na train**, transform na val/test
- Opcjonalnie PCA: fit na train, `n_components` z `src/config.py`

## Artefakty

| Plik | Autor | Opis |
|------|--------|------|
| `reports/results/dataset_stats.json` | O1 | mean/std RGB z train, liczności klas per split |
| `reports/figures/eda_*` | O1 | min. 4 wykresy do raportu |

## Test set

- Używany **wyłącznie** w skryptach finalnej ewaluacji po zamrożeniu hiperparametrów (D17+).
- Osoby O2/O3: tuning i CV **tylko na train**; early stopping CNN na **val**.

---

*Status: szkielet sesji wspólnej — O1 uzupełnia szczegóły implementacji do D3.*
