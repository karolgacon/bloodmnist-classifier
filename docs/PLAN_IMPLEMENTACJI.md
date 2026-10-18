# Plan implementacji — BloodMNIST Classifier (UMwDM)

## Śledzenie postępu (dla zespołu i agentów)

| Dokument | Opis |
|----------|------|
| **[PROGRESS.yaml](PROGRESS.yaml)** | **Główna tablica zadań** — statusy `todo` / `in_progress` / `done` / `blocked` |
| [AGENTS.md](AGENTS.md) | Jak aktualizować postęp (Cursor u Marcina, Mateusza, Julki, Karola) |
| [GITHUB_SETUP.md](GITHUB_SETUP.md) | Pierwszy push na GitHub |

Podgląd w terminalu: `python scripts/progress_summary.py` (z roota repo).

**Zespół:** Karol, Marcin, Mateusz, Julka — role O1–O4 przypisz w `PROGRESS.yaml` → `role_assignment` i w tabeli poniżej.

| Rola | Osoba (przypisz) |
|------|------------------|
| O1 — Data Lead | *(do ustalenia)* |
| O2 — Classical ML | *(do ustalenia)* |
| O3 — Deep Learning | *(do ustalenia)* |
| O4 — XAI & integracja | *(do ustalenia)* |

Skrót ról: [ROLES.md](ROLES.md). Kontrakt danych: [data_contract.md](data_contract.md).

### Checklist pod ocenę (synchronizuj z PROGRESS.yaml → `grade_checklist`)

- [ ] Oficjalny split + cytowania
- [ ] Baseline + sklearn + CNN + benchmark ResNet-18
- [ ] Metryki macro + CM + słabe klasy nazwane
- [ ] Tuning opisany (CV, metryka)
- [ ] Grad-CAM + SHAP/importance
- [ ] Analiza błędów z grafiką
- [ ] Streamlit + instrukcja w README
- [ ] Reprodukowalność (seed, requirements)
- [ ] Etyka i ograniczenia

---

## Kontekst i cel oceny

**Problem diagnostyczny:** automatyczna klasyfikacja typów krwinek obwodowych na podstawie mikroskopowych obrazów barwionych (morfologia krwi / różnicowanie rozmazu — **edukacyjnie, nie klinicznie**).

**Dane:** [BloodMNIST](https://medmnist.com/) — `pip install medmnist` — 17 092 obrazy RGB **28×28**, 8 klas, oficjalny podział **11 959 / 1 712 / 3 421** (train/val/test). Źródło: Acevedo et al.; API: `INFO['bloodmnist']`.

| Label | Klasa (raport PL) |
|------|-------------------|
| 0 | Bazofil |
| 1 | Eozynofil |
| 2 | Erytroblast |
| 3 | Niedojrzałe granulocyty |
| 4 | Limfocyt |
| 5 | Monocyt |
| 6 | Neutrofil |
| 7 | Płytka krwi |

**Benchmark MedMNIST v2 (ResNet-18 @28):** macro-AUC ≈ **0,998**, accuracy ≈ **0,958** — w raporcie porównanie i omówienie odchyleń.

**Szablon raportu** (`Raport Końcowy Przykład.docx`): Wstęp → Przygotowanie danych → Modelowanie i ewaluacja → Interpretacja → Implementacja (Streamlit) → Podsumowanie → Załączniki.

**Stos:** Python 3.10+, `medmnist`, **PyTorch** + `torchvision`, **scikit-learn**, `matplotlib`/`seaborn`, opcjonalnie `optuna`, **Streamlit**, **Grad-CAM** (CNN) + **SHAP** (klasyczne ML).

**Zasady jakości:**

- Test **tylko raz** na końcu; tuning i early stopping wyłącznie na train/val.
- Stały `random_seed`; `requirements.txt` z wersjami.
- Cytowanie MedMNIST v2 (Scientific Data 2023) + Acevedo; disclaimer *dataset not intended for clinical use*; limit 28×28.

## Architektura kodu

```text
bloodmnist-classifier/
  src/data/          load.py, datasets.py, eda.py
  src/features/      tabular.py
  src/models/        resnet.py, train_cnn.py, train_sklearn.py
  src/evaluation/    metrics.py, plots.py
  src/explain/       gradcam.py, shap_sklearn.py
  src/app/           streamlit_app.py
  scripts/           run_eda, train_*, error_analysis, make_results_table
  reports/figures/   reports/results/
  models/            best_cnn.pt, best_sklearn.joblib
```

## Zakres funkcjonalny (raport + kod)

### 1. Wstęp (O4 składa, input od zespołu)

- Cel: 8 klas krwinek; znaczenie hematologiczne (skrót).
- Opis danych: modalność, N, balans klas (zweryfikować na train).
- Zakres: EDA, preprocessing, baseline + klasyczne ML + CNN, ewaluacja, XAI, Streamlit.

### 2. Przygotowanie danych (O1)

- **EDA (min. 4 figury):** rozkład klas; siatka obrazów per klasa; histogram RGB; średni prototyp per klasa; opcjonalnie % klas train/val/test.
- **Preprocessing CNN:** normalizacja mean/std z train **lub** ImageNet (jedna decyzja w raporcie); augmentacja tylko train: obrót ±15°, ostrożny flip, lekki `ColorJitter`.
- **Preprocessing ML:** flatten `3×28×28 → 2352`, `StandardScaler` fit na train; `PCA(50–100)` dla SVM (opisać trade-off w raporcie).
- Wyłącznie oficjalny split MedMNIST.

### 3. Modelowanie i ewaluacja (O2 + O3)

| Model | Rola | Uwagi |
|--------|------|--------|
| Majority / dummy | baseline | obowiązkowo |
| Logistic Regression | interpretowalny | `class_weight='balanced'` |
| SVM RBF | nieliniowy | na PCA jeśli RAM/czas |
| Random Forest | ensemble | ograniczyć `n_estimators` |
| ResNet-18 (stem 28×28) | główny | jak MedMNIST |
| Opcjonalnie Custom CNN | edukacyjnie | 3–4 conv blocks |

**CNN:** Adam, batch 64–128, 50–100 epok, early stopping (patience 7–10) na **val loss**, `models/best_cnn.pt`.

**Tuning:**

- sklearn: `RandomizedSearchCV`/`GridSearchCV` na **train**, scoring `f1_macro` lub `balanced_accuracy`, `StratifiedKFold(3–5)`.
- CNN: siatka lub Optuna (lr, weight_decay) — do 20–30 trialów na val.

**Metryki w raporcie:** accuracy, **balanced accuracy**, **macro F1**, **macro AUC OvR**, confusion matrix, classification report; uzasadnienie macro przy lekkim imbalance.

**Na wysoką ocenę:** tabela wszystkich modeli na tym samym teście + wykres macro-F1; ablacja CNN **z/bez augmentacji**.

### 4. Interpretacja i błędy (O4)

- **Grad-CAM:** 6–12 błędów + 3 trafne; komentarz (np. eozynofil vs bazofil).
- **SHAP** (PCA + najlepszy sklearn) lub permutation importance (RF); jasno: cechy ≠ morfologia bez CNN.
- **Analiza błędów:** CM z %; top pary true→pred; galeria miniatur.

### 5. Streamlit (`src/app/streamlit_app.py`)

1. **Predykcja** — galeria test / upload; top-3; CNN + opcjonalnie sklearn.
2. **Metryki** — `reports/results/*.json`, CM, leaderboard.
3. **Wyjaśnialność** — Grad-CAM.
4. **O projekcie** — disclaimer, cytowania, zespół.

### 6. Podsumowanie raportu

- Wnioski vs benchmark; ograniczenia (28×28, komórki „normalne”); rozwój (MedMNIST+ 64/128/224, pełna rozdzielczość Acevedo, ensemble).

## Podział na 4 osoby

### O1 — Data Lead

**Raport:** *Przygotowanie danych* + *Opis danych* we Wstępie.

1. `BloodMNIST`, `download=True` przy pierwszym uruchomieniu.
2. `scripts/run_eda.py` → `reports/figures/` (≥4 wizualizacje).
3. `datasets.py`: `train_transform` / `val_test_transform`.
4. `reports/results/dataset_stats.json` (mean/std z train).
5. `docs/data_contract.md` — batch PyTorch, X/y sklearn, **zakaz tuningu na test**.
6. Wykres spójności rozkładu train/val/test.

**Done:** jedna funkcja importu danych dla O2/O3; figury pod raport. **~1 tydzień.**

### O2 — Classical ML Lead

**Raport:** modele klasyczne, CV, tuning, ewaluacja w tabeli.

1. `tabular.py` — flatten + PCA.
2. Majority baseline na test.
3. Pipeline: Scaler → LR, SVC, RF.
4. RandomizedSearch/Grid per model → JSON parametrów.
5. `metrics.py` + `plots.py` (format JSON jak O3).
6. Jedna ewaluacja test → `sklearn_*.json`, `confusion_matrix_sklearn.png`.
7. `models/best_sklearn.joblib` dla O4.

**Done:** ≥3 modele w tabeli. **Zależność:** data_contract (D2–3).

### O3 — DL Lead

**Raport:** CNN, trening, benchmark MedMNIST.

1. `resnet.py` — conv 3×3, stem pod 28×28 (MedMNIST).
2. `train_cnn.py` — logi train/val (CSV/TensorBoard).
3. Early stopping → `best_cnn.pt`.
4. Tuning lr, weight_decay (Optuna opcjonalnie).
5. Ablacja bez augmentacji.
6. Opcjonalnie Custom CNN.
7. `predict_proba(tensor)` dla Streamlit.

**Done:** test acc **≥0,93** lub uzasadnione w analizie błędów; krzywe uczenia. **Zależność:** DataLoaders O1.

### O4 — XAI & Integration Lead

**Raport:** interpretacja, błędy, Streamlit, podsumowanie, załączniki.

1. `gradcam.py`, `shap_sklearn.py`.
2. `error_analysis.py` → `reports/figures/errors/`.
3. Streamlit 4 zakładki, `@st.cache_resource`.
4. `make_results_table.py` → `leaderboard.csv`.
5. Skład raportu .docx/.pdf.
6. Organizacja **test day** (checklist: brak tuningu na test).

**Done:** `streamlit run src/app/streamlit_app.py`; sekcje 4–6; tabela wszystkich modeli.

## Harmonogram (3–4 tygodnie)

| Tydzień | Wszyscy | O1 | O2 | O3 | O4 |
|---------|---------|----|----|----|-----|
| 1 | Setup (Faza 1) | EDA + pipeline | Stuby + metryki | Stuby ResNet + krótki trening | Szkic Streamlit |
| 2 | Sync 30 min | normalizacja/aug | Tuning sklearn | Tuning CNN + ablacje | Grad-CAM |
| 3 | Freeze HP | poprawki figur | final sklearn | final CNN | SHAP + app |
| 4 | **Test day** | tekst §2 | tekst §3 ML | tekst §3 DL | raport + demo |

**Checkpointy:** **D3** data_contract | **D10** metrics_schema | **D17** freeze hyperparams | **D20** test day (inference only).

## Checklist pod maksymalną ocenę

Odznaczaj w sekcji na górze tego pliku **oraz** ustaw `grade_checklist` w [PROGRESS.yaml](PROGRESS.yaml).
