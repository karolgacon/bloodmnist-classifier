# BloodMNIST Classifier

Klasyfikacja **8 typów krwinek obwodowych** na zbiorze [BloodMNIST](https://medmnist.com/) (MedMNIST v2): EDA, modele klasyczne (scikit-learn), **ResNet-18** (PyTorch), Grad-CAM / SHAP, demo **Streamlit**, raport z przedmiotu UMwDM.

> **Uwaga:** Dataset MedMNIST nie jest przeznaczony do użycia klinicznego. Obrazy 28×28 nie odzwierciedlają pełnej diagnostyki morfologii krwi.

## Zespół

Karol, Marcin, Mateusz, Julka — role O1–O4 przypisz w [docs/PROGRESS.yaml](docs/PROGRESS.yaml).

| Rola | Osoba | Zakres |
|------|--------|--------|
| O1 — Data Lead | *(przypisz)* | EDA, pipeline danych, `data_contract` |
| O2 — Classical ML | *(przypisz)* | LR, SVM, RF, tuning CV |
| O3 — Deep Learning | *(przypisz)* | ResNet-18, trening, benchmark |
| O4 — XAI & integracja | *(przypisz)* | Grad-CAM, SHAP, Streamlit, raport |

**Postęp:** [docs/PROGRESS.yaml](docs/PROGRESS.yaml) · **Agenci AI:** [docs/AGENTS.md](docs/AGENTS.md) · **GitHub:** [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md)

Plan: [docs/PLAN_IMPLEMENTACJI.md](docs/PLAN_IMPLEMENTACJI.md) · Role: [docs/ROLES.md](docs/ROLES.md)

## Dataset

- **BloodMNIST:** 17 092 obrazy RGB 28×28, 8 klas
- Split (oficjalny): train **11 959** | val **1 712** | test **3 421**
- Instalacja: `pip install medmnist`

## Instalacja

```bash
cd bloodmnist-classifier
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Uruchamiaj skrypty i testy z **katalogu głównego** repozytorium (import pakietu `src`).

```bash
python -m pytest tests/
```

## Roadmap

Szczegóły i statusy zadań: **`python scripts/progress_summary.py`** lub [docs/PROGRESS.yaml](docs/PROGRESS.yaml).

- [x] Szkielet repozytorium (Faza 1 — większość zadań P1-*)
- [ ] Przypisanie ról O1–O4 + push GitHub (P1-07, P1-08)
- [ ] Faza 2: zadania O1-* … O4-* według PROGRESS.yaml
- [ ] Test day (D20)

## Uruchomienie (po implementacji)

```bash
python scripts/run_eda.py
python scripts/train_sklearn.py
python scripts/train_cnn.py
streamlit run src/app/streamlit_app.py
```

## Cytowania

- Yang, J. et al. MedMNIST v2. *Scientific Data*, 2023.
- Acevedo, A. et al. Dataset of microscopic peripheral blood cell images.

## Licencja

MIT — see [LICENSE](LICENSE).
