# Podział ról — BloodMNIST Classifier

**Zespół:** Karol, Marcin, Mateusz, Julka.

Przypisanie O1–O4 ustalcie na sesji i wpisz w [PROGRESS.yaml](PROGRESS.yaml) (`role_assignment`) oraz w tabeli poniżej.

| ID | Rola | Osoba | Główne pliki |
|----|------|--------|--------------|
| O1 | Data Lead | *(przypisz)* | `src/data/*`, `scripts/run_eda.py`, `reports/figures/` |
| O2 | Classical ML | *(przypisz)* | `src/features/tabular.py`, `src/models/train_sklearn.py`, `src/evaluation/*` |
| O3 | Deep Learning | *(przypisz)* | `src/models/resnet.py`, `src/models/train_cnn.py`, `scripts/train_cnn.py` |
| O4 | XAI & integracja | *(przypisz)* | `src/explain/*`, `src/app/streamlit_app.py`, `scripts/error_analysis.py`, `scripts/make_results_table.py` |

**Twoje zadania:** filtruj w [PROGRESS.yaml](PROGRESS.yaml) po `owner: O1` (lub O2/O3/O4).

## Checkpointy zespołowe

| Dzień | Deliverable | Status w PROGRESS.yaml |
|-------|-------------|-------------------------|
| **D3** | `docs/data_contract.md` zatwierdzony (O1) | `milestones.D3` |
| **D10** | Metryki wg `metrics_schema.json` (O2 + O3) | `milestones.D10` |
| **D17** | Zamrożone hiperparametry | `milestones.D17` |
| **D20** | **Test day** — tylko inference | `milestones.D20` |

## Zasady

1. **Nie tunować na test** — test raz na końcu (O4 organizuje test day).
2. **`RANDOM_SEED=42`** w `src/config.py` — używać wszędzie.
3. Sync ~30 min w tygodniu 2: format JSON metryk, `data_contract`.
4. Po każdej sesji z agentem: zaktualizuj [PROGRESS.yaml](PROGRESS.yaml) + commit.

Pełna specyfikacja: [PLAN_IMPLEMENTACJI.md](PLAN_IMPLEMENTACJI.md). Instrukcja dla agentów: [AGENTS.md](AGENTS.md).
