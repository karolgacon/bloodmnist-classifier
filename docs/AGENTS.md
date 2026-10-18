# Instrukcja dla agentów AI (Cursor / inne)

Zespół: **Karol, Marcin, Mateusz, Julka**. Każda osoba może pracować z własnym agentem — utrzymujcie **spójny** stan zadań w jednym pliku.

## Źródło prawdy

| Plik | Cel |
|------|-----|
| **[PROGRESS.yaml](PROGRESS.yaml)** | Status zadań (`todo` / `in_progress` / `done` / `blocked`) — **edytuj ten plik** |
| [PLAN_IMPLEMENTACJI.md](PLAN_IMPLEMENTACJI.md) | Specyfikacja *co* i *jak* (rzadko zmieniana) |
| [ROLES.md](ROLES.md) | Kto jest O1–O4 i które katalogi |
| [data_contract.md](data_contract.md) | Zasady danych (O1, D3) |

**Nie edytuj** plików planu w `.cursor/plans/` — tylko repo.

## Jak aktualizować postęp

1. Otwórz `docs/PROGRESS.yaml`.
2. Znajdź zadanie po `id` (np. `O2-03`) lub `owner` (`O1`…`O4`).
3. Zmień `status` i opcjonalnie `notes` / listę `artifacts`.
4. Po ukończeniu milestone (D3, D10, …) ustaw `milestones.D3.status: done`.
5. Przy zamknięciu punktu z checklisty oceny ustaw odpowiednie pole w `grade_checklist` na `true`.
6. Commit z wiadomością np. `chore(progress): O2-03 done — LR tuning`.

### Przykład

```yaml
- id: O2-03
  status: done
  notes: RandomizedSearchCV, f1_macro, zapis w reports/results/logistic_regression.json
```

## Przypisanie ról (jednorazowo na sesji)

W `PROGRESS.yaml` ustaw:

```yaml
role_assignment:
  O1: Mateusz   # przykład — użyjcie realnego podziału zespołu
  O2: Marcin
  O3: Julka
  O4: Karol
```

Zsynchronizuj tabelę w [ROLES.md](ROLES.md) i [README.md](../README.md).

## Zasady dla agentów

- **Test set:** nie używać do tuningu (tylko D20, inference).
- **Metryki:** format z [reports/results/metrics_schema.json](../reports/results/metrics_schema.json).
- Przed `done` na zadaniu O2/O3: plik wynikowy JSON istnieje w `reports/results/` lub model w `models/`.
- Po większej zmianie: `python -m pytest tests/` z roota repo.

## Szybki podgląd (dla człowieka)

Opcjonalnie wygeneruj tabelę z YAML (lokalnie):

```bash
python scripts/progress_summary.py
```

(Skrypt może dodać O4 — jeśli brak, czytaj PROGRESS.yaml bezpośrednio.)
