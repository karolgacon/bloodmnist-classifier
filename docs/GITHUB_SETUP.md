# GitHub — utworzenie zdalnego repozytorium

Wykonajcie z katalogu **`bloodmnist-classifier/`** (tam jest już `git init`).

## 1. Konto i nowe repo na GitHub

1. Zaloguj się na [github.com](https://github.com).
2. **+** → **New repository**.
3. **Repository name:** `bloodmnist-classifier` (jak lokalnie).
4. **Public** lub **Private** — według preferencji grupy / uczelni.
5. **Nie** zaznaczaj „Add a README” / „Add .gitignore” — macie je już lokalnie.
6. Kliknij **Create repository**.

## 2. Pierwszy commit lokalnie (jeśli jeszcze nie ma commita)

PowerShell:

```powershell
cd "C:\Users\Karol\Desktop\Studia\II Stopień\Sem2\UMwDM\bloodmnist-classifier"
git status
git add .
git commit -m "chore: initial project skeleton and docs (UMwDM BloodMNIST)"
```

## 3. Podłączenie remote i push

Zamień `TWOJ_USER` na login GitHub (np. organizacja zespołu):

```powershell
git branch -M main
git remote add origin https://github.com/TWOJ_USER/bloodmnist-classifier.git
git push -u origin main
```

Jeśli `remote origin` już istnieje:

```powershell
git remote set-url origin https://github.com/TWOJ_USER/bloodmnist-classifier.git
git push -u origin main
```

### Uwierzytelnianie

- **HTTPS:** Personal Access Token zamiast hasła ([GitHub → Settings → Developer settings → PAT](https://github.com/settings/tokens)).
- **SSH:** `git@github.com:TWOJ_USER/bloodmnist-classifier.git` po skonfigurowanym kluczu SSH.

## 4. Zespół (Mateusz, Julia, Marcin, Karol)

- **Private repo:** Settings → Collaborators → dodaj konta.
- **Organizacja:** dodaj repozytorium do teamu lub daj write access.

Każda osoba:

```powershell
git clone https://github.com/TWOJ_USER/bloodmnist-classifier.git
cd bloodmnist-classifier
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Przed pracą z agentem: przeczytaj [AGENTS.md](AGENTS.md) i sprawdź [PROGRESS.yaml](PROGRESS.yaml).

## 5. Po pushu — zaktualizuj postęp

W `docs/PROGRESS.yaml` ustaw zadanie **P1-08** na `status: done`.

Opcjonalnie w README dodaj link:

```markdown
**Repo:** https://github.com/TWOJ_USER/bloodmnist-classifier
```

## 6. Zalecany workflow

- Gałąź na osobę lub feature: `git checkout -b o1/eda` → PR na `main`.
- Commity ze zmianą statusu: `chore(progress): O1-02 done`.
- **Nie commituj** dużych plików: `models/*.pt`, `*.joblib` (są w `.gitignore`).
