# 3. Modelowanie i ewaluacja — modele klasyczne (O2)

> **Status szkicu:** wyniki dotyczą walidacji krzyżowej (train) i zbioru **walidacyjnego**. Wyniki na zbiorze testowym zostaną uzupełnione po test day (D20) — `python scripts/evaluate_sklearn_test.py` → `reports/results/sklearn_*.json`. Miejsca do uzupełnienia oznaczono **[TEST]**.

## 3.1. Reprezentacja cech

Modele scikit-learn otrzymują obraz jako wektor 28·28·3 = 2 352 wartości pikseli, standaryzowanych (`StandardScaler`) cecha po cesze (sekcja 2.3). Dla SVM z jądrem RBF wymiar redukowano dodatkowo metodą PCA. Skaler, PCA i model tworzą jeden obiekt `Pipeline` (`src/features/tabular.py`, `src/models/train_sklearn.py`), dzięki czemu podczas walidacji krzyżowej wszystkie transformacje są dopasowywane wyłącznie do foldów uczących i nie dochodzi do wycieku informacji.

Regresję logistyczną uczono na pełnych 2 352 cechach. W próbie wstępnej (bez strojenia: C = 1 dla PCA, C = 0,01 dla pełnych cech) wariant z PCA(100) osiągał na zbiorze walidacyjnym macro F1 0,76 wobec 0,81 dla pełnych cech, a ponadto tracił interpretowalność współczynników na poziomie pikseli. Las losowy również korzysta z pełnych cech — dobrze radzi sobie z wysoką wymiarowością i nie wymaga skalowania (standaryzacja nie zmienia kolejności wartości cechy, więc nie wpływa na podziały w drzewach; pozostawiono ją dla jednolitości potoku).

## 3.2. Modele

| Model | Rola | Konfiguracja |
|---|---|---|
| Klasyfikator większościowy | baseline | `DummyClassifier(strategy="most_frequent")` — zawsze przewiduje neutrofil (najliczniejsza klasa treningowa) |
| Regresja logistyczna | model liniowy, interpretowalny | regularyzacja L2, `class_weight="balanced"`, solver L-BFGS |
| SVM z jądrem RBF | model nieliniowy | PCA → `SVC(kernel="rbf", class_weight="balanced")`, prawdopodobieństwa przez skalowanie Platta |
| Las losowy | ensemble drzew | `RandomForestClassifier`, wagi klas strojone |

Ważenie klas (`class_weight="balanced"`) kompensuje umiarkowane niezbalansowanie zbioru (stosunek liczności ok. 2,7 : 1, sekcja 2.1). Predykcją wszystkich modeli jest klasa o największym prawdopodobieństwie (`predict_proba`); na tej podstawie liczone są metryki i macierze pomyłek (tę samą regułę należy stosować w aplikacji).

## 3.3. Strojenie hiperparametrów i wybór modelu

Hiperparametry dobierano metodą `RandomizedSearchCV` (12 losowań na model) z 3-krotną stratyfikowaną walidacją krzyżową (`StratifiedKFold`, tasowanie z ziarnem 42) **wyłącznie na zbiorze treningowym**. Kryterium była **macro F1** — średnia F1 po klasach, w której każda klasa waży tyle samo niezależnie od liczności. Model z najlepszą konfiguracją douczano na całym zbiorze treningowym. Model końcowy wybrano również według wyniku walidacji krzyżowej. Zbiór walidacyjny wykorzystano jedynie do wstępnego wyboru reprezentacji cech dla regresji logistycznej (sekcja 3.1) oraz jako niezależną kontrolę wyników; zbiór testowy nie był używany na żadnym etapie.

| Model | Przestrzeń przeszukiwania | Wybrane wartości |
|---|---|---|
| Regresja logistyczna | C ~ log-U(10⁻⁴, 10⁻¹) | C = 0,0064 |
| SVM RBF | liczba składowych PCA ∈ {50, 100, 150}; C ~ log-U(0,5; 50); γ ~ log-U(10⁻⁴; 5·10⁻³) | PCA = 150, C = 19,6, γ = 2,0·10⁻⁴ |
| Las losowy | n_estimators ∈ {100, 200, 300}; max_depth ∈ {None, 20, 40}; min_samples_leaf ∈ {1, 2, 4}; max_features ∈ {sqrt, log2, 0,05}; class_weight ∈ {None, balanced, balanced_subsample} | 300 drzew, max_depth = 20, min_samples_leaf = 4, max_features = 0,05 (≈118 cech), balanced_subsample |

Wszystkie kandydaty wraz z wynikami CV zapisano w `reports/results/sklearn_tuning.json`.

**Regresja logistyczna.** Wynik CV zmienia się łagodnie z parametrem C: najlepszy jest przy C ≈ 0,006 (0,780), spada przy silniejszej regularyzacji (C ≈ 10⁻⁴: 0,72) i nieznacznie przy słabszej (C ≈ 0,08: 0,77). Optimum leży więc przy umiarkowanie silnej regularyzacji, co jest typowe dla 2 352 silnie skorelowanych cech pikselowych.

**Las losowy.** Różnice między najlepszymi konfiguracjami są małe: dwie najlepsze (max_features = 0,05 z min_samples_leaf = 4 oraz max_features = sqrt z min_samples_leaf = 2) różnią się o 0,0004 macro F1, czyli znacznie mniej niż odchylenie między foldami (≈0,009). Wyniki nie wskazują więc jednoznacznie najlepszych wartości poszczególnych parametrów; wszystkie pięć najlepszych konfiguracji używało 200–300 drzew i ograniczonej głębokości (20 lub 40).

**Kompromis PCA dla SVM.** 50, 100 i 150 składowych zachowuje odpowiednio 77,2%, 87,7% i 92,0% wariancji cech. Najlepszy kandydat używał 150 składowych (CV macro F1 0,826), najlepsze kandydaty z 50 i 100 składowymi osiągały 0,805 i 0,798 (ze 100 składowymi wylosowano tylko jednego kandydata). Bardzo słabe wyniki przeszukiwania (0,06–0,66) wystąpiły wyłącznie przy dużych wartościach γ (≥ 2·10⁻³), a żaden kandydat z tak dużym γ nie osiągnął dobrego wyniku. Wszystkie te przypadki miały wprawdzie 150 składowych, ale te same 150 składowych z małym γ dało najlepszy wynik — przyczyną jest więc zbyt wąskie jądro RBF (przeuczenie), a nie liczba składowych. Ponieważ przeszukiwanie losowe zmienia wszystkie parametry jednocześnie, wpływu liczby składowych nie da się z niego dokładnie wyodrębnić; jest on rzędu 2 pp, a o jakości SVM decyduje przede wszystkim dobór γ i C. Większa liczba składowych wydłuża uczenie, ale przy 150 składowych pełne strojenie SVM trwało ok. 4 minuty, więc koszt jest akceptowalny.

## 3.4. Wyniki

Metryki: accuracy, **balanced accuracy** (średnia czułość po klasach), **macro F1** oraz **macro AUC** (one-vs-rest). Ze względu na niezbalansowanie klas za główną metrykę przyjęto macro F1 — accuracy premiuje klasy liczne, co dobrze ilustruje baseline (accuracy 19,5%, ale macro F1 0,04).

**Tabela 3.1.** Modele klasyczne — walidacja krzyżowa na zbiorze treningowym (średnia ± odchylenie standardowe z 3 foldów) i zbiór walidacyjny.

| Model | CV macro F1 | Accuracy (val) | Balanced acc. (val) | Macro F1 (val) | Macro AUC (val) | Czas (strojenie + uczenie) |
|---|---|---|---|---|---|---|
| Baseline (większościowy) | 0,041 ± 0,000 | 0,195 | 0,125 | 0,041 | 0,500 | < 1 min |
| Regresja logistyczna | 0,780 ± 0,007 | 0,829 | 0,823 | 0,809 | 0,975 | 7 min |
| Las losowy | 0,808 ± 0,009 | 0,857 | 0,831 | 0,838 | 0,981 | 18 min |
| **SVM RBF (PCA 150)** | **0,826 ± 0,004** | **0,876** | **0,854** | **0,858** | **0,986** | 4 min |

Czasy zmierzono na 8-rdzeniowym CPU.

**Tabela 3.2.** Wyniki na zbiorze testowym — **[TEST]** uzupełnić po D20 z `reports/results/sklearn_*.json`.

| Model | Accuracy | Balanced acc. | Macro F1 | Macro AUC |
|---|---|---|---|---|
| Baseline | | | | |
| Regresja logistyczna | | | | |
| Las losowy | | | | |
| SVM RBF | | | | |

Kolejność modeli jest taka sama w walidacji krzyżowej i na zbiorze walidacyjnym, a wszystkie modele przewyższają baseline o rząd wielkości. Wyniki na zbiorze walidacyjnym są o ok. 3 pp wyższe niż w CV, ponieważ model końcowy uczono na całym zbiorze treningowym (o 50% większym niż fold uczący w CV). Najlepszym modelem klasycznym jest **SVM RBF** — zapisany jako `models/best_sklearn.joblib` (przeznaczony do aplikacji Streamlit i analizy SHAP, sekcja 4).

## 3.5. Wyniki w podziale na klasy

**Tabela 3.3.** F1 dla poszczególnych klas (zbiór walidacyjny).

| Klasa | Regresja logistyczna | Las losowy | SVM RBF |
|---|---|---|---|
| Bazofil | 0,653 | 0,679 | 0,738 |
| Eozynofil | 0,936 | 0,927 | 0,969 |
| Erytroblast | 0,874 | 0,905 | 0,869 |
| Niedojrzałe granulocyty | 0,634 | 0,711 | 0,734 |
| Limfocyt | 0,794 | 0,826 | 0,811 |
| Monocyt | 0,692 | 0,745 | 0,826 |
| Neutrofil | 0,896 | 0,918 | 0,926 |
| Płytka krwi | 0,994 | 0,994 | 0,994 |

**Klasy łatwe.** Płytki krwi rozpoznawane są niemal bezbłędnie (F1 0,99) — wyróżniają się wielkością, widoczną już na średnim obrazie klasy (Rys. 2.4). Eozynofile (F1 0,93–0,97) wyróżnia charakterystyczna różowo-pomarańczowa barwa ziarnistości, czytelna nawet dla modeli opartych na surowych pikselach.

**Klasy trudne.** We wszystkich modelach najsłabiej rozpoznawane są **niedojrzałe granulocyty** i **bazofile** (F1 0,63–0,74); w regresji logistycznej i lesie losowym słabo wypadają też **monocyty**. Macierz pomyłek SVM (Rys. 3.2) pokazuje, że błędy koncentrują się wokół niedojrzałych granulocytów: najczęstsze pary pomyłek to bazofil → niedojrzały granulocyt (24 obrazy, 20% bazofili), monocyt → niedojrzały granulocyt (23; 16% monocytów), niedojrzały granulocyt → monocyt (18) i niedojrzały granulocyt → bazofil (16). Klasa niedojrzałych granulocytów jest z natury niejednorodna (promielocyty, mielocyty i metamielocyty — sekcja „Opis danych”), a jej przedstawiciele mają duże jądro i ziarnistą cytoplazmę, przez co przy rozdzielczości 28×28 są podobni zarówno do monocytów, jak i do bazofili. Modele oparte na pojedynczych pikselach nie są niezmiennicze względem obrotów — komórki są wprawdzie wyśrodkowane (Rys. 2.4), ale jądro i ziarnistości mogą mieć dowolną orientację, więc ten sam kształt daje zupełnie inne wektory cech. Tu spodziewamy się największej przewagi sieci CNN, które uczą się lokalnych wzorców niezależnie od ich położenia (sekcja 3.x, O3).

Regresja logistyczna, jako model liniowy, najsłabiej radzi sobie z trudnymi klasami (F1 0,63–0,69) — liniowa granica w przestrzeni pikseli słabo je oddziela. Nieliniowy SVM poprawia względem niej F1 monocytów o 13 pp, niedojrzałych granulocytów o 10 pp, a bazofili o 8,5 pp.

## 3.6. Podsumowanie części klasycznej

- Wszystkie modele klasyczne znacząco przewyższają baseline (macro F1 0,81–0,86 vs 0,04 na zbiorze walidacyjnym).
- Najlepszy model klasyczny: **SVM RBF na 150 składowych PCA** — macro F1 0,858, macro AUC 0,986 na zbiorze walidacyjnym; **[TEST]** na zbiorze testowym.
- Główne źródło błędów: rozróżnienie niedojrzałych granulocytów od monocytów i bazofili.
- Ograniczenie podejścia: cechy pikselowe nie uwzględniają struktury przestrzennej obrazu, a wyjaśnienia (SHAP, sekcja 4) dotyczą składowych PCA lub pikseli, a nie cech morfologicznych komórki.

## Reprodukowalność

```bash
python scripts/train_sklearn.py           # strojenie + wyniki na zbiorze walidacyjnym (~30 min, 8 rdzeni)
python scripts/evaluate_sklearn_test.py   # tylko test day (D20)
```

Ziarno losowości 42 (`src/config.py`) ustalone w podziale CV, przeszukiwaniu hiperparametrów, PCA i modelach. Liczba losowań, foldów i domyślna liczba składowych PCA — `SKLEARN_*` w `src/config.py`.

---

## Podpisy rysunków

- **Rys. 3.1.** Macro F1 modeli klasycznych na zbiorze walidacyjnym. — `reports/figures/sklearn_f1_macro_val.png`
- **Rys. 3.2.** Macierz pomyłek SVM RBF na zbiorze walidacyjnym; wartości — udział klasy prawdziwej (wiersz), w nawiasie liczba obrazów. — `reports/figures/sklearn_cm_svm_rbf_val.png` (po D20: `reports/figures/confusion_matrix_sklearn.png`, zbiór testowy)
- **Rys. 3.3–3.4** (opcjonalnie, załączniki). Macierze pomyłek regresji logistycznej i lasu losowego. — `reports/figures/sklearn_cm_logistic_regression_val.png`, `reports/figures/sklearn_cm_random_forest_val.png`
