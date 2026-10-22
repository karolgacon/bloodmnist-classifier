# Opis danych (do Wstępu)

W projekcie wykorzystano zbiór **BloodMNIST**, część benchmarku MedMNIST v2 [1]. Zbiór powstał z publicznej kolekcji mikroskopowych obrazów krwinek krwi obwodowej opracowanej przez Acevedo i in. [2]: 17 092 obrazy pojedynczych komórek z rozmazów barwionych metodą May-Grünwalda-Giemsy, zarejestrowane analizatorem CellaVision DM96 w Hospital Clínic w Barcelonie. Próbki pochodziły od osób bez infekcji oraz chorób hematologicznych i onkologicznych, dlatego zbiór obejmuje wyłącznie komórki o prawidłowej morfologii. W MedMNIST oryginalne obrazy (360×363 px) przycięto centralnie i przeskalowano do rozdzielczości **28×28 pikseli, 3 kanały RGB**.

Zadanie to klasyfikacja wieloklasowa do **8 klas**: bazofil, eozynofil, erytroblast, niedojrzałe granulocyty (promielocyty, mielocyty, metamielocyty), limfocyt, monocyt, neutrofil oraz płytka krwi. Korzystamy wyłącznie z **oficjalnego podziału** MedMNIST w proporcji ok. 7:1:2 — 11 959 obrazów treningowych, 1 712 walidacyjnych i 3 421 testowych. Klasy są umiarkowanie niezbalansowane: najliczniejsza klasa (neutrofile, 19,5% zbioru treningowego) jest ok. 2,7 razy liczniejsza od najmniej licznych (limfocyty i bazofile, po 7,1%).

Zgodnie z deklaracją autorów MedMNIST zbiór nie jest przeznaczony do zastosowań klinicznych; projekt ma charakter wyłącznie edukacyjny.

---

# 2. Przygotowanie danych

## 2.1. Eksploracyjna analiza danych

Analizę przeprowadzono wyłącznie na zbiorze treningowym, aby żadne decyzje dotyczące przetwarzania nie korzystały z informacji o danych testowych. Wszystkie figury i statystyki generuje skrypt `scripts/run_eda.py`; wartości liczbowe zapisano w `reports/results/dataset_stats.json`.

**Rozkład klas.** Liczności klas w zbiorze treningowym wahają się od 849 (limfocyty) do 2 330 (neutrofile) (Rys. 2.1). Cztery klasy — neutrofile, eozynofile, niedojrzałe granulocyty i płytki krwi — stanowią łącznie 68% obrazów. Niezbalansowanie nie jest skrajne, ale wystarczające, by sama dokładność (accuracy) faworyzowała klasy liczne; dlatego jako główne metryki przyjęto wartości uśrednione makro (macro F1, balanced accuracy), a w modelach klasycznych rozważono ważenie klas.

**Spójność podziałów.** Udział każdej klasy w zbiorach treningowym, walidacyjnym i testowym jest praktycznie identyczny — różnice nie przekraczają 0,1 punktu procentowego (Rys. 2.2). Oficjalny podział jest więc stratyfikowany, a wyniki na zbiorze walidacyjnym powinny dobrze przewidywać wyniki testowe pod względem składu klas.

**Wygląd obrazów.** Losowe próbki (Rys. 2.3) pokazują, że klasy różnią się przede wszystkim kształtem jądra, wielkością komórki i barwą cytoplazmy: neutrofile mają jądro segmentowane, eozynofile — wyraźnie różowo-pomarańczowe ziarnistości, erytroblasty i limfocyty — zwarte, ciemne jądro, a płytki krwi są małe w porównaniu z pozostałymi komórkami. Przy rozdzielczości 28×28 część szczegółów (np. ziarnistości bazofili, struktura chromatyny) jest jednak słabo widoczna, a pary takie jak monocyt/niedojrzały granulocyt czy bazofil/limfocyt bywają wizualnie podobne. W kilku obrazach tło rozmazu ma szaroniebieski zamiast różowego odcień, co wskazuje na zmienność barwienia lub oświetlenia między preparatami.

**Średnie obrazy klas.** Uśrednienie wszystkich obrazów treningowych danej klasy (Rys. 2.4) potwierdza, że komórka jest zawsze wyśrodkowana, a klasy różnią się systematycznie wielkością: płytki krwi tworzą mały punkt, erytroblasty i limfocyty — zwarte ciemne skupisko, a monocyty i niedojrzałe granulocyty zajmują największą część kadru. Średni obraz eozynofili ma wyraźnie różowy odcień. Sugeruje to, że już proste cechy oparte na pikselach niosą informację o klasie, co uzasadnia porównanie modeli klasycznych z CNN.

**Rozkład intensywności.** Histogramy kanałów RGB (Rys. 2.5) są dwumodalne: niskie wartości odpowiadają ciemnofioletowym jądrom, wysokie — jasnemu tłu. Średnie intensywności wynoszą 203 (R), 168 (G) i 178 (B), a odchylenia standardowe ok. 55, 62 i 30. Kanał niebieski ma zatem wyraźnie mniejszą zmienność niż pozostałe. W kanale czerwonym 15% pikseli osiąga wartość maksymalną 255 (nasycenie jasnego tła) i występuje to w 97,6% obrazów. Udział nasyconych pikseli jest podobny we wszystkich klasach (9–19%), więc nie stanowi informacji o etykiecie, którą model mógłby wykorzystać w sposób niepożądany.

## 2.2. Przetwarzanie danych dla sieci CNN

**Normalizacja.** Obrazy skalowano do zakresu [0, 1], a następnie standaryzowano każdy kanał średnią i odchyleniem standardowym obliczonymi **wyłącznie na zbiorze treningowym**: średnia (0,7943; 0,6597; 0,6962), odchylenie standardowe (0,2156; 0,2416; 0,1179). Zrezygnowano ze statystyk ImageNet, ponieważ sieć trenowana jest od podstaw, a rozkład barw rozmazów krwi (dominacja jasnego, różowego tła) znacząco odbiega od zdjęć naturalnych. Standaryzacja per kanał wyrównuje też małą zmienność kanału niebieskiego. Poprawność statystyk zweryfikowano testem: po normalizacji zbiór treningowy ma w każdym kanale średnią 0 i odchylenie standardowe 1 (z dokładnością do 10⁻³).

**Augmentacja** (tylko zbiór treningowy):

- losowy obrót o kąt z zakresu ±15°;
- losowe odbicia w poziomie i pionie (p = 0,5) — komórki na rozmazie nie mają wyróżnionej orientacji, więc odbicie nie zmienia klasy;
- delikatna zmiana jasności, kontrastu i nasycenia (±10%) oraz bardzo mała zmiana odcienia (±0,02), symulująca różnice w barwieniu między preparatami. Zakres odcienia celowo ograniczono, ponieważ barwa ziarnistości jest cechą rozróżniającą klasy (np. eozynofil vs bazofil).

Obrót kwadratowego obrazu odsłania jego narożniki — przy kącie 15° jest to ok. 10% powierzchni. Domyślne wypełnienie czernią wprowadzałoby artefakt nieobecny w danych (piksele czarne praktycznie nie występują w BloodMNIST), a po normalizacji dawało wartości rzędu −4 odchyleń standardowych i ostre krawędzie. Narożniki wypełniono więc kolorem tła RGB (238, 208, 190), wyznaczonym jako mediana pikseli brzegowych obrazów treningowych. Kolor ten jest niemal jednakowy we wszystkich klasach, więc nie przenosi informacji o etykiecie.

Zbiory walidacyjny i testowy poddawane są wyłącznie normalizacji. W celu oceny wpływu augmentacji przygotowano wariant zbioru treningowego bez augmentacji (ablacja, sekcja 3).

## 2.3. Przetwarzanie danych dla modeli klasycznych

Na potrzeby modeli scikit-learn każdy obraz spłaszczono do wektora 28·28·3 = 2 352 cech (wartości pikseli). Cechy standaryzowano (`StandardScaler`), a dla modeli kosztownych obliczeniowo (SVM z jądrem RBF) zredukowano wymiar metodą PCA. Liczba składowych stanowi kompromis między zachowaną wariancją a czasem uczenia i została dobrana w sekcji 3. Zarówno skaler, jak i PCA dopasowywano wyłącznie do zbioru treningowego, a następnie stosowano do zbiorów walidacyjnego i testowego.

## 2.4. Zasady wykorzystania podziałów

- Zbiór treningowy: uczenie modeli, walidacja krzyżowa i strojenie modeli klasycznych, statystyki normalizacji.
- Zbiór walidacyjny: wczesne zatrzymywanie (early stopping) i dobór hiperparametrów CNN.
- Zbiór testowy: jednokrotna, końcowa ocena wszystkich modeli po zamrożeniu hiperparametrów.

Stałe ziarno losowości (42) zapewnia powtarzalność losowania próbek, kolejności batchy i podziałów walidacji krzyżowej.

---

## Podpisy rysunków

- **Rys. 2.1.** Liczność klas w zbiorze treningowym BloodMNIST (n = 11 959). — `reports/figures/eda_class_distribution.png`
- **Rys. 2.2.** Udział klas w oficjalnych podziałach treningowym, walidacyjnym i testowym. — `reports/figures/eda_split_consistency.png`
- **Rys. 2.3.** Losowe obrazy (28×28 px) każdej z 8 klas ze zbioru treningowego. — `reports/figures/eda_sample_grid.png`
- **Rys. 2.4.** Średni obraz każdej klasy (średnia pikseli wszystkich obrazów treningowych danej klasy). — `reports/figures/eda_class_prototypes.png`
- **Rys. 2.5.** Rozkład intensywności pikseli w kanałach R, G i B zbioru treningowego; linia przerywana — średnia, oś pionowa ograniczona do 0,04 (wartość 255 w kanale R: 15,0% pikseli). — `reports/figures/eda_rgb_histogram.png`

## Literatura

[1] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, B. Ni, „MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification", *Scientific Data*, t. 10, nr 41, 2023. doi:10.1038/s41597-022-01721-8

[2] A. Acevedo, A. Merino, S. Alférez, Á. Molina, L. Boldú, J. Rodellar, „A dataset of microscopic peripheral blood cell images for development of automatic recognition systems", *Data in Brief*, t. 30, 105474, 2020. doi:10.1016/j.dib.2020.105474
