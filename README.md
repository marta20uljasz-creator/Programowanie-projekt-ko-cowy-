# Programowanie-projekt-ko-cowy-
Projekt zaliczeniowy
# Analiza danych dotyczących samobójstw z wykorzystaniem metod eksploracji danych i uczenia maszynowego

## Cel projektu

Celem projektu była analiza danych dotyczących samobójstw na świecie oraz próba przewidywania wskaźnika liczby samobójstw na 100 tysięcy mieszkańców przy wykorzystaniu metod uczenia maszynowego. W projekcie wykorzystałam zbiór danych zawierający informacje o liczbie samobójstw w różnych krajach, grupach wiekowych, płciach oraz poziomie rozwoju gospodarczego państw.

Projekt obejmował cały proces analizy danych – od wczytania i poznania zbioru danych, poprzez eksploracyjną analizę danych (EDA), przygotowanie cech, budowę modeli predykcyjnych, aż do oceny ich skuteczności.

## Wczytanie oraz wstępna analiza danych

Na początku zaimportowałam niezbędne biblioteki Python, takie jak Pandas, NumPy oraz Matplotlib. Następnie wczytałam zbiór danych z pliku „master.csv” do obiektu DataFrame o nazwie „suicide”.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

suicide = pd.read_csv('master.csv')
```

Po wczytaniu danych wyświetliłam pierwsze rekordy tabeli za pomocą funkcji `head()`, aby zapoznać się ze strukturą zbioru danych oraz rodzajem przechowywanych informacji.

Kolejnym krokiem było wykorzystanie funkcji `info()`, która pozwoliła sprawdzić liczbę rekordów, nazwy kolumn, typy danych oraz ewentualne braki danych. Dzięki temu mogłam ocenić jakość zbioru i przygotować się do dalszej analizy.

Następnie przeprowadziłam analizę podstawowych statystyk opisowych przy użyciu funkcji:

```python
suicide.describe(include='all')
```

Uzyskałam informacje dotyczące średnich wartości, odchyleń standardowych, wartości minimalnych i maksymalnych dla danych liczbowych oraz rozkładu wartości dla zmiennych kategorycznych.

Dodatkowo sprawdziłam najczęściej występujące wartości w kolumnach:

* country,
* sex,
* age,
* generation.

Pozwoliło mi to lepiej zrozumieć strukturę danych oraz określić, które grupy występują najczęściej w badanym zbiorze.

## Eksploracyjna analiza danych (EDA)

Po zapoznaniu się ze strukturą danych przeszłam do eksploracyjnej analizy danych. Jej celem było znalezienie zależności między zmiennymi oraz identyfikacja potencjalnych wzorców.

### Rozkład liczby samobójstw

Pierwszym wykonanym wykresem był histogram przedstawiający rozkład zmiennej „suicides/100k pop”, która stanowi główną zmienną analizowaną w projekcie.

Do stworzenia wykresu wykorzystałam funkcję `histplot()` z biblioteki Seaborn.

Histogram pozwolił określić:

* jak rozkładają się wartości wskaźnika samobójstw,
* czy występują wartości odstające,
* czy dane mają rozkład normalny.

Dodatkowo została dodana krzywa KDE, która umożliwiła lepszą ocenę kształtu rozkładu.

### Analiza grup wiekowych

Następnie przygotowałam wykres pudełkowy (boxplot) przedstawiający poziom samobójstw w poszczególnych grupach wiekowych.

```python
sns.boxplot(
    data=suicide,
    x='age',
    y='suicides/100k pop'
)
```

Wykres ten pozwolił zidentyfikować różnice pomiędzy grupami wiekowymi oraz zauważyć, w których grupach występują największe wartości wskaźnika samobójstw.

Dzięki temu mogłam stwierdzić, że wiek jest jedną z istotnych cech wpływających na analizowane zjawisko.

### Analiza wpływu poziomu PKB

Kolejnym etapem było sprawdzenie zależności pomiędzy poziomem zamożności kraju a wskaźnikiem samobójstw.

W tym celu utworzyłam nową zmienną:

```python
suicide['gdp_group']
```

Zmienne PKB zostały podzielone na cztery grupy przy użyciu funkcji `qcut()`:

* Najbiedniejsze,
* Średnio-niskie,
* Średnio-wysokie,
* Najbogatsze.

Następnie stworzyłam wykres pudełkowy pokazujący rozkład liczby samobójstw dla każdej grupy gospodarczej.

Analiza ta pozwoliła sprawdzić, czy istnieje zależność pomiędzy poziomem rozwoju ekonomicznego państwa a liczbą samobójstw.

### Analiza korelacji

Bardzo ważnym etapem projektu było obliczenie macierzy korelacji pomiędzy zmiennymi liczbowymi.

W pierwszej kolejności wyznaczyłam współczynniki korelacji:

```python
corr_matrix = numeric_data.corr()
```

Następnie wyświetliłam korelacje względem zmiennej docelowej:

```python
corr_matrix["suicides/100k pop"].sort_values(ascending=False)
```

Pozwoliło to określić, które zmienne są najbardziej związane z poziomem samobójstw.

W celu lepszej wizualizacji przygotowałam mapę ciepła (heatmap), która przedstawiała wszystkie zależności pomiędzy zmiennymi liczbowymi. Dzięki temu mogłam szybko zauważyć zarówno dodatnie, jak i ujemne korelacje.

### Analiza wpływu płci

Dodatkowo wykonałam wykres słupkowy przedstawiający średni poziom samobójstw dla kobiet i mężczyzn.

W tym celu wykorzystałam funkcję:

```python
sns.barplot()
```

Analiza wykazała wyraźne różnice pomiędzy płciami, co sugeruje, że płeć stanowi ważny czynnik wpływający na analizowane zjawisko.

## Podział danych na zbiór treningowy i testowy

Przed rozpoczęciem budowy modeli uczenia maszynowego podzieliłam dane na zbiór treningowy oraz testowy.

Najpierw zastosowałam klasyczny podział danych:

```python
train_test_split()
```

Następnie wykorzystałam bardziej zaawansowaną metodę:

```python
StratifiedShuffleSplit
```

Stratyfikacja została wykonana względem zmiennej „age”. Dzięki temu proporcje grup wiekowych zostały zachowane zarówno w zbiorze treningowym, jak i testowym.

Takie podejście zwiększa wiarygodność późniejszej oceny modeli.

## Inżynieria cech

W celu poprawy jakości modeli utworzyłam dodatkową cechę:

```python
gdp_population_ratio
```

Cecha ta została obliczona jako iloraz:

```python
gdp_per_capita / population
```

Jej zadaniem było uchwycenie dodatkowych zależności pomiędzy wielkością populacji a poziomem PKB przypadającym na mieszkańca.

Po utworzeniu nowej zmiennej ponownie przeanalizowałam korelacje w zbiorze danych, aby sprawdzić, czy nowa cecha wnosi dodatkowe informacje.

## Przygotowanie danych do modelowania

Przed trenowaniem modeli konieczne było odpowiednie przygotowanie danych.

Najpierw oddzieliłam zmienną docelową:

```python
suicides/100k pop
```

od pozostałych cech.

Usunęłam również kolumny:

* suicides_no,
* population,

ponieważ mogły prowadzić do nadmiernego wpływu na model lub zawierały informacje częściowo pokrywające się z celem predykcji.

Następnie utworzyłam pipeline przetwarzania danych.

Dla danych liczbowych zastosowałam:

* uzupełnianie braków medianą (`SimpleImputer`),
* standaryzację (`StandardScaler`).

Dla danych kategorycznych zastosowałam:

* kodowanie One-Hot Encoding (`OneHotEncoder`).

Całość została połączona za pomocą:

```python
ColumnTransformer
```

Dzięki temu wszystkie operacje przygotowujące dane wykonywane były automatycznie i w uporządkowany sposób.

## Budowa modelu regresji liniowej

Pierwszym wykorzystanym algorytmem była regresja liniowa.

Model został wytrenowany przy użyciu klasy:

```python
LinearRegression()
```

Po nauczeniu modelu wygenerowałam prognozy dla zbioru treningowego oraz obliczyłam błąd RMSE:

```python
mean_squared_error()
```

RMSE (Root Mean Squared Error) informuje o średniej wielkości błędu predykcji i jest jedną z najczęściej stosowanych miar jakości modeli regresyjnych.

W celu uzyskania bardziej wiarygodnej oceny przeprowadziłam również walidację krzyżową (Cross Validation) z wykorzystaniem 10 podziałów danych.

Pozwoliło to ocenić stabilność modelu oraz sprawdzić, czy wyniki nie są przypadkowe.

## Budowa modelu Random Forest

Drugim zastosowanym algorytmem był Random Forest Regressor.

Model został utworzony przy użyciu:

```python
RandomForestRegressor()
```

Las losowy jest algorytmem zespołowym składającym się z wielu drzew decyzyjnych, dzięki czemu potrafi modelować bardziej złożone zależności niż regresja liniowa.

Po wytrenowaniu modelu obliczyłam wartość RMSE dla zbioru treningowego oraz przeprowadziłam walidację krzyżową.

Uzyskane wyniki pozwoliły porównać skuteczność obu modeli.

## Strojenie hiperparametrów

Aby poprawić jakość modelu Random Forest, zastosowałam metodę Grid Search.

Przetestowałam różne kombinacje parametrów:

* liczby drzew (`n_estimators`),
* maksymalnej liczby cech (`max_features`),
* maksymalnej głębokości drzewa (`max_depth`),
* minimalnej liczby próbek potrzebnych do podziału (`min_samples_split`).

W tym celu wykorzystałam klasę:

```python
GridSearchCV()
```

Algorytm automatycznie sprawdził wszystkie kombinacje parametrów i wybrał model osiągający najlepsze wyniki.

## Ocena końcowa modeli

Po znalezieniu najlepszego modelu przeprowadziłam końcową ocenę na zbiorze testowym.

Dla zbioru testowego odtworzyłam wszystkie wcześniej utworzone cechy, a następnie zastosowałam pipeline przetwarzania danych.

Wyznaczyłam prognozy dla:

* regresji liniowej,
* Random Forest.

Następnie obliczyłam wartość RMSE dla obu modeli.

Na końcu zestawiłam wyniki w tabeli porównawczej.

Dzięki temu mogłam jednoznacznie określić, który model lepiej radzi sobie z przewidywaniem liczby samobójstw na 100 tysięcy mieszkańców.

## Wnioski

Przeprowadzona analiza pozwoliła zidentyfikować zależności pomiędzy liczbą samobójstw a cechami demograficznymi i ekonomicznymi. Szczególnie istotnymi czynnikami okazały się wiek, płeć oraz wskaźniki związane z poziomem rozwoju gospodarczego.

Zastosowanie metod eksploracyjnej analizy danych umożliwiło lepsze poznanie struktury zbioru danych, natomiast wykorzystanie modeli uczenia maszynowego pozwoliło na budowę systemu przewidującego poziom samobójstw.

Porównanie regresji liniowej z algorytmem Random Forest wykazało, że bardziej zaawansowane modele zespołowe są w stanie lepiej odwzorowywać złożone zależności występujące w danych społecznych i ekonomicznych. Projekt pokazał również znaczenie odpowiedniego przygotowania danych, inżynierii cech oraz walidacji modeli dla uzyskania wiarygodnych wyników predykcyjnych.
