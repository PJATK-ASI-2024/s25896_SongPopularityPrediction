# Projekt
## Wpowadzenie
Projekt ma na celu przewidzieć czy piosenka będzie popularna, na podstawie klasyfikacji binarnej (np. popularność > 70 jako "popularna").
### Cel projektu
Stworzenie modelu, który na podstawie atrybutów muzycznych piosenki, takich jak BPM, energia, taneczność czy akustyczność, przewidzi, czy utwór osiągnie wysoką popularność.
## Źródło danych
Zbiór zawierający dane o piosenkach z lat 2000-2020, w tym ich popularność, gatunek muzyczny, atrybuty dźwiękowe oraz metadane, takie jak czas trwania utworu.
Dataset pochodzi z [Kaggle](https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset)
## Charakterystyka danych:
* Licba rekordów: 2000
* Atrybuty:
    * BPM, energia, taneczność, akustyczność, popularność.
    * Gatunek muzyczny, rok wydania, tytuł i wykonawca.

## Podział danych 


## Wnioski z raportu analizy danych
### Braki danych
* Dataset nie zawiera brakujących wartości w żadnej kolumnie, co wskazuje na wysoką jakość danych.
* Nie było potrzeby imputacji lub usuwania danych.
* Nie występują równiez zduplikowane rekordy

### Rozkład zmiennych
#### Zmienne numeryczne:
* Histogramy zmiennych numerycznych, takich jak Popularity, Energy, czy Danceability, wskazują na zróżnicowane rozkłady:
    * Popularity: Rozkład jest nieco prawostronny, z większą liczbą mniej popularnych utworów (większość wartości w zakresie 40-70).
    * Energy i Danceability: Obie zmienne są zbliżone do rozkładu normalnego, z lekkim przesunięciem w kierunku wyższych wartości.
    * Wartości odstające w zmiennych takich jak Loudness i Valence są obecne, ale nie wpływają znacząco na całość analizy.
#### Zmienne kategoryczne:
* Genre ma kilkanaście unikalnych kategorii, przy czym kilka gatunków dominuje, np. Pop i Hip-Hop.
* Artist zawiera znaczną liczbę unikalnych wartości, co sugeruje, że większość wykonawców występuje tylko raz w datasetcie.

### Korelacje między zmiennymi
* Silną korelację między zmiennymi:
    * Energy i Loudness: Współczynnik korelacji wynosi ~0.85, co sugeruje, że energetyczne utwory są głośniejsze.
    * Danceability i Valence: Współczynnik ~0.6, co może sugerować, że bardziej taneczne utwory mają pozytywny nastrój.
* Brak korelacji między zmiennymi:
    * Popularity i Tempo: Sugeruje to, że tempo utworu nie wpływa bezpośrednio na jego popularność.

### Wartości odstające 
* Zmienne takie jak Popularity i Loudness:
    * Wykresy pudełkowe wskazują na obecność wartości odstających, ale ich wpływ nie jest na tyle istotny, aby je usuwać.
    * Wartości odstające mogą reprezentować ekstremalnie popularne utwory lub nietypowe style muzyczne.