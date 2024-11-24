# Przewidywanie popularności piosenek na podstawie atrybutów utwóru
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
* Opis kolumn: 
    * Title: Tytuł utworu
    * Artist: Nazwa artysty
    * Top Genre: Gatunek utworu
    * Year: Rok wydania utworu
    * Beats per Minute(BPM): Tempo utworu
    * Energy: Im wyzsza wartość tym bardziej energiczny utwór
    * Danceability: Im wyzsza wartość tym łatwiej tańczyć do utwóru
    * Loudness: Poziom głośności utworu
    * Valance: Im wyzsza wartość tym bardziej pozytywny utwór
    * Length: Długość trwania utworu
    * Acoustic: Im wyzsza wartość tym bardziej akustyczny utwór
    * Speechiness: Im wyzsza wartość tym więcej tekstu zawiera piosenka
    * Popularity: Im wyzsza wartość tym jest bardziej popularny utwór

## Przygotowanie danych
Usunięte zostały kolumny takie jak: Artist, Title, Index, poniewaz były to wartości unikalne, które nie wpływają na predykcję.
Kolumny Year oraz Top Genre rowniez zostały usunięte, ze względu na to, ze nie są one przydatne w dalszych procesach a redukcja zbędnych kolumn zmniejsza wymiarowość danych, co pozwola na szybsze działanie modeli AutoML.

## Braki danych
* Dataset nie zawiera brakujących wartości w żadnej kolumnie, co wskazuje na wysoką jakość danych.
* Nie było potrzeby imputacji lub usuwania danych.
* Nie występują równiez zduplikowane rekordy

## Analiza wyników z AutoML
Do znalezienia najlepszego modelu został uzyty TPOT.
Modele rekomendowane przez TPOT:
1. **SGDClassifier**:
    * CV score: 0.887
    * Najwyższy wynik CV score, szybki i efektywny w przypadku danych liniowych
2. **LogisticRegression**:
    * Wynik CV: 0.883
3. **GradientBoostingClassifier**:
    * Wynik CV: 0.883
   
## Wnioski
* SGDClassifier uzyskał najwyższy wynik walidacji krzyżowej (CV score = 0.887) spośród wszystkich przetestowanych modeli.
* Model osiągnął dokładność 0.804 na zbiorze testowym - Jest to wysoka dokładność w porównaniu z innymi modelami, co oznacza, że model radzi sobie dobrze również na nowych danych

## Wyniki prototypu
* Dokładność modelu na zbiorze testowym: 0.8042959427207638
* R²: -0.10846560846560882
* RMSE: 0.31281308898385013
* MAE: 0.09785202863961814





