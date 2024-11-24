import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data/basic_Data.csv')

# Podgląd danych
print("Dane podstawowe:")
display(data.head())

# Wybór zmiennej docelowej: przewidywanie popularności
# Przekształca "Popularity" w zmienną binarną: 1 - popularny (>75), 0 - mniej popularny
data['TopList'] = (data['Popularity'] > 75).astype(int)

# Podstawowa eksploracja: zmienne kategoryczne i numeryczne
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("Kolumny numeryczne:", numeric_cols)
print("Kolumny kategoryczne:", categorical_cols)

# Rozkład zmiennych numerycznych
print("Rozkłady zmiennych numerycznych:")
display(data.describe())

# Rozkład zmiennych kategorycznych
print("Rozkłady zmiennych kategorycznych:")
for col in data.select_dtypes(include='object').columns:
    print(f"Rozkład dla {col}:")
    display(data[col].value_counts())

# Analiza brakujących danych
missing_values = data.isnull().sum()
print("Missing values:", missing_values[missing_values > 0])

# Wizualizacja brakujących danych
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Brakujące dane")
plt.show()

# Wizualizacje:
## Histogramy dla zmiennych numerycznych
data[numeric_cols].hist(bins=30, figsize=(10, 8))
plt.suptitle("Histogramy zmiennych numerycznych")
plt.show()

# ## Wykresy pudełkowe dla zmiennych numerycznych
# for col in numeric_cols:
#     sns.boxplot(x=data[col])
#     plt.title(f'Boxplot dla {col}')
#     plt.show()


# Wykres pudełkowy dla Popularity
sns.boxplot(x=data['Popularity'])
plt.title("Wykres pudełkowy: Popularity")
plt.show()


## Automatyczna analiza danych
profile = ProfileReport(data, title="Spotify data Report (Basic Set)")
profile.to_file("raport.html")

# Przygotowanie danych do modelu
X = data.drop(['Popularity', 'TopList'], axis=1)
y = data['TopList']

# Zakodowanie zmiennych kategorycznych
X = pd.get_dummies(X, drop_first=True)

# # Normalizacja danych numerycznych
# scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Analiza AutoML z TPOT
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Odczytaj wyniki
score = tpot.score(X_test, y_test)
print(f"Dokładność modelu na zbiorze testowym: {score}")

# Eksport najlepszego modelu do pliku
tpot.export('best_pipeline.py')


