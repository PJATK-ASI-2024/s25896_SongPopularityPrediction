import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# Ścieżka wyjściowa na wykresy
output_folder = "diagrams"
os.makedirs(output_folder, exist_ok=True)

# Wczytanie danych
data = pd.read_csv('data/basic_Data.csv')

# Podgląd danych
print("Dane podstawowe:")
print(data.head())

# Usuwanie kolumn które nie są potrzebne
data = data.drop(columns=['Index', 'Title', 'Artist', 'Year', 'Top Genre'])

# Wybór zmiennej docelowej
data['TopList'] = (data['Popularity'] > 75).astype(int)

# Eksploracja danych
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

print("Kolumny numeryczne:", numeric_cols)
print("Kolumny kategoryczne:", categorical_cols)
print("Rozkłady zmiennych numerycznych:")
print(data.describe())
print("Rozkłady zmiennych kategorycznych:")
for col in categorical_cols:
    print(f"Rozkład dla {col}:")
    print(data[col].value_counts())

# Analiza brakujących danych
missing_values = data.isnull().sum()
print("Brakujące dane:", missing_values[missing_values > 0])

# Wizualizacje
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Brakujące dane")
plt.savefig(os.path.join(output_folder, "missing_data_heatmap.png"), dpi=300)
plt.close()

data[numeric_cols].hist(bins=30, figsize=(10, 8))
plt.suptitle("Histogramy zmiennych numerycznych")
plt.savefig(os.path.join(output_folder, "numeric_columns_histograms.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Popularity'])
plt.title("Wykres pudełkowy: Popularity")
boxplot_path = os.path.join(output_folder, "popularity_boxplot.png")
plt.savefig(boxplot_path, dpi=300)
plt.close()

# Automatyczny raport
profile = ProfileReport(data, title="Spotify data Report")
profile.to_file("raport.html")

# Zapisuje przygotowane dane do dalszej analizy
data.to_csv("dataset/processed_data.csv", index=False)
