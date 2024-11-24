import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import os
import logging

output_folder = "diagrams"
os.makedirs(output_folder, exist_ok=True)

# Funkcja do wyświetlania wyników TPOT
def tpot_display_scores(tpot, top_n=3):
    try:
        # Pobranie wszystkich pipeline'ów przetestowanych przez TPOT
        pipelines = tpot.evaluated_individuals_
        unique_pipeline_types = set()

        for pipeline_name in pipelines.keys():
            pipeline_type = pipeline_name.split("(")[0]
            unique_pipeline_types.add(pipeline_type)

        # Wyliczenie maksymalnych wyników CV dla każdego typu pipeline'u
        pipeline_scores = {
            pipeline_type: max(
                pipelines[pipeline_name]["internal_cv_score"]
                for pipeline_name in pipelines
                if pipeline_name.startswith(pipeline_type)
            )
            for pipeline_type in unique_pipeline_types
        }

        top_pipelines = sorted(
            pipeline_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:top_n]

        # Wyświetlenie wyników
        print("\nNajlepsze pipeline'y:")
        logging.info("Najlepsze pipeline'y:")
        for rank, (pipeline_type, score) in enumerate(top_pipelines, start=1):
            print(f"Miejsce {rank}:")
            print(f"Rodzaj pipeline'u: {pipeline_type}")
            print(f"Wynik CV: {score:.6f}\n")
            logging.info(f"Miejsce {rank}: Rodzaj pipeline'u: {pipeline_type}, Wynik CV: {score:.6f}")

    except Exception as e:
        print(f"Wystąpił błąd podczas wyświetlania wyników: {e}")
        logging.error(f"Błąd: {e}")

# Wczytanie przygotowanych danych
data = pd.read_csv("dataset/processed_data.csv")

# Przygotowanie danych
X = pd.get_dummies(data.drop(['Popularity', 'TopList'], axis=1), drop_first=True)
y = data['TopList']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TPOT AutoML
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, log_file="tpot_log.txt")
tpot.fit(X_train, y_train)

# Wyświetlenie wyników TPOT
tpot_display_scores(tpot)

# Eksport najlepszego modelu
tpot.export('best_pipeline.py')

# Dokładność na zbiorze testowym
score = tpot.score(X_test, y_test)
print(f"Dokładność modelu na zbiorze testowym: {score}")

# Zapis wyników
with open("automl_results.txt", "w") as file:
    file.write(f"Dokładność modelu na zbiorze testowym: {score}\n")
