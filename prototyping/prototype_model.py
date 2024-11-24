import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Wczytanie danych
data = pd.read_csv("dataset/processed_data.csv")

# Przygotowanie danych
X = pd.get_dummies(data.drop(['Popularity', 'TopList'], axis=1), drop_first=True)
y = data['TopList']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline wygenerowany przez TPOT
exported_pipeline = SGDClassifier(
    alpha=0.001,
    eta0=0.1,
    fit_intercept=False,
    l1_ratio=1.0,
    learning_rate="invscaling",
    loss="hinge",
    penalty="elasticnet",
    power_t=0.0
)

exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_test)

# Metryki
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Wyniki
print(f"R²: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Zapis metryk
with open("model_metrics.txt", "w") as file:
    file.write(f"R²: {r2}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write("\nRaport klasyfikacji:\n")
    file.write(classification_report(y_test, y_pred))
