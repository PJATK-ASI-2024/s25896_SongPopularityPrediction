import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier
import pickle

os.environ["no_proxy"] = "*"

# Paths
PROCESSED_DATA_PATH = "processed_data/processed_data_cleaned.csv"
MODEL_SAVE_PATH = "models/trained_model.joblib"
REPORT_SAVE_PATH = "reports/evaluation_report.txt"

# Ładowanie danych
def load_processed_data(**kwargs):
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}")
    data = pd.read_csv(PROCESSED_DATA_PATH)
    kwargs['ti'].xcom_push(key='raw_data', value=data.to_json())

# Split data into training and testing sets
def split_data(**kwargs):
    ti = kwargs['ti']
    raw_data_json = ti.xcom_pull(task_ids='load_processed_data', key='raw_data')
    df = pd.read_json(raw_data_json)

    target_column = 'Popularity'
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")

    # Feature-target split
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Push splits to XCom
    ti.xcom_push(key='X_train', value=X_train.to_json())
    ti.xcom_push(key='X_test', value=X_test.to_json())
    ti.xcom_push(key='y_train', value=y_train.to_json())
    ti.xcom_push(key='y_test', value=y_test.to_json())

# Trenowanie modelu
def train_model(**kwargs):
    ti = kwargs['ti']
    X_train = pd.read_json(ti.xcom_pull(task_ids='split_data', key='X_train'))
    y_train_json = ti.xcom_pull(task_ids='split_data', key='y_train')

    # Konwersja y_train z JSON do pandas.Series
    y_train = pd.read_json(y_train_json, typ='series', orient='index')

    # Trening modelu TPOT na zbalansowanych danych
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, cv=3, random_state=42)
    tpot.fit(X_train, y_train)

    # Zapisanie wytrenowanego modelu
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(tpot.fitted_pipeline_, f)

    ti.xcom_push(key='model_path', value=MODEL_SAVE_PATH)

# Evaluate the model
def evaluate_model(**kwargs):
    """Evaluate the trained model and save the report."""
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='train_model', key='model_path')

    # Ładownie modelu
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # dane testowe
    X_test = pd.read_json(ti.xcom_pull(task_ids='split_data', key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(task_ids='split_data', key='y_test'), typ='series')

    # Ewaluacja
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Zapisanie raportu
    os.makedirs(os.path.dirname(REPORT_SAVE_PATH), exist_ok=True)
    with open(REPORT_SAVE_PATH, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")

# Definicja DAGa
with DAG(
    "model_training_dag",
    default_args={
        "start_date": datetime(2024, 11, 29),
        "catchup": False,
    },
    schedule_interval=None,
    tags=["ML", "training"]
) as dag:
    load_data_task = PythonOperator(
        task_id="load_processed_data",
        python_callable=load_processed_data
    )

    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    # Kolejność tasków
    load_data_task >> split_data_task >> train_model_task >> evaluate_model_task
