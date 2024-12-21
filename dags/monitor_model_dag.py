import os
import pickle
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Paths
MODEL_PATH = "models/trained_model.joblib"
NEW_DATA_PATH = "dataset/new_data.csv"
VALIDATION_REPORT_PATH = "reports/validation_report.txt"

CRITICAL_ACCURACY_THRESHOLD = 0.8

os.environ["no_proxy"] = "*"

# Ładowanie modelu i danych
def load_model_and_data(**kwargs):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(NEW_DATA_PATH):
        raise FileNotFoundError(f"New data not found at {NEW_DATA_PATH}")

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load data
    data = pd.read_csv(NEW_DATA_PATH)
    kwargs['ti'].xcom_push(key='model', value=model)
    kwargs['ti'].xcom_push(key='new_data', value=data.to_json())

# Ocena jakości modelu
def evaluate_model(**kwargs):
    ti = kwargs['ti']
    model = ti.xcom_pull(task_ids='load_model_and_data', key='model')
    new_data_json = ti.xcom_pull(task_ids='load_model_and_data', key='new_data')
    data = pd.read_json(new_data_json)

    target_column = "TopList"
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")

    # Split features and target
    X_new = data.drop(target_column, axis=1)
    y_new = data[target_column]

    # Model prediction
    y_pred = model.predict(X_new)

    # Calculate metrics
    accuracy = accuracy_score(y_new, y_pred)
    precision = precision_score(y_new, y_pred, average='weighted')
    recall = recall_score(y_new, y_pred, average='weighted')

    # Log metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Save metrics to XCom for further use
    ti.xcom_push(key='accuracy', value=accuracy)
    ti.xcom_push(key='precision', value=precision)
    ti.xcom_push(key='recall', value=recall)

    # Save report
    os.makedirs(os.path.dirname(VALIDATION_REPORT_PATH), exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")

# Testy jednostkowe
def run_tests(**kwargs):
    ti = kwargs['ti']
    model = ti.xcom_pull(task_ids='load_model_and_data', key='model')

    # Test 1: Czy model przewiduje wyniki
    try:
        dummy_data = pd.DataFrame([[1, 2, 3]], columns=["feature1", "feature2", "feature3"])
        model.predict(dummy_data)
        print("Test 1 passed: Model can make predictions.")
    except Exception as e:
        print(f"Test 1 failed: {str(e)}")

    # Test 2: Czy pipeline obsługuje brak danych
    try:
        empty_data = pd.DataFrame()
        model.predict(empty_data)
        print("Test 2 failed: Model should not accept empty data.")
    except ValueError:
        print("Test 2 passed: Model correctly rejects empty data.")

# Sprawdzanie progu jakości
def validate_quality(**kwargs):
    ti = kwargs['ti']
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')

    if accuracy < CRITICAL_ACCURACY_THRESHOLD:
        print(f"Model quality is below critical threshold! Accuracy: {accuracy}, Threshold: {CRITICAL_ACCURACY_THRESHOLD}")
    else:
        print(f"Model quality is acceptable. Accuracy: {accuracy}")

# Definicja DAG-a
with DAG(
    "monitor_model_dag",
    default_args={
        "catchup": False,
    },
    schedule_interval=None,
    tags=["ML", "validation"]
) as dag:
    load_task = PythonOperator(
        task_id="load_model_and_data",
        python_callable=load_model_and_data
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    test_task = PythonOperator(
        task_id="run_tests",
        python_callable=run_tests
    )

    validate_task = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality
    )

    # Kolejność tasków
    load_task >> evaluate_task >> test_task >> validate_task
