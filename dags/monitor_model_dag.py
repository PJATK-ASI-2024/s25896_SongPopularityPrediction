import os
import pickle
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Paths
MODEL_PATH = "models/trained_model.joblib"
NEW_DATA_PATH = "dataset/new_data.csv"
VALIDATION_REPORT_PATH = "reports/validation_report.txt"

CRITICAL_ACCURACY_THRESHOLD = 0.8
EMAIL_RECIPIENT = "s25896@pjwstk.edu.pl" 

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

# Sprawdzanie progu jakości
def validate_quality(**kwargs):
    ti = kwargs['ti']
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')

    if accuracy < CRITICAL_ACCURACY_THRESHOLD:
        print(f"Model quality is below critical threshold! Accuracy: {accuracy}, Threshold: {CRITICAL_ACCURACY_THRESHOLD}")
        ti.xcom_push(key='send_email', value=True)
    else:
        print(f"Model quality is acceptable. Accuracy: {accuracy}")
        ti.xcom_push(key='send_email', value=False)

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

    validate_task = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality
    )

    email_task = EmailOperator(
        task_id="send_alert_email",
        to=EMAIL_RECIPIENT,
        subject="Model Quality Alert",
        html_content="""
        <h3>Model Quality Alert</h3>
        <p>The model quality has fallen below the critical threshold.</p>
        <p>Please review the validation report for more details.</p>
        <p>Accuracy: {{ task_instance.xcom_pull(task_ids='evaluate_model', key='accuracy') }}</p>
        <p>Critical Threshold: {{ params.critical_threshold }}</p>
        """,
        params={"critical_threshold": CRITICAL_ACCURACY_THRESHOLD},
        trigger_rule="all_done" 
    )

    # Kolejność tasków
    load_task >> evaluate_task >> validate_task >> email_task
