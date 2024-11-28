import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sklearn.model_selection import train_test_split

os.environ["no_proxy"] = "*"

SERVICE_ACCOUNT_FILE = "/Users/juliachainska/Downloads/projekt-442922-58b70167860c.json"

# Pobieranie danych z lokalnego pliku
def download_data(ti):
    data = pd.read_csv('/Users/juliachainska/Documents/Studia/7sem/asi/Projekt/s25896_SongPopularityPrediction/dags'
                       '/Spotify-2000.csv')
    # Zapisanie danych do XCom
    ti.xcom_push(key='raw_data', value=data.to_json())


# Podział danych
def split_data(ti):
    # Pobieranie danych z poprzedniego taska
    raw_data_json = ti.xcom_pull(task_ids="download_data", key="raw_data")
    data = pd.read_json(raw_data_json)

    # Podział danych
    basic_data, additional_data = train_test_split(data, test_size=0.3, random_state=42)

    # Logowanie do debugowania
    print("Basic data sample:\n", basic_data.head())
    print("Additional data sample:\n", additional_data.head())

    # Zapisanie podzielonych danych do XCom w formacie JSON
    ti.xcom_push(key='basic_train_data', value=basic_data.to_json())
    ti.xcom_push(key='additional_train_data', value=additional_data.to_json())


# Wysyłanie danych do Google Sheets
def upload_data(**kwargs):
    # Pobieranie danych z XCom
    basic_train_data_json = kwargs['ti'].xcom_pull(key='basic_train_data', task_ids='split_data')
    additional_train_data_json = kwargs['ti'].xcom_pull(key='additional_train_data', task_ids='split_data')

    # Logowanie do debugowania
    print("Basic train data JSON:", basic_train_data_json)
    print("Additional train data JSON:", additional_train_data_json)

    # Zamiana JSON na DataFrame
    basic_train_data = pd.read_json(basic_train_data_json)
    additional_train_data = pd.read_json(additional_train_data_json)

    # Wyślij dane do Google Sheets
    upload_data_to_google_sheets("Basic_train_data", basic_train_data)
    upload_data_to_google_sheets("Additional_train_data", additional_train_data)

def upload_data_to_google_sheets(worksheet_name, data):
    # Konfiguracja połączenia z Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE).with_scopes(scope)
    client = gspread.authorize(credentials)

    # Otwórz arkusz (lub stwórz nowy jeśli nie istnieje)
    try:
        sheet = client.open("Projekt").worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        sheet = client.open("Projekt").add_worksheet(title=worksheet_name, rows=1000, cols=20)

    # Aktualizowanie arkusza danymi
    sheet.update([data.columns.values.tolist()] + data.values.tolist())

# Definicja DAGa
with DAG(
        dag_id="split_data_dag",
        schedule_interval=None,
        catchup=False,
) as dag:
    # Zad. 1: Pobierz dataset
    task_download_data = PythonOperator(
        task_id="download_data",
        python_callable=download_data,
    )

    # Zad. 2: Podziel dane
    task_split_data = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        provide_context=True,
    )

    # Zad. 3: Wyślij dane do Google Sheets
    task_upload_data = PythonOperator(
        task_id="upload_data",
        python_callable=upload_data,
        provide_context=True,
    )

    # Kolejność tasków
    task_download_data >> task_split_data >> task_upload_data
