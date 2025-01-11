import os
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SERVICE_ACCOUNT_FILE = "/Users/juliachainska/Downloads/projekt-442922-58b70167860c.json"

os.environ["no_proxy"] = "*"

def download_data_from_gsheets(**kwargs):
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    client = gspread.authorize(credentials)

    # Pobieranie danych zbioru modelowego
    spreadsheet = client.open("Projekt")
    basic_sheet = spreadsheet.worksheet('Basic_train_data')
    data = basic_sheet.get_all_values()

    # Przekształcenie danych na DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    # Zapisanie danych do XCom
    kwargs['ti'].xcom_push(key='raw_data', value=df.to_dict())
    return "Data downloaded successfully."


# Funkcja do czyszczenia danych
def clean_data(**kwargs):
    # Pobranie danych
    raw_data = kwargs['ti'].xcom_pull(task_ids='download_data', key='raw_data')
    df = pd.DataFrame(raw_data)

    # Usunięcie brakujących wartości
    df.dropna(inplace=True)

    # Usunięcie duplikatów
    df.drop_duplicates(inplace=True)

    # Usunięcie kolumn 'artist', 'title', 'index', jeśli istnieją
    columns_to_remove = ['artist', 'title', 'index']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

    # Zapisanie przetworzonych danych do XCom
    kwargs['ti'].xcom_push(key='cleaned_data', value=df.to_dict())
    return "Data cleaned successfully."


# Funkcja do standaryzacji i normalizacji danych
def normalize_standardize(**kwargs):
    # Pobranie danych
    cleaned_data = kwargs['ti'].xcom_pull(task_ids='clean_data', key='cleaned_data')
    df = pd.DataFrame(cleaned_data)

    # Debugowanie
    print("Dane przed przetwarzaniem:", df.head())
    print("Typy danych przed konwersją:", df.dtypes)

    # Konwersja kolumn numerycznych
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Wybór kolumn numerycznych
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    print("Kolumny numeryczne:", numeric_columns)

    # Standaryzacja
    scaler_standard = StandardScaler()
    df_standardized = scaler_standard.fit_transform(df[numeric_columns])

    # Normalizacja
    scaler_minmax = MinMaxScaler()
    df_normalized = scaler_minmax.fit_transform(df[numeric_columns])

    # Aktualizacja DataFrame
    df[numeric_columns] = df_normalized

    # Debugowanie wyników
    print("Dane po przeskalowaniu:", df.head())

    # Zapisanie przetworzonych danych do XCom
    kwargs['ti'].xcom_push(key='normalized_data', value=df.to_dict())
    return "Data normalized and standardized successfully."


# Funkcja do zapisania przetworzonych danych w Google Sheets
def save_processed_data(**kwargs):
    # Pobranie danych z poprzedniego taska
    normalized_data = kwargs['ti'].xcom_pull(task_ids='normalize_standardize', key='normalized_data')
    df = pd.DataFrame(normalized_data)

    # Walidacja danych
    if not df.replace([float('inf'), float('-inf')], float('nan')).isnull().all().all():
        df = df.replace([float('inf'), float('-inf')], float('nan')).fillna(0)

    # Połączenie z Google Sheets
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
    client = gspread.authorize(credentials)

    # Tworzenie nowego arkusza Google Sheets
    spreadsheet = client.open('Projekt')
    sheet = spreadsheet.worksheet('Processed_data')
    sheet.update([df.columns.tolist()] + df.values.tolist())

    print(f"Dane zapisano w Google Sheets: {spreadsheet.url}")
    return f"Processed data saved to Google Sheets: {spreadsheet.url}"



# Definicja DAGa
with DAG(
    dag_id="process_data_dag",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    # Zad. 1: Pobranie danych z Google Sheets
    task_download_data = PythonOperator(
        task_id="download_data",
        python_callable=download_data_from_gsheets,
        provide_context=True,
    )

    # Zad. 2: Czyszczenie danych
    task_clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
        provide_context=True,
    )

    # Zad. 3: Standaryzacja i normalizacja danych
    task_normalize_standardize = PythonOperator(
        task_id='normalize_standardize',
        python_callable=normalize_standardize,
        provide_context=True,
    )

    # Zad. 4: Zapisanie przetworzonych danych
    task_save_processed_data = PythonOperator(
        task_id='save_processed_data',
        python_callable=save_processed_data,
        provide_context=True,
    )

    # Kolejność wykonywania zadań
    task_download_data >> task_clean_data >> task_normalize_standardize >> task_save_processed_data
