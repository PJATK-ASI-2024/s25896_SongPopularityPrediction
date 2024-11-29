import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.environ["no_proxy"] = "*"

output_folder = "visualizations"
def process_data():
    # Load data
    df = pd.read_csv('/Users/juliachainska/Documents/Studia/7sem/asi/Projekt/s25896_SongPopularityPrediction/dags'
                     '/Spotify-2000.csv')

    # Data cleaning
    df = df.drop(columns=['Index', 'Title', 'Artist', 'Year', 'Top Genre'])
    df['TopList'] = (df['Popularity'] > 75).astype(int)
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Save processed data
    os.makedirs('processed_data', exist_ok=True)
    df.to_csv('processed_data/processed_data_cleaned.csv', index=False)

    # Visualization
    os.makedirs(output_folder, exist_ok=True)
    selected_columns = df.select_dtypes(include=['float64', 'int64']).iloc[:, :5]
    sns.pairplot(selected_columns)
    plt.savefig(f'{output_folder}/pairplot.png')
    plt.close()


with DAG(
    dag_id='data_processing_dag',
    schedule_interval=None
) as dag:
    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )
