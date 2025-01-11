import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

# Ścieżki
DOCKER_IMAGE_NAME = "jchainska/model-api"
DOCKERFILE_PATH = "/Users/juliachainska/Documents/Studia/7sem/asi/6-projekt/Dockerfile"
DOCKER_BUILD_CONTEXT = "/Users/juliachainska/Documents/Studia/7sem/asi/6-projekt"

# Funkcja do budowy obrazu Dockera
def build_docker_image():
    os.system(f"docker build -t {DOCKER_IMAGE_NAME}:latest {DOCKER_BUILD_CONTEXT}")

# Publikacja obrazu Dockera
def push_docker_image():
    docker_username = os.getenv("DOCKER_USERNAME")
    docker_password = os.getenv("DOCKER_PASSWORD")
    if not docker_username or not docker_password:
        raise ValueError("DOCKER_USERNAME or DOCKER_PASSWORD not set in environment variables")

    os.system(f"echo {docker_password} | docker login -u {docker_username} --password-stdin")
    os.system(f"docker push {DOCKER_IMAGE_NAME}:latest")

# Definicja DAG-a
with DAG(
    "6_contenerysation_and_api",
    default_args={
        "catchup": False,
    },
    schedule_interval=None,
    tags=["ML", "Docker", "API"],
) as dag:
    # Task: Budowa obrazu Dockera
    build_image_task = PythonOperator(
        task_id="build_docker_image",
        python_callable=build_docker_image,
    )

    # Task: Publikacja obrazu Dockera na Docker Hub
    push_image_task = PythonOperator(
        task_id="push_docker_image",
        python_callable=push_docker_image,
    )

    # Kolejność tasków
    build_image_task >> push_image_task
