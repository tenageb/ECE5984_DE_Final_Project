from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from batch_ingest import ingest_data
from EDA import perform_eda
from transformation import transform_data
from featureExtraction import feature_extract
from build_train_model import build_train

default_args = {
    'owner': 'airflow',
    'depends_on_past' : False,
    'start_date': datetime.now() - timedelta(days=1),
    'email' : ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries' : 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'DE_project_dag',
    default_args = default_args,
    descriptions = 'African conflict analysis',
    schedule_interval='@daily',
)
# Define tasks
ingest_etl = PythonOperator(
    task_id = 'ingest_data',
    python_callable = ingest_data,
    dag = dag,
)

eda_etl = PythonOperator(
    task_id = 'perform_eda',
    python_callable = perform_eda,
    dag = dag,
)

transform_etl = PythonOperator(
    task_id = 'transform_data',
    python_callable = transform_data,
    dag = dag,
)

feature_etl = PythonOperator(
    task_id = 'extract_features',
    python_callable = feature_extract,
    dag = dag,
)

model_etl = PythonOperator(
    task_id = 'build_train_models',
    python_callable = 'build_train',
    dag = 'dag',
)
ingest_etl >> eda_etl >> transform_etl >> feature_etl >> model_etl
