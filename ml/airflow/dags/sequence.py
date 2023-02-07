from ease import evaluation, dataload

import glob
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq

import psycopg2

from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import warnings
warnings.filterwarnings("ignore")

def get_newdata():
    credential_path = 'key.json'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    q = "SELECT DISTINCT userid FROM `data.new_interaction`"
    data = client.query(q).to_dataframe()
    unique_user = data['userid'].values.tolist()
    
    connection = psycopg2.connect("host=34.64.92.204 dbname=steam user=wonsam password=dnjstkawh2 port=5443")
    cur = connection.cursor()
    cur.execute(f"SELECT id,userid, gameid, playtime_total FROM history WHERE userid not in {unique_user}")
    new_data = cur.fetchall()
    
    client.load_table_from_dataframe(new_data, data.new_interaction)


def check_score():
    train, _ = dataload()
    precision = evaluation(train)
    
    return precision

with DAG(
    dag_id = "model_train",
    description = "model_train",
    start_date=days_ago(2),
    schedule_interval="0 22 * * *",
    tags=['model']
) as dag:
    
    
    t1 = PythonOperator(
        task_id = "get_newdata",
        python_callable=get_newdata,
        depends_on_past=True,
        owner="JWS",
        retries=3,
        retry_delay=timedelta(minutes=5)
    )
    
    
    t2 = PythonOperator(
        task_id = "check_score",
        python_callable=check_score,
        depends_on_past=True,
        owner="JWS",
        retries=3,
        retry_delay=timedelta(minutes=5)
    ) 
    
    t1 >> t2