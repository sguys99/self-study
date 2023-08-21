# https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
# his tutorial showcases how you can use MLflow end-to-end to
# 여기서는 첫번쨰로 모델 학습과 관련된 코드

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        
    train, test = train_test_split(data)
    
    X_train = train.drop(["quality"],axis = 1)
    X_test = test.drop(["quality"], axis = 1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)
        
        predicted_qualities = lr.predict(X_test)
        
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
        
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = lr.predict(X_train)
        signature = infer_signature(X_train, predictions) # 어떤 입력을 받고, 어떤 출력을 발생시키는지 정보를 기록하기 위해
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)
        
        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(lr, "model", 
                                     registered_model_name='ElasticnetWineModel', signature=signature) # uri 저장??
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature) # 로컬 저장