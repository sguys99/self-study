# https://towardsdatascience.com/manage-your-machine-learning-lifecycle-with-mlflow-part-1-a7252c859f72
# 커맨드 창에서 mlflow를 입력하면...

# Usage: mlflow [OPTIONS] COMMAND [ARGS]...
#
# Options:
#   --version  Show the version and exit.
#   --help     Show this message and exit.
#
# Commands:
#   artifacts    Upload, list, and download artifacts from an MLflow...
#   azureml      Serve models on Azure ML.
#   db           Commands for managing an MLflow tracking database.
#   deployments  Deploy MLflow models to custom targets.
#   experiments  Manage experiments.
#   gc           Permanently delete runs in the `deleted` lifecycle stage.
#   models       Deploy MLflow models locally.
#   run          Run an MLflow project from the given URI.
#   runs         Manage runs.
#   sagemaker    Serve models on SageMaker.
#   server       Run the MLflow tracking server.
#   ui           Launch the MLflow tracking UI for local viewing of run...

# 일단 심플 코드를 구현하자.

import os
from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", 5)

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", 1)
    log_metric("foo", 2)
    log_metric("foo", 3)

    # Log an artifact (output file)
    with open("output.txt", "w") as f:
        f.write("Hello world!")
    log_artifact("output.txt")

# 프로그램을 실행하면 mlruns 라는 폴더가 생성된다.
# 그다음에 mlflow ui를 입력
# http://127.0.0.1:5000에 진입
# 파라미터와 로그 메트릭이 표시된다.