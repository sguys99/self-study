# pip requirements 규정하기..
# https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html

import tempfile # 임시로 파일을 만들때 사용
from sklearn.datasets import load_iris
import xgboost as xgb
import mlflow

def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()#\n, \r과 같은 개행문자 단위로 읽은 것을 분리해서 리스트를 리턴

def get_pip_requirements(run_id, artifact_path, return_constraints=False):
    client = mlflow.tracking.MlflowClient()
    req_path = client.download_artifacts(run_id, f"{artifact_path}/requirements.txt")
    reqs = read_lines(req_path) # artifacts 경로 안의 requirements.txt 파일을 읽어서 리스트로 디펜던시를 저장

    if return_constraints:
        con_path = client.download_artifacts(run_id, f"{artifact_path}/constraints.txt")
        cons = read_lines(con_path)
        return set(reqs), set(cons)

    return set(reqs)

def main():