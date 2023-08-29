import mlflow

try:
    mlflow.create_experiment(name="experimenting_regression")
except mlflow.exceptions.MlflowException:
    print("The experiment alearty exists")


# with mlflow.start_run(experiment_id="652742843694052020"):
#     print("Hello from Experiment")

# -------------------------------------------------------------------

# experiment = mlflow.get_experiment(experiment_id="652742843694052020")
# print(type(experiment))
# print(experiment.name)

# -------------------------------------------------------------------

# experiment = mlflow.get_experiment_by_name(name="experimenting_regression")
# print(experiment._experiment_id)

# -------------------------------------------------------------------
# print(mlflow.search_experiments()) #list experiments에서 변경됨

# -------------------------------------------------------------------
# mlflow.delete_experiment(experiment_id="652742843694052020") # 삭제된 것이 아니라 trash로 이동함
