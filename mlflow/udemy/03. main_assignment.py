import mlflow

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

X = iris.data

try:
    mlflow.create_experiment(name="iris_experiment")
except mlflow.exceptions.MlflowException:
    print("The experiment already exists")

with mlflow.start_run(
    run_id="4508f79c59074ed5aa86ca4dda94fd77",  # runid를 지정하면 동일부분에 덮어쓰기 가능
    experiment_id=mlflow.get_experiment_by_name("iris_experiment").experiment_id,
):
    # get the length and width
    length, width = X.shape
    mlflow.log_dict({"length": length, "width": width}, "shape.json")

    # Clustering
    optimal_inertia = 1e10
    optimal_k = 0
    for k in range(2, 11):
        kmeans = KMeans(k, random_state=0)
        kmeans = kmeans.fit(X)
        inertia = kmeans.inertia_
        mlflow.log_metric(key="inertia", value=inertia)

        if inertia < optimal_inertia:
            optimal_inertia = inertia
            optimal_k = k

    mlflow.log_param(key="optimal_k", value=optimal_k)

    optimal_kmeans = KMeans(optimal_k, random_state=0).fit(X)
    labels = optimal_kmeans.labels_

    fig, ax = plt.subplots()
    ax.hist(labels)
    mlflow.log_figure(fig, "labels_hist.png")
