import mlflow

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans


iris = datasets.load_iris()

X = iris.data


experiment_name = "gsc-phu-rxa-01"
run_name = "baseline(rx_a_linreg_single_step_selected_op)"

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(name=experiment_name)
else:
    experiment_id = experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id], 
                          filter_string=f"tags.mlflow.runName='{run_name}'", 
                          run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)

if runs.empty:
    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    
else:
    run_id = runs.loc[0, "run_id"]
    mlflow.start_run(run_id=run_id)
    
    

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

mlflow.end_run()
