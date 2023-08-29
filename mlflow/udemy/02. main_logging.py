import mlflow
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt

try:
    mlflow.create_experiment(name="logging_experiment")
except mlflow.exceptions.MlflowException:
    print("The experiment alearty exists")

with mlflow.start_run(
    experiment_id=mlflow.get_experiment_by_name("logging_experiment").experiment_id
):
    # a parameter or parameters
    # code for determining the optimal tree depth
    # mlflow.log_param(key="optimal_tree_depth", value=5)

    # a metric or metrics
    # my_list = range(0, 10)
    # for item in my_list:
    #     mlflow.log_metric(key="neural_network_score", value=item)

    # a figure
    N = 400
    t = np.linspace(0, 2 * np.pi, N)
    r = 0.5 * np.cos(t)
    x, y = r * np.cos(t), r * np.sin(t)

    fig, ax = plt.subplots()
    ax.plot(x, y, "k")
    ax.set(aspect=1)

    mlflow.log_figure(fig, artifact_file="Logged_figure_from_mlflow.png")

# artifacts
# dictionaries
# image
