import numpy as np
import pandas as pd

import subprocess

import urllib.request
import zipfile
import xgboost as xgb


def my_dot_export(xg, num_trees, filename, title="", direction="TB"):
    """Exports a specified number of trees from an XGBoost model as a graph
    visualization in dot and png formats.

    Args:
        xg: An XGBoost model.
        num_trees: The number of tree to export.
        filename: The name of the file to save the exported visualization.
        title: The title to display on the graph visualization (optional).
        direction: The direction to lay out the graph, either 'TB' (top to
            bottom) or 'LR' (left to right) (optional).
    """
    res = xgb.to_graphviz(xg, num_trees=num_trees)
    content = f"""    node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    """
    out = res.source.replace(
        "graph [ rankdir=TB ]", f"graph [ rankdir={direction} ];\n {content}"
    )
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot
    dot_filename = filename
    with open(dot_filename, "w") as fout:
        fout.write(out)
    png_filename = dot_filename.replace(".dot", ".png")
    subprocess.run(f"dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}".split())


url = (
    "https://github.com/mattharrison/datasets/raw/master/data/" "kaggle-survey-2018.zip"
)
fname = "kaggle-survey-2018.zip"
member_name = "multipleChoiceResponses.csv"


def extract_zip(src, dst, member_name):
    """Extract a member file from a zip file and read it into a pandas
    DataFrame.

    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file
            to be read into a DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the
            member file.
    """
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode="wb") as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        raw = kag.iloc[1:]
        return raw


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Tweak the Kaggle survey data and return a new DataFrame.

    This function takes a Pandas DataFrame containing Kaggle
    survey data as input and returns a new DataFrame. The
    modifications include extracting and transforming certain
    columns, renaming columns, and selecting a subset of columns.

    Parameters
    ----------
    df_ : pd.DataFrame
        The input DataFrame containing Kaggle survey data.

    Returns
    -------
    pd.DataFrame
        The new DataFrame with the modified and selected columns.
    """
    return (
        df_.assign(
            age=df_.Q2.str.slice(0, 2).astype(int),
            education=df_.Q4.replace(
                {
                    "Master’s degree": 18,
                    "Bachelor’s degree": 16,
                    "Doctoral degree": 20,
                    "Some college/university study without earning a bachelor’s degree": 13,
                    "Professional degree": 19,
                    "I prefer not to answer": None,
                    "No formal education past high school": 12,
                }
            ),
            major=(
                df_.Q5.pipe(topn, n=3).replace(
                    {
                        "Computer science (software engineering, etc.)": "cs",
                        "Engineering (non-computer focused)": "eng",
                        "Mathematics or statistics": "stat",
                    }
                )
            ),
            years_exp=(
                df_.Q8.str.replace("+", "", regex=False)
                .str.split("-", expand=True)
                .iloc[:, 0]
                .astype(float)
            ),
            compensation=(
                df_.Q9.str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("500000", "500", regex=False)
                .str.replace(
                    "I do not wish to disclose my approximate yearly compensation",
                    "0",
                    regex=False,
                )
                .str.split("-", expand=True)
                .iloc[:, 0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
            ),
            python=df_.Q16_Part_1.fillna(0).replace("Python", 1),
            r=df_.Q16_Part_2.fillna(0).replace("R", 1),
            sql=df_.Q16_Part_3.fillna(0).replace("SQL", 1),
        )  # assign
        .rename(columns=lambda col: col.replace(" ", "_"))
        .loc[
            :,
            "Q1,Q3,age,education,major,years_exp,compensation,"
            "python,r,sql".split(","),
        ]
    )


def topn(ser, n=5, default="other"):
    """
    Replace all values in a Pandas Series that are not among
    the top `n` most frequent values with a default value.

    This function takes a Pandas Series and returns a new
    Series with the values replaced as described above. The
    top `n` most frequent values are determined using the
    `value_counts` method of the input Series.

    Parameters
    ----------
    ser : pd.Series
        The input Series.
    n : int, optional
        The number of most frequent values to keep. The
        default value is 5.
    default : str, optional
        The default value to use for values that are not among
        the top `n` most frequent values. The default value is
        'other'.

    Returns
    -------
    pd.Series
        The modified Series with the values replaced.
    """
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


from feature_engine import encoding, imputation
from sklearn import base, pipeline


class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):
    """
    A transformer for tweaking Kaggle survey data.

    This transformer takes a Pandas DataFrame containing
    Kaggle survey data as input and returns a new version of
    the DataFrame. The modifications include extracting and
    transforming certain columns, renaming columns, and
    selecting a subset of columns.

    Parameters
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable.
        If not specified, the target variable will not be set.

    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """

    def __init__(self, ycol=None):
        self.ycol = ycol

    def transform(self, X):
        return tweak_kag(X)

    def fit(self, X, y=None):
        return self


def get_rawX_y(df, y_col):
    raw = df.query(
        'Q3.isin(["United States of America", "China", "India"]) '
        'and Q6.isin(["Data Scientist", "Software Engineer"])'
    )
    return raw.drop(columns=[y_col]), raw[y_col]


kag_pl = pipeline.Pipeline(
    [
        ("tweak", TweakKagTransformer()),
        (
            "cat",
            encoding.OneHotEncoder(
                top_categories=5, drop_last=True, variables=["Q1", "Q3", "major"]
            ),
        ),
        (
            "num_inpute",
            imputation.MeanMedianImputer(
                imputation_method="median", variables=["education", "years_exp"]
            ),
        ),
    ]
)

import subprocess


def my_dot_export(xg, num_trees, filename, title="", direction="TB"):
    """Exports a specified number of trees from an XGBoost model as a graph
    visualization in dot and png formats.

    Args:
        xg: An XGBoost model.
        num_trees: The number of tree to export.
        filename: The name of the file to save the exported visualization.
        title: The title to display on the graph visualization (optional).
        direction: The direction to lay out the graph, either 'TB' (top to
            bottom) or 'LR' (left to right) (optional).
    """
    res = xgb.to_graphviz(xg, num_trees=num_trees)
    content = f"""    node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    """
    out = res.source.replace(
        "graph [ rankdir=TB ]", f"graph [ rankdir={direction} ];\n {content}"
    )
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot
    dot_filename = filename
    with open(dot_filename, "w") as fout:
        fout.write(out)
    png_filename = dot_filename.replace(".dot", ".png")
    subprocess.run(f"dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}".split())


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score

from typing import Any, Dict, Union


def hyperparameter_tuning(
    space: Dict[str, Union[float, int]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    early_stopping_rounds: int = 50,
    metric: callable = accuracy_score,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for an XGBoost classifier.

    This function takes a dictionary of hyperparameters, training
    and test data, and an optional value for early stopping rounds,
    and returns a dictionary with the loss and model resulting from
    the tuning process. The model is trained using the training
    data and evaluated on the test data. The loss is computed as
    the negative of the accuracy score.

    Parameters
    ----------
    space : Dict[str, Union[float, int]]
        A dictionary of hyperparameters for the XGBoost classifier.
    X_train : pd.DataFrame
        The training data.
    y_train : pd.Series
        The training target.
    X_test : pd.DataFrame
        The test data.
    y_test : pd.Series
        The test target.
    early_stopping_rounds : int, optional
        The number of early stopping rounds to use. The default value
        is 50.
    metric : callable
        Metric to maximize. Default is accuracy

    Returns
    -------
    Dict[str, Any]
        A dictionary with the loss and model resulting from the
        tuning process. The loss is a float, and the model is an
        XGBoost classifier.
    """
    int_vals = ["max_depth", "reg_alpha"]
    space = {k: (int(val) if k in int_vals else val) for k, val in space.items()}
    space["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=evaluation, verbose=False)

    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return {"loss": -score, "status": STATUS_OK, "model": model}


import plotly.graph_objects as go


def plot_3d_mesh(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> go.Figure:
    """
    Create a 3D mesh plot using Plotly.

    This function creates a 3D mesh plot using Plotly, with
    the `x_col`, `y_col`, and `z_col` columns of the `df`
    DataFrame as the x, y, and z values, respectively. The
    plot has a title and axis labels that match the column
    names, and the intensity of the mesh is proportional
    to the values in the `z_col` column. The function returns
    a Plotly Figure object that can be displayed or saved as
    desired.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x_col : str
        The name of the column to use as the x values.
    y_col : str
        The name of the column to use as the y values.
    z_col : str
        The name of the column to use as the z values.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the 3D mesh plot.
    """
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                intensity=df[z_col] / df[z_col].min(),
                hovertemplate=f"{z_col}: %{{z}}<br>{x_col}: %{{x}}<br>{y_col}: "
                "%{{y}}<extra></extra>",
            )
        ],
    )

    fig.update_layout(
        title=dict(text=f"{y_col} vs {x_col}"),
        scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
        width=700,
        margin=dict(r=20, b=10, l=10, t=50),
    )
    return fig


import plotly.express as px
import plotly.graph_objects as go


def plot_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str,
    opacity: float = 1,
) -> go.Figure:
    """
    Create a 3D scatter plot using Plotly Express.

    This function creates a 3D scatter plot using Plotly Express,
    with the `x_col`, `y_col`, and `z_col` columns of the `df`
    DataFrame as the x, y, and z values, respectively. The points
    in the plot are colored according to the values in the
    `color_col` column, using a continuous color scale. The
    function returns a Plotly Express scatter_3d object that
    can be displayed or saved as desired.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x_col : str
        The name of the column to use as the x values.
    y_col : str
        The name of the column to use as the y values.
    z_col : str
        The name of the column to use as the z values.
    color_col : str
        The name of the column to use for coloring.
    opacity : float
        The opacity (alpha) of the points.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the 3D mesh plot.
    """
    fig = px.scatter_3d(
        data_frame=df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        color_continuous_scale=px.colors.sequential.Viridis_r,
        opacity=opacity,
    )
    return fig
