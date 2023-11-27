# functions to plots metrics of models from
# a dataframe of metrics and models names
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import numpy as np


# function to plot models metrics
def plot_metrics_and_time(df_m):
    # Metrics comparison
    plt.figure(figsize=(15, 5))
    sns.set_style("whitegrid", {"grid.color": ".8", "grid.linestyle": ":"})

    plt.subplot(1, 2, 1)
    data = df_m[["accuracy", "recall_macro", "f1_macro", "recall_0", "precision_0"]]
    sns.set_style("whitegrid", {"grid.color": ".8", "grid.linestyle": ":"})
    sns.lineplot(data=data, dashes=True, marker="D", palette="Set2")
    plt.axhline(y=0.5, linestyle="--", color="red")
    plt.xticks(np.arange(len(df_m)), list(df_m["model"].values), rotation=90)
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylabel("Metric value")
    plt.title("Metrics comparison")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.28, 0.37), fontsize=9)

    plt.subplot(1, 2, 2)

    # time to fit and time to predict
    data = df_m[["time_train", "time_predict"]]
    sns.set_style("whitegrid", {"grid.color": ".8", "grid.linestyle": ":"})
    sns.lineplot(data=data, dashes=True, marker="D", palette="Set2")
    plt.xticks(np.arange(len(df_m)), list(df_m["model"].values), rotation=90)
    # plt.yticks(np.linspace(0, 1 , 11))
    plt.ylabel("time (s)")
    plt.title("Training and Prediction time")

    plt.show()


# fucntion to visualize a model metrics in a radar chart make sure to
# remove time_train and time_predict before applying the function


def plot_radar_metrics(df, model_name):
    """
    Function to plot a radar chart of a model metrics given in df
    input : metrics such as accuracy, f1_macro etc
    output : radar chart of metrics values
    ! Make sure df contains only the metrics to be visualized
    """
    dp = df.T.reset_index()
    dp.columns = ["theta", "r"]
    title = model_name + " performance"
    fig = px.line_polar(
        dp, r="r", theta="theta", line_close=True, markers=True, width=500, height=500
    )
    fig.update_traces(fill="toself")
    fig.update_layout(title_text=title, title_x=0.5)
    fig.show()


# Viusalize two models metrics in a radar chart


def plot_radar_mult(df):
    """function to plot many models performance in the same radar chart
    input: df, metrics scores of the models and the names of the models
    """
    models_list = df["model"].to_list()
    data = df.set_index("model").T
    metrics = data.index.to_list()

    fig = go.Figure()
    for model in models_list:
        fig.add_trace(
            go.Scatterpolar(
                r=data[model].to_list(),
                theta=metrics,
                # fill='toself',
                name=model,
            )
        )
    fig.update_layout(
        title="Metrics comparison",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        width=600,
        height=600,
        showlegend=True,
    )
    fig.show()


def plot_radar_some(df, models_list):
    """function to plot many models performance for models in lists_of_models.
    input: df, metrics scores of the models and the names of the models

    """
    # models_list = df["model"].to_list()
    data = df.set_index("model").T
    metrics = data.index.to_list()

    fig = go.Figure()
    for model in models_list:
        fig.add_trace(
            go.Scatterpolar(
                r=data[model].to_list(),
                theta=metrics,
                # fill='toself'
                name=model,
            )
        )
    fig.update_layout(
        title="Metrics comparison",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        width=600,
        height=600,
        showlegend=True,
    )
    fig.show()


# function to visualise barplots of a given metric score for multiple models
# metric must be one of :'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_weighted',
#       'recall_0', 'recall_macro', 'recall_weighted', 'precision_0',
#       'precision_macro', 'precision_weighted'
def barplot_metric_mult(df, metric="accuracy", dtick=0.1):
    """
    function to plot a barplot of a given metric for multiple modèles
    metric and models names given in df.
    """

    # make sure the metric is in df
    fig = px.bar(
        df,
        x="model",
        y=metric,
        text_auto=".2f",
        title="Models " + metric,
        color=df[metric],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="cyan")
    fig.update_yaxes(tick0=0, dtick=dtick)
    fig.show()


# Lineplot of models metrics scores
def lineplot_metrics(df):
    """line plots of models metrics score for comparison
    metrics in abciss and model are colors
    """
    data = df.set_index("model")
    fig = px.line(data.T, markers=True, title="Metrics x Models comparison")
    fig.update_yaxes(tick0=0, dtick=0.1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.show()


# Lineplot of models metrics scores
def lineplot_models_vs_metrics(df):
    """line plots of models metrics score for comparison
    models in adciss and metrics are colors
    """
    data = df.set_index("model")
    fig = px.line(data, markers=True, title="Models x Metrics comparison")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    fig.update_yaxes(tick0=0, dtick=0.1)
    fig.show()


# function to plot stack barplot of two metrics
def stack_barplot(df, name1, name2):
    fig = go.Figure(
        data=[
            go.Bar(
                name=name1,
                x=df["model"].to_list(),
                y=df[name1].to_list(),
                offsetgroup=0,
            ),
            go.Bar(
                name=name2,
                x=df["model"].to_list(),
                y=df[name2].to_list(),
                offsetgroup=1,
            ),
        ],
        layout=go.Layout(title=name1 + " versus " + name2, yaxis_title="Value"),
    )
    fig.show()
