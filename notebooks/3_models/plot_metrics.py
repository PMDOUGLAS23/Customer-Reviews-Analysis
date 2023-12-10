# functions to plots metrics of models from
# a dataframe of metrics and models names
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from IPython.core.display import display, HTML


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
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
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
    models in abciss and metrics are colors
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


# function to plot stack barplot of 5 metrics
def stack_mult_barplots(df, name1, name2, name3, name4, name5):
    df = df.round(2)
    fig = go.Figure(
        data=[
            go.Bar(
                name=name1,
                x=df["model"].to_list(),
                y=df[name1].to_list(),
                offsetgroup=0,
                text=df[name1].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name2,
                x=df["model"].to_list(),
                y=df[name2].to_list(),
                offsetgroup=1,
                text=df[name2].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name3,
                x=df["model"].to_list(),
                y=df[name3].to_list(),
                offsetgroup=2,
                text=df[name3].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name4,
                x=df["model"].to_list(),
                y=df[name4].to_list(),
                offsetgroup=3,
                text=df[name4].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name5,
                x=df["model"].to_list(),
                y=df[name5].to_list(),
                offsetgroup=4,
                text=df[name5].to_list(),
                textposition="auto",
            ),
        ],
        layout=go.Layout(title="Models versus of metrics", yaxis_title="Value"),
    )

    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0.5,
        x1=1,
        y1=0.5,
        line=dict(color="magenta", width=1, dash="dash"),
    )

    fig.show()


# function to plot stack barplot of 5 metrics
def barplot_model_vs_metrics_group(df, name1, name2, name3, name4, name5):
    df = df.round(2)
    fig = go.Figure(
        data=[
            go.Bar(
                name=name1,
                x=df["model"].to_list(),
                y=df[name1].to_list(),
                offsetgroup=0,
                text=df[name1].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name2,
                x=df["model"].to_list(),
                y=df[name2].to_list(),
                offsetgroup=1,
                text=df[name2].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name3,
                x=df["model"].to_list(),
                y=df[name3].to_list(),
                offsetgroup=2,
                text=df[name3].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name4,
                x=df["model"].to_list(),
                y=df[name4].to_list(),
                offsetgroup=3,
                text=df[name4].to_list(),
                textposition="auto",
            ),
            go.Bar(
                name=name5,
                x=df["model"].to_list(),
                y=df[name5].to_list(),
                offsetgroup=4,
                text=df[name5].to_list(),
                textposition="auto",
            ),
        ],
        layout=go.Layout(title="Group barplot of Models Metrics", yaxis_title="Value"),
    )

    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0.5,
        x1=1,
        y1=0.5,
        line=dict(color="magenta", width=1, dash="dash"),
    )

    fig.show()


# function to plot stack barplot of 5 metrics
# Group models on metrics
def barplot_models_group_vs_metric(df, model_list):
    # transpose and prepare df
    dt = df.round(2)
    dt = dt.set_index("model").T.reset_index()
    dt.rename(columns={"index": "metric"}, inplace=True)

    fig = go.Figure(
        data=[
            go.Bar(
                name=name,
                x=dt["metric"].to_list(),
                y=dt[name].to_list(),
                offsetgroup=i,
                text=dt[name].to_list(),
                textposition="auto",
            )
            for (i, name) in enumerate(model_list)
        ],
        layout=go.Layout(title="Group of models versus metrics", yaxis_title="Value"),
    )
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0.5,
        x1=1,
        y1=0.5,
        line=dict(color="magenta", width=1, dash="dash"),
    )

    fig.show()


# function to visualize confusion matrix side by side


def display_side_by_side(dfs: list, captions: list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += (
            df.style.set_table_attributes("style='display:inline'")
            .set_caption(caption)
            ._repr_html_()
        )
        output += "\xa0\xa0\xa0"
    display(HTML(output))
