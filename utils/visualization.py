import pandas as pd
import plotly.express as px
import numpy as np


class PlotlyVisualizer:
    def __init__(self, title="Embedding Visualization"):
        self.title = title

    def plot(self, X_embedded, labels, save_path=None):
        """
        X_embedded: numpy array of shape (N, 2) or (N, 3)
        labels: array-like of length N
        """

        if X_embedded.shape[1] not in [2, 3]:
            raise ValueError(
                "Embedding must have 2 or 3 dimensions for visualization."
            )

        df = pd.DataFrame(X_embedded, columns=["x", "y"] + (["z"] if X_embedded.shape[1] == 3 else []))
        df["label"] = labels

        if X_embedded.shape[1] == 3:
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="label",
                title=self.title,
                opacity=0.9,
            )
        else:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="label",
                title=self.title,
                opacity=0.9,
            )

        fig.update_traces(
            marker=dict(size=3),
            hovertemplate="Label: %{customdata[0]}<extra></extra>",
            customdata=df[["label"]].values,
        )

        if save_path is not None:
            fig.write_html(save_path)

        return fig
