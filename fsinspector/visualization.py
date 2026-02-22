"""Plot embedded points (2D/3D) with labels."""
import pandas as pd
import plotly.graph_objects as go
import numpy as np


class EmbeddingVisualizer:
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
        df["label"] = list(labels)
        color_codes = df["label"].astype("category").cat.codes

        if X_embedded.shape[1] == 3:
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=df["x"],
                        y=df["y"],
                        z=df["z"],
                        mode="markers",
                        marker=dict(size=3, color=color_codes, colorscale="Turbo"),
                        customdata=df[["label"]].values,
                        hovertemplate="Label: %{customdata[0]}<extra></extra>",
                    )
                ],
                layout=dict(title=self.title),
            )
        else:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df["x"],
                        y=df["y"],
                        mode="markers",
                        marker=dict(size=3, color=color_codes, colorscale="Turbo"),
                        customdata=df[["label"]].values,
                        hovertemplate="Label: %{customdata[0]}<extra></extra>",
                    )
                ],
                layout=dict(title=self.title),
            )

        if save_path is not None:
            fig.write_html(save_path)

        return fig
