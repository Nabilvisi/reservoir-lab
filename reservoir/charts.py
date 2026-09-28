import numpy as np
import plotly.graph_objects as go

COLORS = ["#43b6ff", "#f6c85f", "#bb96ff"]


def volume(values, spacing, label, exaggeration=8, wells=False):
    z, y, x = np.indices(values.shape)
    dz, dy, dx = spacing
    fig = go.Figure(go.Scatter3d(x=(x.ravel() + .5) * dx, y=(y.ravel() + .5) * dy,
        z=(z.ravel() + .5) * dz, mode="markers", name="Cell centres", showlegend=False,
        marker=dict(size=4.5, symbol="square", color=values.ravel(), colorscale="Viridis", opacity=.8,
                    colorbar=dict(title=label, thickness=12)),
        customdata=values.ravel(), hovertemplate="x %{x:.1f} m<br>y %{y:.1f} m<br>Depth %{z:.2f} m<br>Value %{customdata:.4g}<extra></extra>"))
    if wells:
        for name, xx, yy, color in [("Injector", .5 * dx, .5 * dy, COLORS[0]),
                                    ("Producer", (values.shape[2]-.5)*dx, (values.shape[1]-.5)*dy, COLORS[1])]:
            fig.add_trace(go.Scatter3d(x=[xx, xx], y=[yy, yy], z=[0, values.shape[0]*dz],
                mode="lines+text", text=[name, ""], name=name, line=dict(width=10, color=color)))
    lengths = np.asarray(values.shape) * spacing
    fig.update_layout(height=510, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Relative depth (m)",
                   zaxis=dict(autorange="reversed"), aspectmode="manual",
                   aspectratio=dict(x=1, y=lengths[1]/lengths[2], z=lengths[0]/lengths[2]*exaggeration),
                   camera=dict(eye=dict(x=1.4, y=-1.6, z=1.1))), legend=dict(orientation="h"))
    return fig


def production(history):
    fig = go.Figure()
    for label, col, color in zip(["Water", "Oil", "Gas"], ["water_m3_day", "oil_m3_day", "gas_m3_day"], COLORS):
        fig.add_trace(go.Scatter(x=history.day, y=history[col], name=label, line=dict(color=color, width=3)))
    fig.update_layout(height=340, xaxis_title="Time (days)", yaxis_title="Reservoir volume (m³/day)",
                      margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))
    return fig
