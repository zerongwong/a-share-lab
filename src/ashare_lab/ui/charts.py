from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_chart(frame: pd.DataFrame, *, title: str) -> go.Figure:
    view = frame.tail(250).copy()
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )
    figure.add_trace(
        go.Candlestick(
            x=view["trade_date"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="K线",
            increasing_line_color="#d94a3a",
            decreasing_line_color="#238a65",
        ),
        row=1,
        col=1,
    )
    colors = {"ma20": "#e3a32b", "ma60": "#466fd5", "ma120": "#8856a7"}
    for column, color in colors.items():
        if column in view:
            figure.add_trace(
                go.Scatter(
                    x=view["trade_date"],
                    y=view[column],
                    name=column.upper(),
                    line={"width": 1.4, "color": color},
                ),
                row=1,
                col=1,
            )
    if "volume_shares" in view:
        volume_colors = [
            "#d97b6c" if close >= open_ else "#62a98e"
            for open_, close in zip(view["open"], view["close"], strict=True)
        ]
        figure.add_trace(
            go.Bar(
                x=view["trade_date"],
                y=view["volume_shares"],
                marker_color=volume_colors,
                name="成交量（股）",
            ),
            row=2,
            col=1,
        )
    figure.update_layout(
        title=title,
        height=630,
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return figure
