from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    CustomJS,
    TapTool,
    Range1d,
    HoverTool,
    BoxSelectTool,
)
from bokeh.plotting import figure, output_file, show, output_notebook
import bokeh.palettes
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.layouts import widgetbox, column
from bokeh.models.widgets import Button
import pandas as pd
import sqlite3


def notebook_setup():
    output_notebook()


def plot_landscape(dataframe, color="spacegroup", **kwargs):
    tooltips = [(name, f"@{name}") for name in dataframe.columns]
    categorical = color == "spacegroup" or str(dataframe.dtypes[color]) == "object"

    tools = kwargs.get(
        "tools", "box_zoom,pan,box_select,lasso_select,wheel_zoom,reset,save"
    )
    x = kwargs.get("x", "density")
    y = kwargs.get("y", "energy")
    plot = figure(
        tools=tools,
        tooltips=tooltips,
        title=kwargs.get("title", f"{y} vs. {x}"),
        plot_width=kwargs.get("plot_width", 800),
        plot_height=kwargs.get("plot_height", 600),
    )
    plot.title.text_font = "Arial"
    plot.title.align = "center"
    plot.title.text_font_size = "16pt"
    plot.xaxis.axis_label = kwargs.get("xlabel", x)
    plot.xaxis.axis_label_text_font = "Arial"
    plot.xaxis.axis_label_text_font_size = "12pt"
    plot.yaxis.axis_label = kwargs.get("ylabel", y)
    plot.yaxis.axis_label_text_font = "Arial"
    plot.yaxis.axis_label_text_font_size = "12pt"
    plot.ygrid.grid_line_alpha = 0.6
    plot.xgrid.grid_line_alpha = 0.6
    button = Button(label="Download res", button_type="success")
    if categorical:
        sources = {}
        renderers = {}
        groups = dataframe[color].unique()
        palette = bokeh.palettes.Category20[20]
        for (group, data), color in zip(dataframe.groupby(color), palette):
            sources[group] = ColumnDataSource(data=data)
            renderers[group] = plot.circle(
                x=x,
                y=y,
                size=8,
                source=sources[group],
                fill_color=color,
                fill_alpha=0.8,
                line_color="black",
                line_alpha=1.0,
                muted_color="gray",
                muted_alpha=0.2,
            )
    else:
        colormapper = LinearColorMapper(
            palette=kwargs.get("palette", "Viridis256"),
            low=kwargs.get("cmin", dataframe[color].min()),
            high=kwargs.get("cmax", dataframe[color].max()),
        )
        source = ColumnDataSource(data=dataframe)
        renderer = plot.circle(
            x=x,
            y=y,
            size=8,
            source=source,
            fill_color={"field": color, "transform": colormapper},
            fill_alpha=0.8,
            line_color="black",
            line_alpha=1.0,
            muted_color="gray",
            muted_alpha=0.2,
        )

    return show(column(plot, button))
