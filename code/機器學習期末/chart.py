import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dataframe import DataFrame
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

class Chart:
    # 定義支援的圖表類型
    chart_type_list = ['line', 'bar', 'box', 'swarm', 'strip_swarm', 'count', 'reg', 'dist', 'point', 'pair',
                  'correlation_map', 'scatter', 'heat_map']

    def __init__(self, dataframe=None, column4x=None, chart_type='pair', group_by=None, columns_names_list=None, plotly=False):
        self.dataframe = dataframe
        if column4x is None:
            self.column4x = dataframe.index
        else:
            self.column4x = column4x
        self.chart_type = chart_type
        self.group_by = group_by
        self.columns_names_list = columns_names_list
        self.plotly = plotly
        # 初始化 Matplotlib 圖表
        self.fig, self.ax = plt.subplots(figsize=(18, 6))

    def add_data_to_show(self, data_column=None, column4hover=None, column4size=None):
        print(self.chart_type)
        if self.plotly == True:
            # 根據不同的圖表類型，使用 Plotly 或 Seaborn 進行繪圖
            if self.chart_type == self.chart_type_list[0]:
                self.fig = px.line(self.dataframe, x=self.column4x, y=data_column, color=self.group_by, hover_name=column4hover)
            elif self.chart_type == self.chart_type_list[1]:
                self.ax = sns.barplot(data=self.dataframe, x=self.column4x, y=data_column, hue=self.group_by)
                loc = plticker.MultipleLocator(base=1.0)
                self.ax.xaxis.set_major_locator(loc)
            # 其他圖表類型的處理...
        else:
            # 在 Matplotlib 中繪製圖表
            if self.chart_type == self.chart_type_list[0]:
                self.ax = sns.lineplot(data=self.dataframe, x=self.column4x, y=data_column, markers=True, hue=self.group_by)
            # 其他圖表類型的處理...

    def plot_on_map(self, iso_locations_column=None, circle_size_column=None, animation_frame_column=None,
                    hover_name_column=None, projection='natural earth', scope='world'):
        # 使用 Plotly 繪製地圖
        self.fig = px.scatter_geo(
            self.dataframe,
            locations=iso_locations_column,
            size=circle_size_column,
            animation_frame=animation_frame_column,
            hover_name=hover_name_column,
            color=self.group_by,
            projection=projection,
            scope=scope,
        )
        
    def plot_colored_map(self, iso_locations_column=None, color_column=None, animation_frame_column="Year",
                         scope='world', hover_name_column=None):
        # 使用 Plotly 繪製帶有顏色的地圖
        self.fig = px.choropleth(
            self.dataframe,
            locations=iso_locations_column,
            scope=scope,
            color=color_column,
            hover_name=hover_name_column,
            color_continuous_scale=px.colors.sequential.Plasma,
            animation_frame=animation_frame_column,
            projection='natural earth'
        )

    def show(self):
        if self.plotly:
            self.fig.show()
        else:
            plt.show()

    def config(self, title="", x_label="X", y_label="Y", x_limit_i=None, x_limit_f=None, y_limit_i=None, y_limit_f=None,
               interval=None, x_rotation_angle=90, y_rotation_angle=0, titile_font_size=50, x_label_font_size=15,
               y_label_font_size=15, x_font_size=13, y_font_size=13):
        if self.plotly:
            # 設定 Plotly 圖表的布局
            self.fig.update_layout(
                title_text=title,
            )
        else:
            # 設定 Matplotlib 圖表的標題、軸範圍和標籤
            plt.title(title)
            plt.xlim(x_limit_i, x_limit_f)
            plt.ylim(y_limit_i, y_limit_f)
            plt.xticks(rotation=x_rotation_angle, fontsize=x_font_size)
            plt.yticks(rotation=y_rotation_angle, fontsize=y_font_size)
            self.ax.set_title(title, fontsize=titile_font_size)
            self.ax.set_xlabel(x_label, fontsize=x_label_font_size)
            self.ax.set_ylabel(y_label, fontsize=y_label_font_size)
            if interval is not None:
                loc = plticker.MultipleLocator(base=interval)
                self.ax.xaxis.set_major_locator(loc)

    def save(self, chart_path="output.png", transparent=True):
        print(self.chart_type)
        # 儲存圖表為檔案
        self.fig.savefig(chart_path, transparent=transparent, bbox_inches='tight')
