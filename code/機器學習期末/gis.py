import numpy as np
import time
import sys
import math
from numpy.linalg import matrix_power
import nltk
import lib
from dataframe import DataFrame
from vectorizer import Vectorizer, stemming
from model import Model
from rl import *
from chart import Chart
import pandas as pd
from lib import *
import contextily as cx
from matplotlib import pyplot as plt
import geopandas as gpd
import folium as fl
from shapely.geometry import Point

class GIS:
    """
    GIS 類別
    """
    def __init__(self):
        self.data_layers = {}
        self.fig, self.ax = plt.subplots(figsize=(17,17))

    def add_data_layer(self, layer_path, layer_name):
        # 加入地理資訊圖層
        self.data_layers[layer_name] = gpd.read_file(layer_path)
        
    def get_data_layer(self, layer_name):
        # 取得特定圖層的地理資訊資料
        return self.data_layers.get(layer_name)
        
    def plot(self, layer_name, column4color=None, color=None, alpha=0.5, legend=False, figsize_tuple=(15,10)):
        # 繪製地理資訊圖層
        layer = self.data_layers.get(layer_name).to_crs(epsg=3857)
        layer.plot(ax=self.ax, alpha=alpha, edgecolor='k', color=color, legend=legend, 
                   column=column4color, figsize=figsize_tuple)
        cx.add_basemap(ax=self.ax, source=cx.providers.Esri.WorldImagery)
        
    def show(self, layer_name=None, interactive_mode=False):
        # 顯示地理資訊圖層
        if interactive_mode is True: 
            return self.data_layers.get(layer_name).explore()
        else:
            self.ax.set_aspect('equal')
            plt.show()
        
    def get_crs(self, layer_name):
        """
        取得座標參考系統 (CRS)
        EPSG: European Petroleum Survey Group
        """
        return self.get_data_layer(layer_name).crs
    
    def export(self, layer_name, file_name, file_format='geojson'):
        # 導出地理資訊圖層到檔案
        if file_format == 'geojson':
            self.data_layers[layer_name].to_file(file_name + '.geojson', driver='GeoJSON')
        elif file_format == 'shapefile':
            self.data_layers[layer_name].to_file(file_name + '.shp')
            
    def to_crs(self, layer_name, epsg="3857"):
        # 將圖層資料轉換座標參考系統
        self.data = self.data_layers[layer_name].to_crs(epsg)
        
    def set_crs(self, layer_name, epsg="3857"):
        # 設定圖層座標參考系統
        self.data = self.data_layers[layer_name].set_crs(epsg)
        
    def show_points(self, x_y_csv_path, crs="3857"):
        pass
    
    def show_point(self, x_y_tuple, crs="3857"):
        pass
    
    def add_point(self, x_y_tuple, layer_name, crs="3857"):
        # 新增點到地理資訊圖層
        point = Point(0.0, 0.0)
        row_as_dict = {'geometry': point}
        self.data_layers[layer_name].append(row_as_dict, ignore_index=True)
    
    def new_data_layer(self, layer_name, crs="EPSG:3857"):
        # 創建新的地理資訊圖層
        self.data_layers[layer_name] = gpd.GeoDataFrame(crs=crs)
        self.data_layers[layer_name].crs = crs
        
    def add_column(self, layer_name, column, column_name):
        # 新增資料欄位到地理資訊圖層
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            y = pd.Series(y)
        self.data_layers[layer_name][column_name] = y
        
    def show_data_layer(self, layer_name, number_of_row=None):
        # 顯示地理資訊圖層的資料
        if number_of_row is None:
            print(self.get_data_layer(layer_name))
        elif number_of_row < 0:
            return self.get_data_layer(layer_name).tail(abs(number_of_row)) 
        else:
            return self.get_data_layer(layer_name).head(number_of_row) 
        
    def add_row(self, layer_name, row_as_dict):
        # 新增一列資料到地理資訊圖層
        self.data_layers[layer_name] = self.get_data_layer(layer_name).append(row_as_dict, ignore_index=True)
    
    def get_row(self, layer_name, row_index, column=None):
        # 取得圖層中的特定列
        if column is not None:
            return self.data_layers[layer_name].loc[self.data_layers[layer_name][column] == row_index].reset_index(drop=True)
        return self.data_layers[layer_name].iloc[row_index]
    
    def get_layer_shape(self, layer_name):
        """
        回傳圖層的形狀 (列數, 欄數)
        """
        return self.data_layers[layer_name].shape
    
    def get_columns_names(self, layer_name):
        # 取得圖層的欄位名稱
        header = list(self.data_layers[layer_name].columns)
        return header 
    
    def keep_columns(self, layer_name, columns_names_as_list):
        # 保留指定的欄位
        for p in self.get_columns_names(layer_name):
            if p not in columns_names_as_list:
                self.data_layers[layer_name] = self.data_layers[layer_name].drop(p, axis=1)
                
    def get_area_column(self, layer_name):
        # 取得圖層的面積欄位
        return self.get_data_layer(layer_name).area
    
    def get_row_area(self, layer_name, row_index):
        # 取得特定列的面積
        return self.data_layers[layer_name].area.iloc[row_index]
    
    def get_distance(self, layer_name, index_column, row_index_a, row_index_b):
        if 1 == 1:
            other = self.get_row(layer
