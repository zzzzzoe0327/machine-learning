from datetime import timedelta
from math import ceil
import pandas as pd
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from lib import *
from vectorizer import Vectorizer
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from keras.preprocessing.sequence import TimeseriesGenerator as SG
from sklearn.datasets import load_iris, load_boston
import numpy as np

class DataFrame:
    """
    這是一個簡單的DataFrame類別，提供一些基本的資料處理功能。
    """
    __vectorizer = None
    __generator = None

    def __init__(self, data_link=None, columns_names_as_list=None, data_types_in_order=None, delimiter=',',
                 file_type='csv', line_index=None, skip_empty_line=False, sheet_name='Sheet1'):
        # 初始化DataFrame物件
        if data_link is not None:
            if file_type == 'csv':
                # 讀取CSV檔案
                self.__dataframe = pd.read_csv(data_link, encoding='utf-8', delimiter=delimiter, low_memory=False, error_bad_lines=False, skip_blank_lines=False)
            elif file_type == 'json':
                # 讀取JSON檔案
                self.__dataframe = pd.read_json(data_link, encoding='utf-8')
            elif file_type == 'xls':
                # 讀取Excel檔案
                self.__dataframe = pd.read_excel(data_link, sheet_name=sheet_name)
            elif file_type == 'dict':
                # 從字典建立DataFrame
                self.__dataframe = pd.DataFrame.from_dict(data_link)
            elif file_type == 'matrix':
                # 從矩陣建立DataFrame
                index_name = [i for i in range(len(data_link))]
                colums_name = [i for i in range(len(data_link[0]))]
                self.__dataframe = pd.DataFrame(data=data_link, index=index_name, columns=colums_name)
                """data = array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])
                pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]) """
            types = {}
            if data_types_in_order is not None and columns_names_as_list is not None:
                # 指定資料型態
                self.__dataframe.columns = columns_names_as_list
                for i in range(len(columns_names_as_list)):
                    types[columns_names_as_list[i]] = data_types_in_order[i]
            elif columns_names_as_list is not None:
                # 若沒有指定資料型態，預設為字串
                self.__dataframe.columns = columns_names_as_list
                for p in columns_names_as_list:
                    types[p] = str

            self.__dataframe = self.get_dataframe().astype(types)

            if line_index is not None:
                # 指定索引
                self.__dataframe.index = line_index
        else:
            # 若沒有提供資料，則建立空的DataFrame
            self.__dataframe = pd.DataFrame()
        
    def get_generator(self):
        return self.__generator
    
    def get_index(self):
        return self.__dataframe.index.to_list()
    
    def add_time_serie_row(self, date_column, value_column, value, date_format='%Y'):
        # 新增時間序列資料
        last_date = self.get_index()[-1] + timedelta(days=1)
        dataframe = DataFrame([{value_column: value, date_column: last_date}], file_type='dict')
        dataframe.to_time_series(date_column, value_column, one_row=True, date_format=date_format)
        self.append_dataframe(dataframe.get_dataframe())
        
    def set_generator(self, generator):
        self.__generator = generator

    def set_dataframe(self, data, data_type='df'):
        # 設定DataFrame
        if data_type == 'matrix':
            index_name = [i for i in range(len(data))]
            colums_name = [i for i in range(len(data[0]))]
            self.__dataframe = pd.DataFrame(data=data, index=index_name, columns=colums_name)
        elif data_type == 'df':
            self.__dataframe = data

    def get_data_types(self, show=True):
        # 取得資料型態
        types = self.get_dataframe().dtypes
        if show:
            print(types)
        return types
    
    def set_data_types(self, column_dict_types):
        # 設定資料型態
        self.__dataframe = self.get_dataframe().astype(column_dict_types)
        
    def set_same_type(self, same_type='float64'):
        """
        example of types: float64, object
        """
        for p in self.get_columns_names():
            self.set_column_type(p, same_type)

    def describe(self, show=True):
        # 描述統計資訊
        description = self.get_dataframe().describe()
        if show:
            print(description)
        return description
    
    def reset_index(self):
        # 重置索引
        self.set_dataframe(self.__dataframe.reset_index())

    def get_dataframe_as_sparse_matrix(self):
        # 取得稀疏矩陣
        return scipy.sparse.csr_matrix(self.__dataframe.to_numpy())

    def get_column_as_list(self, column):
        # 取得特定欄位的資料列表
        return list(self.get_column(column))
    
    def get_column_as_joined_text(self, column):
        # 取得特定欄位的資料合併成文字
        return ' '.join(list(self.get_column(column)))

    def get_term_doc_matrix_as_df(self, vectorizer_type='count'):
        # 取得詞頻矩陣
        corpus = list(self.get_column('comment'))
        indice = ['doc' + str(i) for i in range(len(corpus))]
        v = Vectorizer(corpus, vectorizer_type=vectorizer_type)
        self.set_dataframe(DataFrame(v.get_sparse_matrix().toarray(), v.get_features_names(),
                                      line_index=indice, file_type='ndarray').get_dataframe())

    def get_dataframe_from_dic_list(self, dict_list):
        # 由字典列表建立DataFrame
        v = DictVectorizer()
        matrice = v.fit_transform(dict_list)
        self.__vectorizer = v
        self.set_dataframe(DataFrame(matrice.toarray(), v.get_feature_names()).get_dataframe())

    def check_decision_function_on_column(self, column, decision_func):
        # 檢查某欄位是否符合特定條
