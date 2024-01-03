"""
根據 MIT 授權由 EL HACHIMI CHOUAIB 提供
"""
import keras.losses
import sklearn.tree as tree
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool1D, Conv1D, Reshape, LSTM
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error, mean_absolute_error, classification_report, precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from math import floor, ceil
from collections import deque
from math import sqrt
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import tensorflow

class Model:
    def __init__(
        self, 
        data_x=None, 
        data_y=None, 
        model_type='knn', 
        c_or_r_or_ts='c',
        training_percent=1, 
        epochs=50, 
        batch_size=32, 
        generator=None,
        validation_percentage=0.2
        ):
        
        if training_percent != 1:
            self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(data_x, 
                                                                                            data_y,
                                                                                            train_size=training_percent,
                                                                                            test_size=1-training_percent)
        else:
            self.x = data_x
            self.y = data_y
        
        self.x = data_x
        self.y = data_y
        self.__y_pred = None
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__model_type = model_type
        self.__boosted_model = None
        self.__generator = generator
        self.history = 'None'
        self.__c_or_r_ts = c_or_r_or_ts
        self.__validation_percentage = validation_percentage
        
        if model_type == 'dt':
            if c_or_r_or_ts == 'c':
                self.__model = tree.DecisionTreeClassifier()
            else:
                self.__model = tree.DecisionTreeRegressor()

        elif model_type == 'svm':
            if c_or_r_or_ts == 'c':
                self.__model = svm.SVC()
            else:
                self.__model = svm.SVR()
                
        elif model_type == 'lr':
            if c_or_r_or_ts == 'c':
                self.__model = LogisticRegression(random_state=2)
            else:
                self.__model = LinearRegression()

        elif model_type == 'nb':
            if c_or_r_or_ts == 'c':
                self.__model = MultinomialNB()
            else:
                self.__model = GaussianNB()
        
        elif model_type == 'rf':
            if c_or_r_or_ts == 'c':
                self.__model = RandomForestClassifier()
            else:
                self.__model = RandomForestRegressor()

        elif model_type == 'dl':
            self.__model = Sequential()

        elif model_type == 'knn':
            if c_or_r_or_ts == 'c':
                self.__model = KNeighborsClassifier(n_neighbors=5)
            else:
                self.__model = KNeighborsRegressor(n_neighbors=5)
            
        else:
            self.__model = None

    def get_generator(self):
        return self.__generator
    
    def set_generator(self, generator):
        self.__generator = generator
        
    def get_model(self):
        return self.__model
    
    def set_model(self, model):
        self.__model = model
    
    def add_layer(self, connections_number=2, activation_function='relu', input_dim=None):
        if input_dim:
            self.__model.add(Dense(connections_number, activation=activation_function, input_dim=input_dim))
        else:
            self.__model.add(Dense(connections_number, activation=activation_function))
            
    def add_lstm_layer(self, connections_number=2, activation_function='relu', input_shape=None):
        if input_shape is not None:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, input_shape=input_shape, return_sequences=True))
        else:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, return_sequences=True))

    def add_conv_2d_layer(self, filter_nbr=1, filter_shape_tuple=(3,3), input_shape=None, activation_function='relu'):
        if input_shape:
            self.__model.add(Conv2D(filters=filter_nbr, kernel_size=filter_shape_tuple, input_shape=input_shape,
                                    activation=activation_function))
        else:
            self.__model.add(Conv2D(filters=filter_nbr, kernel_size=filter_shape_tuple,
                                    activation=activation_function))
            
    def add_conv_1d_layer(self, filter_nbr=1, filter_shape_int=3, input_shape=None, activation_function='relu', strides=10):
        if input_shape:
            #Input size should be (n_features, 1) == (data_x.shape[1], 1)
            self.__model.add(Conv1D(filters=filter_nbr, kernel_size=filter_shape_int, input_shape=input_shape,
                                    activation=activation_function))
        else:
            self.__model.add(Conv1D(filters=filter_nbr, kernel_size=filter_shape_int,
                                    activation=activation_function))

    def add_pooling_2d_layer(self, pool_size_tuple=(2, 2)):
        self.__model.add(MaxPooling2D(pool_size=pool_size_tuple))

    def add_pooling_1d_layer(self, pool_size_int=2):
        self.__model.add(MaxPool1D(pool_size=pool_size_int))

    def add_flatten_layer(self):
        self.__model.add(Flatten())
        
    def add_reshape_layer(self, input_dim):
        """
        for 1dcnn and 2dcnn use this layer as first layer 
        """
        self.__model.add(Reshape((input_dim, 1), input_shape=(input_dim, )))

    """def add_reshape_layer(self, target_shape=None, input_shape=None):
        self.__model.add(Reshape(target_shape=target_shape, input_shape=input_shape))"""

    def add_dropout_layer(self, rate_to_keep_output_value=0.2):
        """ dropout default initial value """
        self.__model.add(Dropout(rate_to_keep_output_value))

    def train(self, loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.SGD(lr=0.001), metrics_as_list=['accuracy']):
        """
        if you pass y as integers use loss='sparse_categorical_crossentropy'
        class Adadelta: Optimizer that implements the Adadelta algorithm.
        class Adagrad: Optimizer that implements the Adagrad algorithm.
        class Adam: Optimizer that implements the Adam algorithm.
