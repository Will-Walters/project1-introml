import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import stumpy
import numpy as np
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport
import itertools
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, MaxAbsScaler, RobustScaler, QuantileTransformer
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from prettytable import PrettyTable
import kaleido
import shutil
from sklearn.linear_model import LinearRegression

class TimeSeries:
    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.clustered = self.df.groupby('cluster')
        self.stored = self.df.groupby('store_nbr')
        self.clusters = list(self.df['cluster'].unique())
        self.stores = list(self.df['store_nbr'].unique())
    def get_station(self):
        
    def get_motifs_anomalies(self, column, windows, stores=None, clusters=None):
        if stores is None:
            stores = [self.stores[0]]
        if clusters is None:
            clusters = [self.clusters[set(self.df.loc[df=])]]
        for window in windows:
            mp = stumpy.stump(self.df[column], window)
            # motif_idx is strongest motif
            motif_idx = np.argsort(mp[:, 0])[0]
            nearest_neighbor_idx = mp[motif_idx, 1]

            # This code is going to be utilized to control the axis labeling of the plots
            DAY_MULTIPLIER = 7  # Specify for the amount of days you want between each labeled x-axis tick



