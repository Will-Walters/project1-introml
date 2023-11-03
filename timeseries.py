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
        pass
    def get_
