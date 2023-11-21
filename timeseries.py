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

import scalecast.util
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
from statsmodels.tsa.stattools import adfuller
import scalecast
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from math import floor, sqrt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from hts import HTSRegressor

        # import VAR model
from statsmodels.tsa.api import VAR

from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from autots import AutoTS, load_daily


def forecaster(f, grid):
    f.auto_Xvar_select(
        try_trend=False,
        try_seasonalities=False,
        max_ar=100
    )
    f.set_estimator('rnn')
    f.ingest_grid(grid)
    f.limit_grid_size(10)  # randomly reduce the big grid to 10
    f.cross_validate(k=3, test_length=24)  # three-fold cross-validation
    f.auto_forecast()
class TimeSeries:
    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.clustered = self.df.groupby('cluster')
        self.stored = self.df.groupby('store_nbr')
        self.clusters = list(self.df['cluster'].unique())
        self.stores = list(self.df['store_nbr'].unique())
        # Need to create model for each store for each product, need to engineer other features
        # First create aggregated features, cluster sales, all sales, store sales
        # First try univariate

    def create_hierarchal(self):
        self.df = self.df.drop(
            columns=['id', 'city', 'state', 'store_type', 'family', 'onpromotion'])
        self.df["cluster_store"] = self.df.apply(lambda x: f"{x['cluster']}_{x['store_nbr']}", axis=1)
        df_bottom_level = self.df.pivot(index="date", columns="cluster_store", values="sales")
        df_middle_level = self.df.groupby(["date", "cluster"]) \
            .sum() \
            .reset_index(drop=False) \
            .pivot(index="date", columns="cluster", values="sales")
        df_total = self.df.groupby("date")["sales"] \
            .sum() \
            .to_frame() \
            .rename(columns={"sales": "total"})
        hierarchy_df = df_bottom_level.join(df_middle_level) \
            .join(df_total)
        hierarchy_df.index = pd.to_datetime(hierarchy_df.index)
        self.hierarchy_df = hierarchy_df.resample("D") \
            .sum()
        print(f"Number of time series at the bottom level: {df_bottom_level.shape[1]}")
        print(f"Number of time series at the middle level: {df_middle_level.shape[1]}")
        clusters = self.df["cluster"].unique()
        stores = self.df["cluster_store"].unique()

        total = {'total': list(clusters)}
        cluster = {str(k): [v for v in stores if v.startswith(str(k))] for k in clusters}
        hierarchy = {**total, **cluster}
        print(hierarchy)
        model_bu_arima = HTSRegressor(model='auto_arima', revision_method='BU', n_jobs=0)
        model_bu_arima = model_bu_arima.fit(self.hierarchy_df, hierarchy)
        pred_bu_arima = model_bu_arima.predict(steps_ahead=7)
        print(pred_bu_arima)

    def get_train_test_of_hierarchy(self, indexing):
        curr_df = self.hierarchy_df[self.hierarchy[indexing]]
        self.tr = curr_df.iloc[:-7, :]
        self.te = curr_df.iloc[-7:, :]

    def get_station(self, column):
        result = adfuller(self.df[column])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    def plot_acf_(self):
        self.tr.set_index(self.tr['date'], inplace=True)
        self.tr.sort_index(inplace=True)
        self.tr.drop(columns=['date'], inplace=True)
        self.te.set_index(self.te['date'], inplace=True)
        self.te.sort_index(inplace=True)
        self.te.drop(columns=['date'], inplace=True)
        plot_acf(self.tr, lags=31)
        pyplot.show()

    def plot_pacf_(self):
        plot_pacf(self.tr['sales'], lags=31)
        pyplot.show()


    def scale_sales(self):
        self.tr_original = self.tr
        self.scaler = QuantileTransformer()
        self.tr['sales'] = self.scaler.fit_transform(self.tr['sales'].reshape(-1, 1))

    def run_ar_(self):
        print("AR")
        train = self.tr['sales'].values.tolist()
        test = self.te['sales'].values.tolist()
        model = AutoReg(train, lags=7)
        model_fit = model.fit()
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
        for i in range(len(predictions)):
            print('predicted=%f, actual=%f' % (predictions[i], test[i]))
        rmse = sqrt(mean_squared_error(test, predictions))
        print('Test RMSE: %.3f' % rmse)
        # plot results
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()

    def run_ma_(self):
        print("MA")
        train = self.tr['sales'].values.tolist()
        test = self.te['sales'].values.tolist()
        model = ARIMA(train, order=(0, 0, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
        rmse = sqrt(mean_squared_error(test, yhat))
        print('Test RMSE: %.3f' % rmse)
        pyplot.plot(test)
        pyplot.plot(yhat, color='red')
        pyplot.show()

    def run_arma_(self):
        print("arma")
        train = self.tr['sales'].values.tolist()
        test = self.te['sales'].values.tolist()
        model = ARIMA(train, order=(7, 0, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
        rmse = sqrt(mean_squared_error(test, yhat))
        print('Test RMSE: %.3f' % rmse)
        pyplot.plot(test)
        pyplot.plot(yhat, color='red')
        pyplot.show()

    def run_arima_(self, p, d, q):
        print("arima")
        train = self.tr['sales'].values.tolist()
        test = self.te['sales'].values.tolist()
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        yhat = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')
        rmse = sqrt(mean_squared_error(test, yhat))
        # print('Test RMSE: %.3f' % rmse)
        # pyplot.plot(test)
        # pyplot.plot(yhat, color='red')
        # pyplot.show()
        #print(yhat)
        #print(rmse)
        return [rmse, test, yhat]

    def get_best_arima_(self, ps, ds, qs):
        best = 1000000
        best_name = [None, None, None]
        for i in itertools.product(ps, ds, qs):
            curr = self.run_arima_(i[0], i[1], i[2])
            if curr[0] < best:
                best = curr[0]
                best_name = i
            print(curr)
            print(i)
        pyplot.plot(curr[1])
        pyplot.plot(curr[2], color='red')
        pyplot.show()
        return best_name

    def run_auto_ts_(self):
        model = AutoTS(
            forecast_length=7,
            frequency='infer',
            prediction_interval=0.9,
            ensemble='auto',
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards"
        )
    def get_store(self, s):
        curr = self.df.groupby('store_nbr').get_group(s)
        return TimeSeries(str(s), curr)
    def drop_unnecessary_for_store(self):
        self.df = self.df.drop(columns=['id', 'city', 'state', 'store_type', 'cluster', 'store_nbr', 'family', 'onpromotion'])
        # just sales and onpromotion for base model

    def set_train_test(self, ratio=0.95):
        self.df.set_index(self.df['date'], inplace=True)
        self.df.sort_index(inplace=True)
        self.tr = self.df.iloc[:-7,:]
        self.te = self.df.iloc[-7:,:]

    def run_var(self):
        self.tr = self.tr.drop(columns=['date'])
        pred_dates = self.te.loc[:, 'date']
        self.te = self.te.drop(columns=['date'])
        # fit model
        model = VAR(self.tr)
        model_fit = model.fit()
        pred = model_fit.forecast(model_fit, steps=len(pred_dates))
        print(pred)
    def run_arima(self):
        # Import Libraries
        # Drop the date parameter as we don't need this for Auto_Arima
        hist = self.tr.drop(["Date"], axis=1)
        # Select the exogenous variable
        X = hist.drop(["sales"], axis=1)
        # select the endogenous variable
        y = hist["sales"].values.reshape(-1, 1)
        # Compute the minimum and maximum to be used for later scaling
        scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
        scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y)
        # Scale features of X according to feature_range
        X_hist_scaled = scaler_X.transform(X)
        # Scale features of y according to feature_range
        y_hist_scaled = scaler_y.transform(y)
        # Scale features of X for which y to be predicted
        # Drop the date parameter as we don't need this for Auto_Arima
        pred_drivers_scaled = scaler_X.transform(self.te.drop("date", axis=1))
        # ------------------------------------------------
        # Create an Auto_Arima model
        # ------------------------------------------------
        # Broyden-Fletcher-Goldfarb-Shanno (BFGS) solver is used
        # m: refers to the number of periods in each season
        # information_criterion: to select the best ARIMA model
        model = auto_arima(y_hist_scaled, \
                           exogenous=X_hist_scaled, \
                           method='bfgs', \
                           solver='bfgs', \
                           m=4, \
                           information_criterion='bic')
        # Fit ARIMA
        arimax_reg_eval = model.fit(y_hist_scaled, X_hist_scaled)
        # ------------------------------------------------
        # Forecast based on model fitted
        # ------------------------------------------------
        pred_future_scaled = arimax_reg_eval \
            .predict( \
            n_periods=self.te.shape[0], \
            exogenous=pred_drivers_scaled) \
            .reshape(-1, 1)
        # Undo the scaling of X according to feature_range
        pred_future = scaler_y.inverse_transform(pred_future_scaled)
        # Create a Pandas DataFrame
        predictions = pd.DataFrame(pred_future)
        # Add column header
        predictions.columns = ['sales']
        # Calculate the prediction start/first month
        prediction_start_day = \
            (self.tr["date"].iloc[-1] + \
             pd.DateOffset(days=1)) \
                .strftime('%Y-%m-%d')
        # Total forecasting period considered
        forecastingPeriod = len(self.te)
        # Create a DateTimeIndex
        prediction_date_range = pd.date_range( \
            prediction_start_day, \
            periods=forecastingPeriod, \
            freq='D')
        # Assign the DateTimeIndex as DataFrame index
        predictions.index = prediction_date_range
        print(predictions)
'''
    def run_prob_lstm(self):
        transformer, reverter = find_optimal_transformation(
            f,
            estimator='lstm',
            epochs=10,
            set_aside_test_set=True,  # prevents leakage so we can benchmark the resulting models fairly
            return_train_only=True,  # prevents leakage so we can benchmark the resulting models fairly
            verbose=True,
            m=52,  # what makes one seasonal cycle?
            test_length=24,
            num_test_sets=3,
            space_between_sets=12,
            detrend_kwargs=[
                {'loess': True},
                {'poly_order': 1},
                {'ln_trend': True},
            ],
        )
        rnn_grid = gen_rnn_grid(
            layer_tries=100,
            min_layer_size=1,
            max_layer_size=5,
            units_pool=[100],
            epochs=[100],
            dropout_pool=[0, 0.05],
            validation_split=.2,
            callbacks=EarlyStopping(
                monitor='val_loss',
                patience=3,
            ),
            random_seed=20,
        )  # make a really big grid and limit it manually
        pipeline = Pipeline(
            steps=[
                ('Transform', transformer),
                ('Forecast', forecaster),
                ('Revert', reverter),
            ]
        )
        f = pipeline.fit_predict(f, grid=rnn_grid)
        f.plot(ci=True)
        plt.show()
    def get_motifs_anomalies(self, column, windows, stores=None, clusters=None):
        if stores is None:
            stores = [self.stores[0]]
        if clusters is None:
            #clusters = [self.clusters[set(self.df.loc[df=])]]
            pass
        for window in windows:
            mp = stumpy.stump(self.df[column], window)
            # motif_idx is strongest motif
            motif_idx = np.argsort(mp[:, 0])[0]
            nearest_neighbor_idx = mp[motif_idx, 1]

            # This code is going to be utilized to control the axis labeling of the plots
            DAY_MULTIPLIER = 7  # Specify for the amount of days you want between each labeled x-axis tick
'''


