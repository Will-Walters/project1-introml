import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import stumpy
import matplotlib.pyplot as plt


def preprocess():
    # Only run if you don't have df_train.parquet and df_test.parquet
    df_dict = dict()
    files = []
    for file in os.listdir('.'):
        if file.endswith('.parquet'):
            files.append(file)
            print(file)
            df_dict[file] = pd.read_parquet(file)

    for file in files:
        curr = df_dict[file]
        print(file)
        print(curr.describe())
    df_train1 = df_dict['train.parquet'].merge(df_dict['oil.parquet'], on='date', how='left')
    df_train1 = df_train1.merge(df_dict['holidays_events.parquet'], on='date', how='left')
    df_train1 = df_train1.merge(df_dict['stores.parquet'], on='store_nbr', how='left')
    df_train1 = df_train1.merge(df_dict['transactions.parquet'], on=['date', 'store_nbr'], how='left')
    df_train = df_train1.rename(columns={"type_x": "holiday_type", "type_y": "store_type"})
    df_train.to_parquet('df_train.parquet')

    df_test1 = df_dict['test.parquet'].merge(df_dict['oil.parquet'], on='date', how='left')
    df_test1 = df_test1.merge(df_dict['holidays_events.parquet'], on='date', how='left')
    df_test1 = df_test1.merge(df_dict['stores.parquet'], on='store_nbr', how='left')
    df_test1 = df_test1.merge(df_dict['transactions.parquet'], on=['date', 'store_nbr'], how='left')
    df_test = df_test1.rename(columns={"type_x": "holiday_type", "type_y": "store_type"})
    df_test.to_parquet('df_test.parquet')


def compare_two_timeseries(A, B, window):
    mp = stumpy.stump(T_A=A, m=window, T_B=B, ignore_trivial=False)
    A_motif_index = mp[:,0].argmin()
    plt.xlabel("Subsequence")
    plt.ylabel("Matrix Profile")
    plt.scatter(A_motif_index,
                mp[A_motif_index, 0],
                c='red',
                s=100)
    plt.show()

    def check_1d_store():
        pass
    def check_timeseries(t):
        nas = sum(pd.isna(t))
        if nas > 0:
            print("Warning")
            print(nas)
            print("Missing values")
        else:
            print("No missing values")

if __name__ == "__main__":

    # preprocess()

    df_train = pd.read_parquet('df_train.parquet')
    df_test = pd.read_parquet('df_test.parquet')

    print(df_train.describe())
    print(df_test.describe())

    # First let's do EDA of df_train
    print("Columns")
    print(df_train.columns)
    print("Shape")
    print(df_train.shape)
    print("Date range")
    dt1 = pd.to_datetime(max(df_train.loc[:, 'date']))
    dt2 = pd.to_datetime(min(df_train.loc[:, 'date']))
    datediff = (dt1 - dt2).days
    print(dt2)
    print(dt1)
    print("Days between")
    print(datediff)
    '''
    Sales is what we will be predicting so we should look at the timeseries
    Time series is daily and is associated with a unique store_nbr each day,
    so for every day there will n rows for n unique stores
    Furthermore, each unique store is in a cluster, which are geographically related stores
    
    Initial Assumptions:
    Every store will have its own behavior, we must assume historical behaivor can predict future behaivor, or else
    there is no point in modeling.
    Clustered stores are assumed to be weakly dependent on eachother.
    
    We must investigate our assumptions and look at the time series aspect of sales
    '''

    # First I'm using a package called stumpy to compare two time series for similarities, hopefully clusters are similar


