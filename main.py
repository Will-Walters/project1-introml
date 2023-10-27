import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import stumpy
import numpy as np
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport
import itertools




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
    # checking if windows of both time series correponsend T_A=A is different than T_A=B
    # matrix profile is literally the euclidean distance between two subsequences in a sequence
    # for each window sized subsequence (window rows of time series) calculate the euclidean distance all other possible
    # subsequences in time series; therefore distance of 0 means it is exactly the same time series subsequence,
    # patterns that are similar have a mp closer to 0
    # used to detect motifs(repeating patterns, reoccuring low matrix profile) and anomalies(large matrix profiles outliers)
    mp = stumpy.stump(T_A=A, m=window, T_B=B, ignore_trivial=False)
    A_motif_index = mp[:,0].argmin()
    plt.xlabel("Subsequence")
    plt.ylabel("Matrix Profile")
    plt.scatter(A_motif_index,
                mp[A_motif_index, 0],
                c='red',
                s=100)
    plt.show()

def create_store_specific_df(large_df):
    # Returns a list of each stores df
    return [df for _, df in large_df.groupby('store_nbr')]

def create_hier_cluster_store(large_df):
    unique_clusters = list(large_df.loc[:,'cluster'].unique())
    new_df = large_df.groupby(by=["cluster"])
    tree = dict()
    for cluster in unique_clusters:
        curr = create_store_specific_df(new_df.get_group(cluster))
        tree[cluster] = curr
    # Returns a dictionary(cluster_number : list of store dfs)
    return tree

def check_1d_store():
    # Check
    pass
def check_timeseries(t):
    # Check series for na and impute
    nas = sum(pd.isna(t))
    if nas > 0:
        print("Warning")
        print(nas)
        print("Missing values")
        print("Imputing by forwardfill")
        t.ffill(inplace=True)
        print("Sanity check")
        print(sum(pd.isna(t)))
    else:
        print("No missing values")

def get_cluster_corr_mean(df, cluster_num):
    # Get details on avg correlation between store sales in same cluster
    pass

def get_cov_matrix(df):
    # Get matrix of all store numbers
    split_data = create_hier_cluster_store(df)
    for cluster in split_data.keys():
        curr = split_data[cluster]


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

    print(df_train.isnull().sum())
    print(df_train.shape)
    # Drop columns that have nas, we can impute and use later
    e_df = df_train.dropna(axis=1, how='any')
    hier_dict = create_hier_cluster_store(e_df)
    hk = hier_dict.keys()
    sales_df_dict = dict()
    for i in hk:
        stores_df = hier_dict[i]
        temp_dict = dict()
        for s in stores_df:
            store_nbr = s.loc[:, "store_nbr"].values[0]
            ssales = (pd.Series(s.loc[:,'sales'])).reset_index(drop=True)
            temp_dict[('storenum_'+str(store_nbr))] = ssales
            if 'date' not in temp_dict.keys():
                temp_dict['date'] = pd.Series(s.loc[:,'date'])
            
        temp_df = pd.DataFrame(temp_dict)
        temp_df.set_index('date', inplace=True)
        sales_df_dict[i] = temp_df
        print(temp_df.corr())
    print(sales_df_dict[1].shape)

    # for i in list(itertools.combinations(hk, 2)):
    #     print(i)
    #     s1, s2 = i
    #     df1 = sales_df_dict[s1]
    #     df2 = sales_df_dict[s2]
    #     print(df1.corrwith(df2))




    #get_cov_matrix(e_df)


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

    # profile = ProfileReport(df_train, tsmode=True, sortby="Date Local")
    # profile.to_file('profile_report.html')

