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

def make():
    pass


def startup_01():
    e_df = df_train.dropna(axis=1, how='any').reset_index(drop=True)
    hier_dict = create_hier_cluster_store(e_df)
    hk = hier_dict.keys()
    sales_df_dict = dict()
    sales_family_dict = dict()
    # Each store also has sales for each family of product, lets investigate these separate families to solve for each
    for cluster in hk:
        cl_list = hier_dict[cluster]
        sales_family_dict[cluster] = dict()
        for store in cl_list:
            store_nbr = store.loc[:, 'store_nbr'].values[0]
            temp_df = store.groupby('family')
            sales_family_dict[cluster][store_nbr] = dict()
            # print(temp_df.head())
            for f in list(store.loc[:, 'family'].unique()):
                g = temp_df.get_group(f)
                g.drop_duplicates(subset=['date'], keep='first', inplace=True)
                ra = pd.date_range(start=g['date'].min(), end=g['date'].max())
                missing_dates = ra.difference(g['date'])
                print(missing_dates)
                name = str(f).replace('/','&') + "_sales_" + str(store_nbr)+ "_"+str(cluster) +"_01" + ".parquet"
                print(name)
                sales_family_dict[cluster][store_nbr][f] = name
                g.to_parquet(name)
    with open('01_df_arch.pickle', 'wb') as handle:
        pickle.dump(sales_family_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def read_01():
    df_dict = dict()
    for file in os.listdir('.'):
        if file.endswith('01.parquet'):
            print(file)
            df_dict[file] = pd.read_parquet(file)
    return df_dict

def read_a1():
    df_dict = dict()
    for file in os.listdir('.'):
        if file.endswith('a1.parquet'):
            print(file)
            df_dict[file] = pd.read_parquet(file)
    return df_dict



# visualize
def visualize_ts(family, df, title):
    for c in ['sales']:
        scaler = QuantileTransformer()
        df.loc[:,'sales'] = scaler.fit_transform(df[['sales']])
        fig = px.histogram(df, x="date", y=c, histfunc="avg", title=title)
        fig.update_traces(xbins_size="M1")
        fig.update_xaxes(showgrid=True, ticklabelmode="period", dtick="M1", tickformat="%b\n%Y")
        fig.update_layout(bargap=0.1)
        fig.add_trace(go.Scatter(mode="markers", x=df["date"], y=df[c], name="daily", opacity=0.35))
        # fig.update_xaxes(
        #     rangeslider_visible=True,
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="YTD", step="year", stepmode="todate"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )
        im = "plots/"+(str(family)).replace('/','&')+"_QuantTrans_a1.png"
        fig.write_image(im)

# modeling

def each_family_total_df(cfg, df_dict):
    each_family = dict()
    for c in cfg.keys():
        for s in cfg[c].keys():
            for f in cfg[c][s].keys():
                curr_df = df_dict[cfg[c][s][f]].reset_index(drop=True)
                if f in each_family:
                    each_family[f] = ((pd.concat([each_family[f], curr_df])).sort_values(by=['date'],ascending=False)).reset_index(drop=True)
                else:
                    each_family[f] = (curr_df.sort_values(by=['date'], ascending=False)).reset_index(drop=True)
    return each_family

def lstm_trainer(df, w, batch):
    df.drop(columns=['id', 'family', 'city', 'state', 'date'], inplace=True)
    # store_nbr, onpromotion, store_type, cluster ; sales
    convert_dict = {'store_nbr':'category','onpromotion':'category','store_type':'category','cluster':'category','sales':float}
    df = df.astype(convert_dict)
    #encoder = OneHotEncoder()
    #df1 = df.loc[:, df.columns !='sales']
    #df1 = encoder.fit_transform(df1)
    scaler = MinMaxScaler()
    df['sales'] = scaler.fit_transform(df['sales'].shape(1, -1))
    #df.loc[:, df.columns !='sales'] = df1
    df = pd.get_dummies(df)
    print(type(df))
    train_X, test_X, train_y, test_y = train_test_split(df, df[:,'sales'], test_size=0.2, random_state=12, shuffle=True)
    train_generator = TimeseriesGenerator(train_X, train_y,length=w,sampling_rate=1,batch_size=batch)
    test_generator = TimeseriesGenerator(test_X, test_y, length=w, sampling_rate=1, batch_size=batch)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(w, 5), return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanSquaredError()])
    history = model.fit_generator(train_generator, epochs=50,
                                  validation_data=test_generator, shuffle=False,
                                  callbacks=[early_stopping])
    model.evaluate_generator(test_generator, verbose=0)
    predictions = model.predict_generator(test_generator)
    print(predictions.shape[0])

def what_stores_sales_zero(df):
    df_filtered = cur.groupby('store_nbr').filter(lambda x: (x['sales'].mean() == 0))
    stores = list(df_filtered.loc[:,'store_nbr'].unique())
    return stores





if __name__ == "__main__":

    # preprocess()
    #
    # df_train = pd.read_parquet('df_train.parquet').set_index('date',drop=False)
    # df_test = pd.read_parquet('df_test.parquet').set_index('date',drop=False)
    #
    # print(df_train.describe())
    # print(df_test.describe())
    #
    # # First let's do EDA of df_train
    # print("Columns")
    # print(df_train.columns)
    # print("Shape")
    # print(df_train.shape)
    # print("Date range")
    # dt1 = pd.to_datetime(max(df_train.loc[:, 'date']))
    # dt2 = pd.to_datetime(min(df_train.loc[:, 'date']))
    # datediff = (dt1 - dt2).days
    # print(dt2)
    # print(dt1)
    # print("Days between")
    # print(datediff)
    #
    # print(df_train.isnull().sum())
    # print(df_train.shape)
    # Drop columns that have nas, we can impute and use later
    #startup_01()
    # dfs_config = None
    # with open('01_df_arch.pickle', 'rb') as handle:
    #     dfs_config = pickle.load(handle)
    #
    # df_dict = read_01()
    # # for c in dfs_config.keys():
    # #     for s in dfs_config[c].keys():
    # #         for f in dfs_config[c][s].keys():
    # #             df = df_dict[dfs_config[c][s][f]]
    # #             print(dfs_config[c][s][f])
    # #             print(df.columns)
    # #             print((df.loc[:, 'sales']).describe())
    # # print(len(dfs_config.keys()))
    # # print(len(dfs_config[1].keys()))
    # # for i in dfs_config[1].keys():
    # #     print(len(dfs_config[1][i].keys()))
    # print(df_dict['BABY CARE_sales_51_17_01.parquet'].describe(include='all'))
    # each_fam = each_family_total_df(dfs_config,df_dict)
    print(tf.config.list_physical_devices('GPU'))
    # for i in each_fam.keys():
    #     name = str(i).replace('/','&')+"_a1"+".parquet"
    #     each_fam[i].to_parquet(name)
    # li = list(each_fam.keys())
    # with open('families_a1.pickle', 'wb') as handle:
    #     pickle.dump(li, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_dict = read_a1()

    with open('families_a1.pickle', 'rb') as handle:
        families_config = pickle.load(handle)

    # what stores do not sell certain products?
    # drop stores with mean 0, can recover later with full list of store_nbrs, and check what is missing in families
    family_store_zeros = dict()
    summ = dict()
    new_dfs = dict()
    for i in families_config:
        name = str(i).replace('/','&') +"_a1.parquet"
        cur = df_dict[name]
        family_store_zeros[i] = what_stores_sales_zero(cur)
        print(i)
        print(cur)
        print(family_store_zeros[i])
        print(cur.loc[:,'store_nbr'].nunique())
        new_dfs[i] = cur[~(cur['store_nbr'].isin(family_store_zeros[i]))]
        summ[i] = new_dfs[i].loc[:,'sales'].describe()
        print(new_dfs[i].loc[:,'store_nbr'].nunique())
        visualize_ts(i,new_dfs[i],i)
        # df_filtered = cur.groupby('store_nbr').filter(lambda x: (x['sales'].mean() >= 1))
        # # See what stores have deterministic sales
        # print(len(df_filtered)-len(cur))
        # # # look at each cluster
        # df_clustered = cur.groupby('cluster')
        # clusters = list(cur.loc[:,'cluster'].unique())
        # for j in clusters:
        #     clu = df_clustered.get_group(j)
        #     stores = list(clu.loc[:,'store_nbr'].unique())
        #     sto = clu.groupby('store_nbr').get_group(stores[0])
        #     fig = px.line(sto, x='date', y='sales')
        #     fig.show()
            #visualize_ts(i, sto, 'store '+str(stores[0])+" of cluster "+str(j)+" "+str(i))
            #visualize_ts(i, clu, 'cluster '+str(j)+" "+str(i))
        #visualize_ts(i,cur,'automotive')
        #break
    print(family_store_zeros)
    table = PrettyTable()
    table.add_column('Family', list(summ.keys()))
    table.add_column('Sales summary without irrelevant stores', list(summ.values()))

    print(table)
    print(df_dict.keys())
    visualize_ts('books', new_dfs['AUTOMOTIVE'], 'Post dropped AUTOMOTIVE')
    # with open('sales_stats.txt', 'w') as f:
    #     f.write(str(table))



    # Found some duplicated dates in dataset sales_family_dict
    # Currently is data stucture dict(cluster:dict(store:dict(family_sales:dataframe)))





    # for i in hk:
    #     stores_df = hier_dict[i]
    #     temp_dict = dict()
    #     for s in stores_df:
    #         store_nbr = s.loc[:, "store_nbr"].values[0]
    #         ssales = (pd.Series(s.loc[:,'sales'])).reset_index(drop=True)
    #         temp_dict[('storenum_'+str(store_nbr))] = ssales
    #         if 'date' not in temp_dict.keys():
    #             temp_dict['date'] = pd.Series(s.loc[:,'date'])
    #
    #     temp_df = pd.DataFrame(temp_dict)
    #     temp_df.set_index('date', inplace=True)
    #     sales_df_dict[i] = temp_df
    #     print(temp_df.corr())
    # print(sales_df_dict[1].shape)

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


