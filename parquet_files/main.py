import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == "__main__":
    # df_dict = dict()
    # files = []
    # for file in os.listdir('.'):
    #     if file.endswith('.parquet'):
    #         files.append(file)
    #         print(file)
    #         df_dict[file] = pd.read_parquet(file)
    #
    # for file in files:
    #     curr = df_dict[file]
    #     print(file)
    #     print(curr.describe())

    ## combine datasets
    # df_train1 = df_dict['train.parquet'].merge(df_dict['oil.parquet'], on='date', how='left')
    # df_train1 = df_train1.merge(df_dict['holidays_events.parquet'], on='date', how='left')
    # df_train1 = df_train1.merge(df_dict['stores.parquet'], on='store_nbr', how='left')
    # df_train1 = df_train1.merge(df_dict['transactions.parquet'], on=['date', 'store_nbr'], how='left')
    # df_train = df_train1.rename(columns={"type_x": "holiday_type", "type_y": "store_type"})
    # df_train.to_parquet('df_train.parquet')
    #
    # df_test1 = df_dict['test.parquet'].merge(df_dict['oil.parquet'], on='date', how='left')
    # df_test1 = df_test1.merge(df_dict['holidays_events.parquet'], on='date', how='left')
    # df_test1 = df_test1.merge(df_dict['stores.parquet'], on='store_nbr', how='left')
    # df_test1 = df_test1.merge(df_dict['transactions.parquet'], on=['date', 'store_nbr'], how='left')
    # df_test = df_test1.rename(columns={"type_x": "holiday_type", "type_y": "store_type"})
    # df_test.to_parquet('df_test.parquet')

    df_train = pd.read_parquet('df_train.parquet')
    df_test = pd.read_parquet('df_test.parquet')
    #
    # print(df_train.loc[:,'store_nbr'].nunique())
    #
    print(df_test.loc[:,'onpromotion'].head())
    dt1 = pd.to_datetime(max(df_train.loc[:, 'date']))
    dt2 = pd.to_datetime(min(df_train.loc[:, 'date']))
    datediff = (dt1-dt2).days
    print(datediff)
    print(df_train.loc[:,'holiday_type'].unique())