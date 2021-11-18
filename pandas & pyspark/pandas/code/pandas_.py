#@author ankur

import pandas as pd
import numpy as np
import os

class create_ts():
    def __init__(self, num_ts, date_range, mat_dims, mat_vals, seed = 0):
        self.num_ts = num_ts
        self.min_date, self.max_date = date_range
        self.rows, self.cols = mat_dims
        self.min_val, self.max_val = mat_vals
        self.df()
    
    def df(self):
        df = pd.DataFrame()
        for num_ts in range(self.num_ts):
            x_df = pd.DataFrame(np.random.random((self.rows, self.cols)))
            y_df = pd.DataFrame(np.random.randint(self.min_val, self.max_val, (self.rows, 1)))
            date_df = pd.DataFrame(pd.date_range(start = self.min_date, end = self.max_date, freq = 'M'))
            x_df.columns = [f'col_{i}' for i in range(self.cols)]
            y_df.columns = ['y']
            date_df.columns = ['date']
            df_temp = pd.concat([date_df, x_df, y_df], axis = 1)
            df_temp['ts_id'] = num_ts
            df = pd.concat([df, df_temp], axis = 0).reset_index(drop=True)
        return df

#####################################Run this when the data needs to be changed#####################################

# df = create_ts(20, ["2020-01-01", "2021-10-31"], [22,5], [5,100])
# df = df.df()
# df.to_csv('../data/testdf.csv', index = False)
# print(df)

class create_features():
    def __init__(self, file):
        self.file = file
        data = self.read_data()
        data = self.common_features(data)
        data = self.offset_features(data)

    def read_data(self):
        return pd.read_csv(self.file)

    def common_features(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        return data

    def offset_features(self, data):
        for i in range(3):
            data['lag_' + str(i)] = data.groupby('ts_id')['col_0'].shift(i+1).fillna(method = 'bfill')
        data['offset'] = data.groupby('ts_id')['y'].shift(3).rolling(2).mean().fillna(method = 'bfill')
        print(data.head(20))

features = create_features('../data/testdf.csv')
# print(features.data.head(2))