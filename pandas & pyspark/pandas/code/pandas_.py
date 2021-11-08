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

df = create_ts(20, ["2020-01-01", "2021-10-31"], [22,5], [5,100])
df = df.df()
print(df)