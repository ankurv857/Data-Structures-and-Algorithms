#author @ankurverma

import pandas as pd
import numpy as np
import os

#create a dataframe
class createdf():
    def __init__(self, mat_dims, mat_vals, seed = 0):
        self.rows, self.cols = mat_dims
        self.min_val, self.max_val = mat_vals
        self.df()
    
    def df(self):
        x_matrix = np.random.random((self.rows, self.cols))
        y_matrix = np.random.randint(self.min_val, self.max_val, (self.rows, 1))
        x_df, y_df =  pd.DataFrame(x_matrix), pd.DataFrame(y_matrix)
        x_df.columns = [f'col_{i}' for i in range(self.cols)]
        y_df.columns = ['y']
        return x_df, y_df

#Single Layer Perceptron
class SLPerceptron():
    def __init__(self, input_data, lr = 0.01, epoch = 5, batch_size = 2):
        self.X, self.y = input_data
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight = list(np.random.random_sample(size = list(self.X.shape)[1]))
        self.fit()

    def fit(self):
        for e in range(self.epoch):
            curr_error = self.loss(self.predict(), self.y)
            if curr_error == 0 :
                break
            for index, feature in enumerate(list(self.X.T.to_numpy())):
                gradient = self.gradient(feature)
                self.weight[index] = self.weight[index] + self.lr * self.weight[index] * gradient
            print(self.predict(), self.y, self.weight) ; exit()


    def predict(self):
        preds = []
        x = list(self.X.to_numpy())
        for item in x:
            pred = np.dot(item, self.weight)
            preds.append(pred)
        return preds

    def gradient(self, feature):
        return np.mean([i for i in list(feature)])

    def loss(self, predict, actual):
        actual = list(actual['y'].to_numpy())
        error =  np.mean([abs(i - j) for i,j in zip(actual, predict)])
        return error

df = createdf([25,5], [0,2])
x_df, y_df = df.df()
model = SLPerceptron(df.df())

#Logistic Regression
class logisticreg():
    def __init__(self, input_data, lr = 0.01, epoch = 5):
        self.X, self.y = input_data
        self.lr = lr
        self.epoch = epoch
        self.weight = list(np.random.random_sample(size = list(self.X.shape)[1]))
        self.fit()

    def fit(self):
        for e in range(self.epoch):
            curr_error = self.loss(self.predict(), self.y)
            if curr_err == 0:
                break
            for index, feature in enumerate(list(self.X.T.to_numpy())):
                gradient = self.gradient(feature)
                self.weight[index] = self.weight[index] + self.lr * self.weight[index] * gradient
            print(self.predict(), self.y, self.weight) ; exit()

    def predict(self):
        preds = []
        x = list(self.X.to_numpy())
        for item in x:
            pred = self.sigmoid(np.dot(item, self.weight))
            preds.append(pred)
        return preds

    def loss(self, actual, predict):
        actual = list(actual['y'].to_numpy())
        error = np.mean([(-i * np.log(j) - (1-i) * np.log(1-j)) for i, j in zip(actual, predict)])
        return error

    def gradient(self, feature):
        return np.mean([i for i in list(feature)])

    def sigmoid(self, a):
        return 1/(1 + np.exp(-a))




