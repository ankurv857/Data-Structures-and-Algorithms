# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 15:04:40 2016

@author: a0v0022
"""

#Basic Python Operations

a = 4.0
b = 5.0

print (a+b) 
print (a*b)
print (a**b)
print (a / b)
print (a % b)


print ("Welcome to Python!")

# user defined function in Python

def spam():
    eggs = 12
    return eggs

print (spam())

#import libraries
import pandas as pd
import numpy as np


#create vectors 
a = [1,2,3,4]
b = ['a','b','d','e']

#convert to dataframe

a = pd.DataFrame(a) 
b = pd.DataFrame(b) 

#create dataframe in python

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})


df

print (df.groupby(['A','B']).mean())


#create a dataset in a single go
df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))


df1
df1['A']


#concatenate two dataframes

c = pd.concat((a,b), axis = 1) 
print (c)

#import csv 

train = pd.read_csv("/Users/a0v0022/Documents/R/Codes Training/Data/train.csv")
print (train.head(50))
print (train.shape)
print (train.info())

#Track a datapoint 
print (train.iloc[1,1])
print (train.iloc[:,1])
print (train.iloc[1,:])

#Length of the data point
print (len(train.iloc[1,1]))

#unique values of a coulmn
print (train.iloc[:,3].unique())

#count the frequency at desired level
print (train.iloc[:,2].value_counts())

#Metadata of a Variable
print (train.iloc[:,1].astype(str))

#create a new variable 
train['Flag_V'] = 'Ankur'
#Rename a variable 
train = train.rename(columns = { 'Flag_V' : 'Flag_A'})

#Concatenate two columns
train['string1'] = train.iloc[:,1].astype(str).str.cat(train.iloc[:,2].astype(str) , sep = '|') 

#Strip and Lower by apply function
train['string1'] = train.iloc[:,4].apply(lambda x : x.strip().lower()) 

#map function to create criteria and use criteria for further application

criterion = train['City'].map(lambda x: x.startswith('D'))
print (train[criterion])
train[criterion & (train['Sales'] > 50)]

train.loc[criterion & (train['Sales'] > 50), 'Item' : 'Sales']

#define a function for an operation

def dep1(x):
    return 1 if 'delhi' in x.lower() else 0

train['Delhi_Flag'] = train.iloc[:,1].apply(dep1)

print (train)

#drop a column
train = train.drop(['string1','Flag_A','Delhi_Flag'], axis = 1 )

print (train.head(10))

#Create a flag with if-else condition

train['Flag'] =  [0 if x < 33 else 
                  1 if 33 <= x < 66 else 
                  2 for x in train.iloc[:,3] ] 
                  
print (train.head(10))    

#subset a dataframe

train_sub = train[(train.Item == "A1") & (train.City == "Delhi") ]
print (train_sub) 

train_sub = train.loc[:,['City','Sales']]
train_sub = train.iloc[:,(1,3)]
print (train_sub)

# sort a dataframe

print (train.sort('Sales').head(1))
print (train.sort('Sales').tail(1))

# Roll ups using Group by in Python

print (train.groupby('City').size().order )
print (train.groupby(['City','Item']).size().order )
print (train.groupby(['City','Item'])['Sales'].sum() )
print (train.groupby(['City','Item'])['Sales'].mean() )
print (train.groupby(['City','Item'])['Sales'].max() )
print (train.groupby(['City','Item'])['Sales'].min() )
#pivot table in python
#print (train.pivot_table(values = ['Sales'],index= ['City'], aggfunc = len) )

# contain and startwith operation in Python
train1 = train[train.Item.str.contains('A')]
train1 = train[train.Item.str.startswith('A')]
train1 = train1[train1.City == 'Delhi']
print (train1)

#Plots in Python
train1.Item.value_counts().sort_index().plot(kind='bar')
train1.plot(x='ID', y='Sales', kind='scatter') 


#create a dataframe and series
import numpy as np

data = np.reshape(np.random.randn(20),(10,2))
print (data)
X = pd.DataFrame(data)
print (X)

labels = np.random.randint(5, size = 10)
print (labels)
Y = pd.Series(labels)
print (Y)
 
# random split say 20:80 for both X and Y
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = .2 , random_state = 0) 
 
print (X_test) 


#export a dataset
train.to_csv('train.csv')


# merge in Python
price = pd.read_csv("/Users/a0v0022/Documents/R/Codes Training/Data/price.csv")
print (price)

train_merged = train.merge(price, left_on = 'Item' , right_on = 'Item' ,how = 'inner')
print (train_merged)


#create a matrix by vectorization for Text Mining

from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(min_df = 1) 
print (vectorizer)

corpus = train.iloc[:,1] 
print (corpus)
x = vectorizer.fit_transform(corpus) 
x.toarray() 
matrix1 = x.toarray() 
print (matrix1)
#vectorizer.vocabulary_.get('harvest')

# Creating a Inverse term  frequency Matrix for text mining

from sklearn.feature_extraction.text import TfidfTransformer 
transformer = TfidfTransformer() 
print (transformer) 
tfidf = transformer.fit_transform(matrix1) 
asd = tfidf.toarray() 
type(asd) 
asd.shape

# convert an array to dataframe

asd1 = pd.DataFrame(asd) 

# create a feature vector
features = vectorizer.get_feature_names() 
len(features) 
print (features)

# use the feature vector to create the column names correspondingly
asd1.columns = features 
print (asd1)

# Concatenate the created data frame with train
train_sub = pd.concat((train,asd1), axis = 1) 
print (train_sub.head())
