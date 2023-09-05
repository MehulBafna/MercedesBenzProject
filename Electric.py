# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 23:31:07 2023

@author: Mehul
"""

import pandas as pd
import seaborn as sns
import plotly
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
plotly.offline.init_notebook_mode()
sns.set()

df_train = pd.read_csv(r'C:\Users\Mehul\Mercedes Task\train.csv')
df_test = pd.read_csv(r'C:\Users\Mehul\Mercedes Task\test.csv')

df_train.drop('State',axis=1,inplace=True)
df_test.drop('State',axis=1,inplace=True)

df_train.drop(columns = ['City', 'County', 'ZIP Code', 
                                       'ID', 'VIN (1-10)', 'Vehicle Location', 'DOL Vehicle ID', 
                                       'Electric Utility'],inplace=True)

df_test.drop(columns = ['City', 'County', 'ZIP Code', 
                                       'ID', 'VIN (1-10)', 'Vehicle Location', 'DOL Vehicle ID', 
                                       'Electric Utility'],inplace=True)

def target_drop(df):
    l=df[(df['Expected Price ($1k)']=='N/')].index
    for i in l:
        df.drop(i,axis=0,inplace=True)
    df['Expected Price ($1k)'] = pd.to_numeric(df['Expected Price ($1k)'], downcast='float')
    df['Expected Price ($1k)'] = df['Expected Price ($1k)']*1000
    df.rename(columns = {'Expected Price ($1k)':'Expected Price'},inplace=True)
    return df

target_drop(df_train)
target_drop(df_test)

df_train = df_train[df_train['Model Year'].notna()]
df_train = df_train[df_train['Make'].notna()]
df_test = df_test[df_test['Model Year'].notna()]
df_test = df_test[df_test['Make'].notna()]

df_train.fillna(df_train['Legislative District'].value_counts().index[0],inplace=True)
df_test.fillna(df_test['Legislative District'].value_counts().index[0],inplace=True)

df_train.groupby('Make')['Model']
for i in df_train.groupby('Make')['Model']:
    df_train.replace(list(i[1].unique()),[m+1 for m in range(len(list(i[1].unique())))],inplace=True)
    
df_test.groupby('Make')['Model']
for i in df_test.groupby('Make')['Model']:
    df_test.replace(list(i[1].unique()),[m+1 for m in range(len(list(i[1].unique())))],inplace=True)
    
df_train['Electric Vehicle Type'].replace(df_train['Electric Vehicle Type'].unique(),\
                                          [m+1 for m in range(len(df_train['Electric Vehicle Type'].unique()))],inplace=True)
df_train['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].replace(df_train['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique(),\
                                          [m+1 for m in range(len(df_train['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique()))],inplace=True)
df_train['Make'].replace(df_train['Make'].unique(),[m+1 for m in range(len(df_train['Make'].unique()))],inplace=True)

df_test['Electric Vehicle Type'].replace(df_test['Electric Vehicle Type'].unique(),\
                                          [m+1 for m in range(len(df_test['Electric Vehicle Type'].unique()))],inplace=True)
df_test['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].replace(df_test['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique(),\
                                          [m+1 for m in range(len(df_test['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique()))],inplace=True)
df_test['Make'].replace(df_test['Make'].unique(),[m+1 for m in range(len(df_test['Make'].unique()))],inplace=True)

X_train = df_train[[column for column in list(df_train.columns) if column!='Expected Price']]
X_test = df_test[[column for column in list(df_test.columns) if column!='Expected Price']]
y_train = df_train['Expected Price']
y_test = df_test['Expected Price']

LM = LinearRegression()
DTR = DecisionTreeRegressor()
RFR = RandomForestRegressor()
XGBR = XGBRegressor()
SVRR = SVR()
ETR = ExtraTreesRegressor()
ABR = AdaBoostRegressor()
GBR = GradientBoostingRegressor()
regressors = [LM,DTR,RFR,XGBR,SVRR,ETR,ABR,GBR]
Reg = ['LM','DTR','RFR','XGBR','SVRR','ETR','ABR','GBR']
R = [['Method','r2_score','RMSE','K-fold(score)']]
R2 = [['Method','r2_score','RMSE','K-fold(score)']]
for regressor in regressors:
    regressor.fit(X_train, y_train)
    reg = regressor.predict(X_train)
    reg2 = regressor.predict(X_test)

pickle.dump(SVRR, open('model.pkl','wb'))