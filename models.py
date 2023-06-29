import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

df=pd.read_csv("ITCFinal.csv - Copy of Sheet1.csv")

le = preprocessing.LabelEncoder()
df['Slot']=le.fit_transform(df['Timing'])
df.drop(['Timing'],axis=1,inplace=True)

x=df.drop(['ABC','Water'],axis=1)
y=df.drop(['Gluten%','SV ml','Moisture','Slot'],axis=1)

# Polynomial Regression model
lr=LinearRegression()
poly = PolynomialFeatures(degree=1)
x_poly = poly.fit_transform(x)

lr.fit(x_poly,y)
filename = 'poly_regmodel.sav'
pickle.dump(lr, open(filename, 'wb'))

# Random Forest
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x, y)
filename = 'randfor_regmodel.sav'
pickle.dump(regr, open(filename, 'wb'))

# XGBoost Regressor
regressor=xgb.XGBRegressor(max_depth=5, n_estimators=500, eval_metric='rmsle')
regressor.fit(x, y)
filename = 'xgboost_regmodel.sav'
pickle.dump(regressor, open(filename, 'wb'))
