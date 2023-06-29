import streamlit as st
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

st.title("ITC Biscuit Manufacturing Analytics")

df=pd.read_csv("ITCFinal.csv - Copy of Sheet1.csv")

le = preprocessing.LabelEncoder()
df['Slot']=le.fit_transform(df['Timing'])
df.drop(['Timing'],axis=1,inplace=True)

x=df.drop(['ABC','Water'],axis=1)
y=df.drop(['Gluten%','SV ml','Moisture','Slot'],axis=1)

x_test1=float(st.number_input("Enter Gluten%"))
x_test2=float(st.number_input("Enter SV"))
x_test3=float(st.number_input("Enter Moisture"))
x_test4= st.selectbox(
    'Which slot of the day is the machine running in?',
    (0,1,2)
)
column_names = list(x.columns)

x_test= {"Gluten%":[],"SV ml":[],"Moisture":[],"Slot":[]}

x_test["Gluten%"].append(x_test1)
x_test["SV ml"].append(x_test2)
x_test["Moisture"].append(x_test3)
x_test["Slot"].append(x_test4)

x_final=pd.DataFrame(x_test)

flag=0

model_name = st.sidebar.selectbox(
    'Select classifier',
    ('Polynomial Regression', 'Random Forest', 'XGBoost')
)

def polynomial_regression(x_final):
    loaded_model1 = pickle.load(open('poly_regmodel.sav', 'rb'))
    poly = PolynomialFeatures(degree=1)
    x_final_poly = poly.fit_transform(x_final)
    y_pred_poly= loaded_model1.predict(x_final_poly)

    return y_pred_poly

def RandomForest_Regressor(x_final):
    loaded_model2 = pickle.load(open('randfor_regmodel.sav', 'rb'))
    y_pred_randf=loaded_model2.predict(x_final)

    return y_pred_randf

def XGBoost_Regressor(x_final): 
    loaded_model3 = pickle.load(open('xgboost_regmodel.sav', 'rb'))
    y_pred_xgbr=loaded_model3.predict(x_final)

    return y_pred_xgbr

if(model_name=='Polynomial Regression'):
    y_pred=polynomial_regression(x_final)

elif(model_name=='Random Forest'):
    y_pred=RandomForest_Regressor(x_final)

else:
    y_pred=XGBoost_Regressor(x_final)

if(x_test1!=0 and x_test2!=0 and x_test3!=0):
    flag=1

if(flag==1):
    st.write("The estimated value of AMC required will be {0:.3f}".format(y_pred[0][0]))
    st.write("The estimated value of water required will be {0:.3f}".format(y_pred[0][1]))
    
flag=2
feedback='YES'

if(flag==2):
    feedback= st.selectbox(
        'Were the output values predicted accurately?',
        ('YES', 'No')
    )

if(feedback=='YES'):
    add=np.concatenate((x_final,y_pred),axis=1)
    array_df = pd.DataFrame(add, columns=df.columns)

else:
    y_abc=float(st.number_input("Enter correct amt of ABC "))
    y_water=float(st.number_input("Enter correct amt of water"))
    y_correct=[]
    y_correct.append(y_abc)
    y_correct.append(y_water)
    y_final=[]
    y_final.append(y_correct)
    add=np.concatenate((x_final,y_final),axis=1)
    array_df = pd.DataFrame(add, columns=df.columns)
    
