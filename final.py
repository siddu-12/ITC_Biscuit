import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from weatherbit.api import Api
API_KEY="38c214d675514b5684b9e0be34e4e2e7"

from sklearn.preprocessing import PolynomialFeatures
from PIL import Image

flag=0
image = Image.open('download.png')
st.image(image)

st.title("Sunfeast All rounder SNS - Variables Prediction")

factory = st.sidebar.selectbox(
    'Select factory',
    ("Ankit Biscuits Kattedan,Hyderabad.",)
)
city = 'Hyderabad'
state = 'Telangana'
country='INDIA'

model_name = st.sidebar.selectbox(
    'Select classifier',
    ('Polynomial Regression', 'Random Forest', 'XGBoost')
)

df=pd.read_csv("ITC_inc_temp.csv")
x=df.drop(['ABC','Water'],axis=1)
y=df.drop(['Gluten%','SV ml','Moisture','Temperature'],axis=1)

x_test= {"Gluten%":[],"SV ml":[],"Moisture":[],"Temperature":[]}

x_test1=float(st.number_input("Enter Gluten%"))
x_test2=float(st.number_input("Enter SV"))
x_test3=float(st.number_input("Enter Moisture"))

def get_temp(city,state):
    column_names = list(x.columns)
    api = Api(API_KEY)
    current_weather = api.get_current(city=city, state=state, country="IN")
    temperature=current_weather.get("temp")[0]['temp']
    return temperature
    # st.write(temperature)

temperature=get_temp(city,state)

x_test["Gluten%"].append(x_test1)
x_test["SV ml"].append(x_test2)
x_test["Moisture"].append(x_test3)
x_test["Temperature"].append(temperature)
    
x_final=pd.DataFrame(x_test)

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

y_pred=[[]]

if(model_name=='Polynomial Regression'):
    y_pred=polynomial_regression(x_final)

elif(model_name=='Random Forest'):
    y_pred=RandomForest_Regressor(x_final)

else:
    y_pred=XGBoost_Regressor(x_final)
    
flag=1

if st.button("Fetch Predictions"):
    st.write("The estimated value of ABC required will be {0:.3f}".format(y_pred[0][0]))
    st.write("The estimated value of water required will be {0:.3f}".format(y_pred[0][1]))

# Feedback Loop and updating database
feedback= st.selectbox(
    'Were the output values predicted accurately?',
    ('YES', 'NO')
)

if(feedback=='NO'):
    y_abc=float(st.number_input("Enter correct amt of ABC "))
    y_water=float(st.number_input("Enter correct amt of water"))
    y_correct=[]
    y_correct.append(y_abc)
    y_correct.append(y_water)
    y_final=[]
    y_final.append(y_correct)
    add=np.concatenate((y_final,x_final),axis=1)
    array_df = pd.DataFrame(add, columns=df.columns)

    if(y_abc!=0 and y_water!=0):
        key="empty"
        key=st.text_input("Enter password key to update")

        if(key=="ITC@123"):
            if st.button("Done"):
                st.write("Feedback updated successfully")

else:
    add=np.concatenate((y_pred,x_final),axis=1)
    array_df = pd.DataFrame(add, columns=df.columns)






