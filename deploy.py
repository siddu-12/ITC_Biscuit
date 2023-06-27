import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title("ITC Biscuit Manufacturing Analytics")

df=pd.read_csv("ITC.csv")

lr=LinearRegression()

x=df.drop(['ABC','Water'],axis=1)
y=df.drop(['Gluten','SV','Moisture'],axis=1)

poly = PolynomialFeatures(degree=1)
x_poly = poly.fit_transform(x)

lr.fit(x_poly,y)

x_test1=float(st.number_input("Enter Gluten"))
x_test2=float(st.number_input("Enter SV"))
x_test3=float(st.number_input("Enter Moisture"))
x_test=[]
x_final=[]
x_test.append(x_test1)
x_test.append(x_test2)
x_test.append(x_test3)

x_final.append(x_test)

x_test_poly=poly.fit_transform(x_final)
y_pred_poly=lr.predict(x_test_poly)

st.write("The estimated value of AMC required will be {0:.3f}".format(y_pred_poly[0][0]))
st.write("The estimated value of water required will be {0:.3f}".format(y_pred_poly[0][1]))


classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'Polynomial Regression', 'Random Forest', 'XGBoost')
)


