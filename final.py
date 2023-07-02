import streamlit as st
import numpy as np
import pandas as pd
import pickle

from weatherbit.api import Api
from sqlalchemy.sql import text
API_KEY="38c214d675514b5684b9e0be34e4e2e7"

from sklearn.preprocessing import PolynomialFeatures

flag=0
st.title("ITC Biscuit Manufacturing Analytics")
df=pd.read_csv("ITC_inc_temp.csv")

# SQL Data handling section 
# conn = st.experimental_connection('itc_biscuit',type = 'sql',autocommit = True)
# conn.session.execute(text('''drop table if exists maida3'''))
# conn.session.execute(text('''create table if not exists maida3(
#     abc float,
#     water float,
#     gluten float,
#     sv float,
#     moisture float,
#     temperature float
# )'''))

# for index, row in df.iterrows():
#     print(row['ABC'])
#     conn.session.execute(text('''insert into maida3(abc,water,gluten,sv,moisture,temperature) values(:abc,:water,:gluten,:sv,:moisture,:temperature)'''),
#                          params = {'abc':row['ABC'],'water':row['Water'],'gluten':row['Gluten%'],'sv':row['SV ml'],'moisture':row['Moisture'],'temperature':row['Temperature']})

# conn.session.commit()
# tbl = conn.query('select * from maida3')
# print('----------')
# print(tbl)

# /////...... ML section begins

x=df.drop(['ABC','Water'],axis=1)
y=df.drop(['Gluten%','SV ml','Moisture','Temperature'],axis=1)

left, right = st.columns(2)
city = left.text_input('Enter City')
state = right.text_input('Enter City State')
country='INDIA'

model_name = st.sidebar.selectbox(
    'Select classifier',
    ('Polynomial Regression', 'Random Forest', 'XGBoost')
)

x_test= {"Gluten%":[],"SV ml":[],"Moisture":[],"Temperature":[]}

x_test1=float(st.number_input("Enter Gluten%"))
x_test2=float(st.number_input("Enter SV"))
x_test3=float(st.number_input("Enter Moisture"))

temperature=0

if st.button('Fetch'):
    column_names = list(x.columns)
    api = Api(API_KEY)
    current_weather = api.get_current(city=city, state=state, country="IN")
    temperature=current_weather.get("temp")[0]['temp']
    # st.write(temperature)

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

if(x_test1!=0 and x_test2!=0 and x_test3!=0):
    if(model_name=='Polynomial Regression'):
        y_pred=polynomial_regression(x_final)

    elif(model_name=='Random Forest'):
        y_pred=RandomForest_Regressor(x_final)

    else:
        y_pred=XGBoost_Regressor(x_final)
    
    flag=1

if(flag==1):
    st.write("The estimated value of AMC required will be {0:.3f}".format(y_pred[0][0]))
    st.write("The estimated value of water required will be {0:.3f}".format(y_pred[0][1]))
    
# Feedback Loop and updating database
feedback= st.selectbox(
    'Were the output values predicted accurately?',
    ('YES', 'NO')
)

if x_final['Gluten%'][0]==0 or x_final['SV ml'][0] == 0 or x_final['Moisture'][0] == 0:
    print('No Response')
else:
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
        # array_df.to_csv('itc_file.csv',index = False,mode = 'a',header = False)

    else:
        add=np.concatenate((y_pred,x_final),axis=1)
        array_df = pd.DataFrame(add, columns=df.columns)
        # array_df.to_csv('itc_file.csv',index = False,mode = 'a',header = False)
    
    
