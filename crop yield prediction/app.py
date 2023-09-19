import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('cropdata.csv')

new_df = df.copy()
new_df.drop(columns=['N','P','K'],axis=1,inplace=True)
x = new_df.drop(['label'], axis=1)
y = new_df['label']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

st.set_page_config(layout="centered")

page_bg_img = '''
    <style>
    .stApp {
    background-image: linear-gradient(blue,lightblue,lightyellow,orange);

    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)

rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(x_train,y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)

user_input = []
st.markdown('Enter the following data seperated by comma')

col = "Temperature, Humidity, PH, Rainfall"
st.write('<p style="color: Blue">Temperature, Humidity, PH, Rainfall</p>',unsafe_allow_html=True)

af = st.text_input('Enter Data',"35,80,7,460")
st.divider()

user_input = af.split(',')
user_input = [float(x) for x in user_input]

rf_pred = rf_model.predict([user_input])
lr_pred = lr_model.predict([user_input])
dt_pred = dt_model.predict([user_input])


for i in rf_pred:
    rf_pred = str(i)

for i in lr_pred:
    lr_pred = str(i)

for i in dt_pred:
    dt_pred = str(i)


final_pred = 0

if(rf_pred == lr_pred or rf_pred == dt_pred):
    final_pred = rf_pred
else:
    final_pred = lr_pred



st.write("Predicted Crop ")
st.subheader(final_pred.capitalize())
st.divider()

