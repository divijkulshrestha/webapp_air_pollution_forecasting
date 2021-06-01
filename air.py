# make imports
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle 
import requests
import json
from datetime import date, datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import pytz

# app title
st.write("""
# Air Quality Prediction App
See forecasted **PM 2.5** Pollutant Levels in Indian cities for the next **48** hours!
""")

# sidebar title
st.sidebar.header('User Input Panel')


# details for calling OpenWeather API
API_KEY = 'acf6b2c17ce7439c51557ef26a8d7c54'
exclude = 'minute,alerts,daily'
units = 'metric'

# list of coordinates
lat_dict = {'New Delhi': 28.55, 'Ahmedabad': 23.066667, 'Bangalore': 12.949, 'Hyderabad': 17.465667, 'Chennai': 12.983333, 'Lucknow': 26.75}
long_dict = {'New Delhi': 77.10, 'Ahmedabad': 72.633333, 'Bangalore': 77.663, 'Hyderabad': 78.466667, 'Chennai': 80.166667, 'Lucknow': 80.883333}
lat = lat_dict.get('Ahmedabad')
lon = long_dict.get('Ahmedabad')

# function that returns API call URLs based on city selected 
def API_call(lat, lon):
    # weather API URL
    lat = str(lat)
    lon = str(lon)
    url = 'https://api.openweathermap.org/data/2.5/onecall?lat='+lat+'&lon='+lon+'&exclude='+exclude+'&units='+units+'&appid='+API_KEY
    # pollution API URL
    url_pol = 'http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat='+lat+'&lon='+lon+'&appid='+API_KEY
    return url, url_pol


# user data/derived data

today = date.today() 
tz = pytz.timezone('Asia/Calcutta') 

# datetime object for current day
now = datetime.now(tz)
# datetime objects for next 2 days
tom = now + timedelta(hours = 24)
dayafter = tom + timedelta(hours = 24)

#next 3 days
day1 = now.strftime("%d/%m")
day2 = tom.strftime("%d/%m")
day3 = dayafter.strftime("%d/%m")

# function to return dataframe as input for ML model
def user_input_features():
    # user selects day from drop down
    day = st.sidebar.selectbox("Select the day: ", ("Today (" + day1 + ")","Tomorrow (" + day2 + ")", "Day After ("+ day3 + ")"))
    
    # logic to calculate range for hour selection slider
    hour = now.hour
    if day == "Today (" + day1 + ")":
        min = now.hour
        max = 23
        day_selected = now
    elif day == "Tomorrow (" + day2 + ")":
        min = 0
        max = 23
        day_selected = tom
    else:
        min = 0
        max = now.hour
        day_selected = dayafter

    # user selects hour from slider
    hr = st.sidebar.slider("Select the hour (24 hour format): ", min, max, min)
    
    # hour = x hours from current hour (0 to 48) for OpenWeather API Call
    # logic to calculate hour based on selected date/time by user 
    if day_selected == now:
        hour = hr 
    elif day_selected == tom:
        hour = (24 - now.hour) + hr 
    else:
        hour = (48 - now.hour) + hr - 1

    # month derived from selected date/time
    month = day_selected.month

    city = st.sidebar.radio(
        "Select your city: ",
        ('Ahmedabad',
        'Bangalore',
        'Chennai',
        'Hyderabad',
        'Lucknow',
        'New Delhi',
        )
    )

    url, url_pol = API_call(lat_dict.get(city), long_dict.get(city))

    # calling OpenWeather APIs, extracting response in json format
    data = requests.get(url)
    data = data.json()
    data_pol = requests.get(url_pol)
    data_pol = data_pol.json()

    # weather attributes extracted
    temp = data['hourly'][hour]['temp']
    pres = data['hourly'][hour]['pressure']*0.75
    hum = data['hourly'][hour]['humidity']
    spd = data['hourly'][hour]['wind_speed']
    dir = data['hourly'][hour]['wind_deg']

    # pollution attributes extracted
    pm10 =data_pol['list'][hour]['components']['pm10'] 
    no2 = data_pol['list'][hour]['components']['no2']
    nh3 = data_pol['list'][hour]['components']['nh3']
    no = data_pol['list'][hour]['components']['no']
    co = data_pol['list'][hour]['components']['co']/1000
    so2 = data_pol['list'][hour]['components']['so2']

    # creating dataframe to return to ML model
    data = {
            'Temperature': temp,
            'Atmospheric_Pressure':pres, 
            'Relative_Humidity':hum,
            'Wind_Dir':dir, 
            'Wind_Speed':spd,
            'Month':month, 
            'Hour':hr, 
            'PM10':pm10, 
            'NO':no, 
            'NO2':no2,
            'NH3':nh3,
            'CO':co,
            'SO2':so2
            }
    features = pd.DataFrame(data, index=[0])
    return features, hour, city

df,h, city = user_input_features()

st.subheader('Forecasted Conditions at that Time')

st.write(df[['Temperature', 'Atmospheric_Pressure', 'Relative_Humidity', 'Wind_Dir', 'Wind_Speed']])
st.write(df[['PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2']])


url, url_pol = API_call(lat_dict.get(city), long_dict.get(city))
data_pol = requests.get(url_pol)
data_pol = data_pol.json()

# printing API forecasted PM 2.5 value
st.write(data_pol['list'][h]['components']['pm2_5'])
st.write("""
_Forecasted by [OpenWeather ML](https://openweathermap.org/)_
""")

# loading the model according to city
if city == 'New Delhi':
    with open("delhi_xgb.bin", 'rb') as f_in:
        model = pickle.load(f_in)

elif city == 'Chennai':        
    with open("chn_rf.bin", 'rb') as f_in:
        model = pickle.load(f_in)
    df = df.drop(['PM10'], axis = 1)

elif city == 'Ahmedabad':        
    with open("ahd_rf.bin", 'rb') as f_in:
        model = pickle.load(f_in)
    df = df.drop(['NH3'], axis = 1)
   
elif city == 'Lucknow':        
    with open("lck_rf.bin", 'rb') as f_in:
        model = pickle.load(f_in)
    df = df.drop(['PM10'], axis = 1)

elif city == 'Bangalore':        
    with open("ban_rf.bin", 'rb') as f_in:
        model = pickle.load(f_in)

else:        
    with open("hyd_lr.bin", 'rb') as f_in:
        model = pickle.load(f_in)

pred = model.predict(df)

# printing model prediction
st.subheader('Predicted PM 2.5 Level using our Model')
prediction = pred[0]
st.write(float("%.2f"% prediction))

# printing AQI category assoicated with prediction
st.subheader('Air Quality Index')
if(prediction >=0 and prediction <=12):
    st.write('Good')
elif(prediction >=12.1 and prediction <= 35.4):
    st.write('Moderate')
elif(prediction >=35.5 and prediction <= 55.4):
    st.write('Unhealty for Sensitive Groups')
elif(prediction >=55.5 and prediction <= 150.4):
    st.write('Unhealthy')
elif(prediction >=150.5 and prediction <= 250.4):
    st.write('Very Unhealthy')
else:
    st.write('Hazardous')    

# printing precautionary measures for each category
st.subheader('Precautionary Measures')
if(prediction >=0 and prediction <=12):
    st.write('None needed!')
elif(prediction >=12.1 and prediction <= 35.4):
    st.write('Unusually sensitive people should consider reducing prolonged or heavy exertion.')
elif(prediction >=35.5 and prediction <= 55.4):
    st.write('People with respiratory or heart disease, the elderly and children should limit prolonged exertion.')
elif(prediction >=55.5 and prediction <= 150.4):
    st.write('People with respiratory or heart disease, the elderly and children should avoid prolonged exertion completely; everyone else should limit prolonged exertion.')
elif(prediction >=150.5 and prediction <= 250.4):
    st.write('People with respiratory or heart disease, the elderly and children should avoid any outdoor activity completely; everyone else should avoid prolonged exertion.')
else:
    st.write('Everyone should avoid any outdoor exertion; people with respiratory or heart disease, the elderly and children should remain indoors.')  
