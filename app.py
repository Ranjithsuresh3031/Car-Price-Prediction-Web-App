import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

model=pk.load(open("Car_pp.pkl",'rb'))
st.header('R² Car Price Predictor')

df=pd.read_csv(r"C:\Users\ranji\OneDrive\Desktop\Rakesh project\cars_clean_data.csv")

Car_Name = st.selectbox('Select car brand', df['Car_Name'].unique())
Car_Model=st.selectbox('Select car Model',df['Car_Model'].unique())
Year=st.slider('Car Manufacture Year',1890,2024)
Kilometer=st.slider("No of km Drive	",10,300000)
Owner=st.selectbox("Number of Owner",df['Owner'].unique())
Fuel=st.selectbox("Fuel Type",df['Fuel'].unique())
Transmission=st.selectbox("Transmission Type",df['Transmission'].unique())
Type=st.selectbox("Car Types",df['Type'].unique())

if st.button("Predict"):
    inputmodel=pd.DataFrame([[Car_Name,Car_Model,Year,Kilometer,Owner,Fuel,Transmission,Type]],columns=['Car_Name','Car_Model','Year','Kilometer','Owner','Fuel','Transmission','Type'])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    inputmodel['Car_Name'].replace(['Maruti', 'Hyundai', 'Tata', 'Renault', 'Honda', 'Ford', 'Datsun','Toyota', 'Mahindra', 'Nissan', 'KIA', 'Volkswagen', 'Skoda','BMW', 'Jeep', 'MG'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],inplace=True)
    inputmodel['Car_Model'].replace(['S PRESSO', 'Xcent', 'Safari', 'Vitara Brezza', 'Tiago', 'Swift',
       'i20', 'Kwid', 'Grand i10', 'IGNIS', 'Brio', 'Elite i20', 'City',
       'Baleno', 'WR-V', 'Amaze', 'Alto 800', 'Celerio', 'Ecosport',
       'Ciaz', 'Redi Go', 'TIAGO NRG', 'Santro Xing', 'FREESTYLE',
       'Dzire', 'Alto', 'NEW SANTRO', 'Alto K10', 'Endeavour',
       'Swift Dzire', 'Wagon R 1.0', 'GRAND I10 NIOS', 'Celerio X',
       'URBAN CRUISER', 'XUV500', 'Verna', 'VENUE', 'NEXON',
       'KUV 100 NXT', 'YARIS', 'XUV 3OO', 'TRIBER', 'Tucson New',
       'TUV300', 'Glanza', 'Eeco', 'Duster', 'i10', 'MAGNITE', 'SONET',
       'Ertiga', 'Jazz', 'SELTOS', 'Ameo', 'Kiger', 'Accord', 'NEW I20',
       'ALTROZ', 'A Star', 'Ritz', 'Micra', 'Eon', 'Creta', 'Bolero',
       'Etios Liva', 'New Wagon-R', 'Micra Active', 'Harrier', 'TIGOR',
       'PUNCH', 'Polo', 'Camry', 'Corolla Altis', 'Civic', 'Vento',
       'S Cross', 'Octavia', 'i20 Active', 'New Elantra', 'BR-V', 'AURA',
       'Thar', 'Zen Estilo', 'NEW I20 N LINE', 'Hexa', 'XL6', 'CRV',
       'Innova', 'Rapid', 'Go', 'Wagon R Stingray', 'TIGUAN', 'Etios',
       'Zest', 'New Figo', 'Kuv100', 'SLAVIA', 'Scorpio', 'Terrano',
       'TAIGUN', 'Captur', 'XUV700', 'Sonata', 'BOLERO NEO', 'BREZZA',
       'Go Plus', 'ALCAZAR', 'Compass', 'Innova Crysta', 'CARENS',
       'KUSHAQ', 'Jetta', 'Pulse', 'Figo Aspire', 'Wagon R',
       'TUV 300 PLUS', 'HECTOR PLUS', 'Bolt', 'HECTOR', 'T-ROC', 'OMNI E',
       'GRAND CHEROKEE', 'Fortuner', 'MARAZZO', 'Sunny'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],inplace=True)
    inputmodel['Fuel'].replace(['PETROL', 'DIESEL', 'CNG', 'LPG'],[1,2,3,4],inplace=True)
    inputmodel['Type'].replace(['HatchBack', 'Sedan', 'SUV', 'Lux_SUV', 'Lux_sedan'],[1,2,3,4,5],inplace=True)
    inputmodel['Transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)
    st.write(inputmodel)
    car_price=model.predict(inputmodel)

    st.markdown('car price is going to be'+str(car_price))