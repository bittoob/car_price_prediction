import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib

train_data = pd.read_csv('final_train.csv')
test_data = pd.read_csv('final_test.csv')


train_data = train_data.drop(['Unnamed: 0'], axis = 1)

test_data = test_data.drop(['Unnamed: 0'], axis = 1)


X = train_data.loc[:,['Year', 'Kilometers_Driven', 'Owner_Type', 'Seats',
       'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)', 'Location_Bangalore',
       'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
       'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
       'Location_Kolkata', 'Location_Mumbai', 'Location_Pune',
       'Fuel_Type_Diesel', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',
       'Transmission_Manual']]


y = train_data.loc[:,['Price']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25)

rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)
y_pred= rf_reg.predict(X_test)

pickle.dump(rf_reg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))

print("rf_reg model output",rf_reg.predict([[2010,72000,1,5.0,22.6,998,58.16,0,0,0,0,0,0,0,0,0,1,0,0,0,1]]))

print("saved model output",model.predict([[2010,72000,1,5.0,22.6,998,58.16,0,0,0,0,0,0,0,0,0,1,0,0,0,1]]))

