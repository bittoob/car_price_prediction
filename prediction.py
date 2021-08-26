import streamlit as st
import numpy as np
import sklearn
import pickle

@st.cache
def load_model(file):
    model = pickle.load(open(file,"rb"))
    return model

lrmodel = load_model('model.pkl')

st.title('Used car Price Prediction')
st.write("Prediction")

n1 = st.number_input("Enter year ",min_value = 1998.00, max_value = 2019.00 , step = 1.00)
n2 = st.number_input("Enter Kilometer ",min_value = 171, max_value = 6500000 , step = 500)
n3 = st.number_input("Enter owner_Type for self : 1 , for both : 2 otherwise : 3",min_value = 1, max_value = 3 , step = 1)
n4 = st.number_input("Enter no. of seats",min_value = 2.0, max_value = 10.0, step = 1.0)
n5 = st.number_input("Enter Mileage",min_value = 0.00, max_value = 33.54, step = 1.00)
n6 = st.number_input("Enter Engine in CC",min_value = 624, max_value = 5998, step = 1)
n7 = st.number_input("Enter Power in bhp",min_value = 34.20, max_value = 560.00, step = 1.00)
n8 = st.number_input("Enter Location_Bangalore(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n9 = st.number_input("Enter Location_Chennai(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n10 = st.number_input("Enter Location_Coimbatore(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n11 = st.number_input("Enter Location_Delhi(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n12 = st.number_input("Enter Location_Hyderabad(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n13 = st.number_input("Enter Location_JAipur(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n14 = st.number_input("Enter Location_Kochi(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n15 = st.number_input("Enter Location_Kolkata(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n16 = st.number_input("Enter Location_Mumbai(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n17 = st.number_input("Enter Location_Pune(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n18 = st.number_input("Enter Fuel_Type_Diesel(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n19 = st.number_input("Enter Fuel_Type_LPG(0 and 1) ",min_value = 0, max_value = 1, step = 1)
n20 = st.number_input("Enter Fuel_Type_Petrol(0 and 1) ",min_value = 0, max_value = 1, step = 1)

n21 = st.number_input("Enter Transmission_manual(0 and 1) ",min_value = 0, max_value = 1, step = 1)

prediction = lrmodel.predict([[n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21]])
price = np.round(prediction[0],2)
status = st.button("Click to see Price")
if status:
    st.write("Predicted Price "+str(price))
    




