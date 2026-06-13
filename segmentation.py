import streamlit as st
import numpy as np
import pandas as pd
import joblib

kmeans=joblib.load("kmeans_model.pkl")
scaler=joblib.load("scaler.pkl")
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment")

age=st.number_input("Age",min_value=18,max_value=100,value=50)
income=st.number_input("Income",min_value=0,max_value=1000000,value=5000)
recency=st.number_input("Recency",min_value=0,max_value=100,value=50)
numwebvisitsmonth=st.number_input("Number Web Visits Per Month",min_value=0,max_value=100,value=50)
totalspending=st.number_input("Total Spending",min_value=0,max_value=5000,value=50)
numstorepurchases=st.number_input("Number of Store Purchases",min_value=0,max_value=100,value=50)
Customersince=st.number_input("Customer Since",min_value=0,max_value=10000,value=5000)
numwebpurchases=st.number_input("Number of Web Purchases",min_value=0,max_value=100,value=50)

input_data=pd.DataFrame({
    "Income":[income],
    "Age":[age],
    "Recency":[recency],
    "NumWebVisitsMonth":[numwebvisitsmonth],
    "Total_Spending":[totalspending],
    "Customer_since":[Customersince],
    "NumStorePurchases":[numstorepurchases],
    "NumWebPurchases":[numwebpurchases],
})


input_scaled=scaler.transform(input_data)
if st.button("Predict Segment"):
    cluster=kmeans.predict(input_scaled)[0]
    st.success(f"predicited segment: cluster{cluster}")

  
