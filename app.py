# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:31:55 2024

@author: HP
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller

# Define function to load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Define function to preprocess data
def preprocess_data(data, target_column):
    # Convert date columns to numerical values
    date_columns = data.select_dtypes(include=['object']).columns
    for column in date_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Handle missing and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    y_imputed = y.fillna(y.mean())  # Impute missing target values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Streamlit app
def main():
    st.title("Coal Price Forecasting")

    # HTML-based interface
    st.markdown("""
        <style>
            .main {background-color: #161616; padding: 20px; border-radius: 10px;}
            .title {font-size: 24px; font-weight: bold;}
            .subtitle {font-size: 20px; margin-top: 10px;}
            .text {font-size: 16px; margin-top: 5px;}
        </style>
        <div class="main">
            <div class="title">Coal Price Forecasting</div>
            <div class="subtitle">Upload your dataset</div>
        </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data loaded successfully!")

        # Display first 5 rows
        st.subheader("First 5 rows of the data")
        st.write(data.head())

        # Connect with MySQL
        user = "root"
        pw = "1234"
        db = "project"
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

        data.to_sql("coaldata", con=engine, if_exists="replace", index=False)
        sql = "select * from coaldata;"
        coaldata = pd.read_sql_query(text(sql), engine.connect())
        st.write("Data loaded from MySQL")
        st.write(coaldata.head(10))

        # Exploratory Data Analysis
        st.subheader("Exploratory Data Analysis")
        columns = [
            'Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD',
            'Coal_RB_5700_FOB_London_Close_USD', 'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
            'Coal_India_5500_CFR_London_Close_USD', 'Price_WTI', 'Price_Brent_Oil',
            'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua', 'Price_All_Share',
            'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD',
            'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate'
        ]

        # Display summary statistics
        st.write("Summary Statistics")
        st.write(data.describe())

        # Display correlation heatmap
        st.write("Correlation Heatmap")
        corr_matrix = coaldata[columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Stationarity test using ADF
        st.subheader("Stationarity Test Using ADF")
        for column in columns:
            result = adfuller(coaldata[column].dropna())
            st.write(f"ADF Statistic for {column}: {result[0]}")
            st.write(f"p-value: {result[1]}")
            st.write("Critical values:")
            for key, value in result[4].items():
                st.write(f"\t{key}: {value:.3f}")

        # Forecasting
        st.subheader("Coal Price Forecasting with Random Forest Regressor")

        # Select target column for forecasting
        target_column = st.selectbox("Select the target column for forecasting", columns)

        if st.button("Train and Forecast"):
            X_train, X_test, y_train, y_test, scaler = preprocess_data(coaldata, target_column)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Generate forecasts
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate MAPE
            train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
            st.write(f"Train MAPE: {train_mape:.2f}%")
            st.write(f"Test MAPE: {test_mape:.2f}%")

            # Plot actual vs forecasted values
            st.write("Actual vs Forecasted Values")

            # Create a DataFrame for actual vs predicted values
            results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Forecasted': y_pred_test
            })

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df.index, results_df['Actual'], label="Actual", marker='o')
            ax.plot(results_df.index, results_df['Forecasted'], label="Forecasted", linestyle='--', marker='x')
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            ax.set_title("Actual vs Forecasted Values")
            ax.legend()
            st.pyplot(fig)

            # Display results as a table
            st.write("Results Table")
            st.write(results_df)

            # Calculate and display success rate
            success_rate = 100 - test_mape
            st.write(f"Success Rate of Forecasting: {success_rate:.2f}%")

if __name__ == '__main__':
    main()