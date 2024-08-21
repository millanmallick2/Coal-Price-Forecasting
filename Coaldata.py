# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:53:37 2024

@author: HP
"""

#import Module
# To test the unit root in the dataset
import dtale  # auto EDA
from autoviz.AutoViz_Class import AutoViz_Class  # auto EDA
import sweetviz as sv  # Auto EDA
from feature_engine.outliers import Winsorizer  # for Outlier Treatment
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as ts
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

data = pd.read_excel(r"C:\Users\HP\Desktop\360 DigiTMG\Project\1. Coal Price Forecasting\Project Initial Documents\Data Set\Final Dataset.xlsx")

user = 'root'  # user name
pw = '1234'  # password
db = 'project'  # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database
data.to_sql('coaldata', con=engine, if_exists='replace', chunksize=1000, index=False)

# loading data from database
sql = 'select * from coaldata'

coaldata = pd.read_sql_query(text(sql), con=engine.connect())

print(coaldata)
coaldata.shape
coaldata.info
coaldata.describe()

# Data Preprocessing
coaldata.isna().sum()

# AUTO EDA
# Sweetviz
pip install sweetviz
s = sv.analyze(coaldata)
s.show_html()

# EDA
# 1st Business Moment
# mean
means = coaldata.iloc[:, 1:].mean()

# median
median = coaldata.iloc[:, 1:].median()

# mode
mode = coaldata.iloc[:, 1:].mode()

# 2nd Business Moment
# Variance
variance = coaldata.iloc[:, 1:].var()

# standard Deviation
Deviation = coaldata.iloc[:, 1:].std()

# 3rd Business Moment
# skewness
skewness = coaldata.iloc[:, 1:].skew()

# 4th Business Moment
# kurtsis
kurtusis = coaldata.iloc[:, 1:].kurt()


# Draw Histrogram
plt.hist(coaldata.Coal_RB_4800_FOB_London_Close_USD)
plt.hist(coaldata.Coal_RB_5500_FOB_London_Close_USD)
plt.hist(coaldata.Coal_RB_5700_FOB_London_Close_USD)
plt.hist(coaldata.Coal_RB_6000_FOB_CurrentWeek_Avg_USD)
plt.hist(coaldata.Coal_India_5500_CFR_London_Close_USD)
plt.hist(coaldata.Price_WTI)
plt.hist(coaldata.Price_Brent_Oil)
plt.hist(coaldata.Price_Dubai_Brent_Oil)
plt.hist(coaldata.Price_ExxonMobil)
plt.hist(coaldata.Price_Shenhua)
plt.hist(coaldata.Price_All_Share)
plt.hist(coaldata.Price_Mining)
plt.hist(coaldata.Price_LNG_Japan_Korea_Marker_PLATTS)
plt.hist(coaldata.Price_ZAR_USD)
plt.hist(coaldata.Price_Natural_Gas)
plt.hist(coaldata.Price_ICE)
plt.hist(coaldata.Price_Dutch_TTF)
plt.hist(coaldata.Price_Indian_en_exg_rate)

# Using SeaBorn
sns.distplot(coaldata.Coal_RB_4800_FOB_London_Close_USD)
sns.distplot(coaldata.Coal_RB_5500_FOB_London_Close_USD)
sns.distplot(coaldata.Coal_RB_5700_FOB_London_Close_USD)
sns.distplot(coaldata.Coal_RB_6000_FOB_CurrentWeek_Avg_USD)
sns.distplot(coaldata.Coal_India_5500_CFR_London_Close_USD)
sns.distplot(coaldata.Price_WTI)
sns.distplot(coaldata.Price_Brent_Oil)
sns.distplot(coaldata.Price_Dubai_Brent_Oil)
sns.distplot(coaldata.Price_ExxonMobil)
sns.distplot(coaldata.Price_Shenhua)
sns.distplot(coaldata.Price_All_Share)
sns.distplot(coaldata.Price_Mining)
sns.distplot(coaldata.Price_LNG_Japan_Korea_Marker_PLATTS)
sns.distplot(coaldata.Price_ZAR_USD)
sns.distplot(coaldata.Price_Natural_Gas)
sns.distplot(coaldata.Price_ICE)
sns.distplot(coaldata.Price_Dutch_TTF)
sns.distplot(coaldata.Price_Indian_en_exg_rate)

# boxplot
coaldata.plot(kind='box', subplots=True)
plt.subplots_adjust(wspace=0.75)
plt.show()

# So outlier is present in all the columns expect 3 columns
sns.boxplot(coaldata)
sns.boxplot(coaldata.Coal_RB_4800_FOB_London_Close_USD)
sns.boxplot(coaldata.Coal_RB_5500_FOB_London_Close_USD)
sns.boxplot(coaldata.Coal_RB_5700_FOB_London_Close_USD)
sns.boxplot(coaldata.Coal_RB_6000_FOB_CurrentWeek_Avg_USD)
sns.boxplot(coaldata.Coal_India_5500_CFR_London_Close_USD)
sns.boxplot(coaldata.Price_WTI)
sns.boxplot(coaldata.Price_Brent_Oil)
sns.boxplot(coaldata.Price_Dubai_Brent_Oil)
sns.boxplot(coaldata.Price_ExxonMobil)
sns.boxplot(coaldata.Price_Shenhua)
sns.boxplot(coaldata.Price_All_Share)
sns.boxplot(coaldata.Price_Mining)
sns.boxplot(coaldata.Price_LNG_Japan_Korea_Marker_PLATTS)
sns.boxplot(coaldata.Price_ZAR_USD)
sns.boxplot(coaldata.Price_Natural_Gas)
sns.boxplot(coaldata.Price_ICE)
sns.boxplot(coaldata.Price_Dutch_TTF)
sns.boxplot(coaldata.Price_Indian_en_exg_rate)


# ----------------- Missing Value detection by FFILL and Bfill
# after applying ffill, we have only one missing value in last column so again we are applying bfill
coaldata = coaldata.ffill()
coaldata = coaldata.bfill()
coaldata.isnull().sum()

# --------------- OUTLIER DETECTION ------------------------------
# Check outliers
coaldata.plot(kind='box', subplots=True, sharey=False, figsize=(30, 6))
a = coaldata.describe()

''' 
we found outlier in 
Coal_RB_4800_FOB_London_Close_USD, 
Coal_RB_5500_FOB_London_Close_USD, 
Coal_RB_5700_FOB_London_Close_USD, 
Coal_RB_6000_FOB_CurrentWeek_Avg_USD, 
Coal_India_5500_CFR_London_Close_USD, 
Price_WTI, 
Price_Brent_Oil, 
Price_Dubai_Brent_Oil, 
Price_Shenhua, 
Price_All_Share, 
Price_Mining, 
Price_LNG_Japan_Korea_Marker_PLATTS, 
Price_Natural_Gas, 
Price_Dutch_TTF, 
Price_Indian_en_exg_rate
'''

# Winsorization

# Coal_RB_4800_FOB_London_Close_USD
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Coal_RB_4800_FOB_London_Close_USD'])
coaldata['Coal_RB_4800_FOB_London_Close_USD'] = winsor_iqr.fit_transform(
    coaldata[['Coal_RB_4800_FOB_London_Close_USD']])

# coal_RB_5500_FOB_London_Close_USD
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Coal_RB_5500_FOB_London_Close_USD'])
coaldata['Coal_RB_5500_FOB_London_Close_USD'] = winsor_iqr.fit_transform(
    coaldata[['Coal_RB_5500_FOB_London_Close_USD']])

# Coal_RB_5700_FOB_London_Close_USD
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Coal_RB_5700_FOB_London_Close_USD'])
coaldata['Coal_RB_5700_FOB_London_Close_USD'] = winsor_iqr.fit_transform(
    coaldata[['Coal_RB_5700_FOB_London_Close_USD']])

# Coal_RB_6000_FOB_CurrentWeek_Avg_USD
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Coal_RB_6000_FOB_CurrentWeek_Avg_USD'])
coaldata['Coal_RB_6000_FOB_CurrentWeek_Avg_USD'] = winsor_iqr.fit_transform(
    coaldata[['Coal_RB_6000_FOB_CurrentWeek_Avg_USD']])

# Coal_India_5500_CFR_London_Close_USD
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Coal_India_5500_CFR_London_Close_USD'])
coaldata['Coal_India_5500_CFR_London_Close_USD'] = winsor_iqr.fit_transform(
    coaldata[['Coal_India_5500_CFR_London_Close_USD']])

# Price_WTI
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_WTI'])
coaldata['Price_WTI'] = winsor_iqr.fit_transform(coaldata[['Price_WTI']])

# Price_Brent_Oil
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Brent_Oil'])
coaldata['Price_Brent_Oil'] = winsor_iqr.fit_transform(
    coaldata[['Price_Brent_Oil']])

# Price_Dubai_Brent_Oil
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Dubai_Brent_Oil'])
coaldata['Price_Dubai_Brent_Oil'] = winsor_iqr.fit_transform(
    coaldata[['Price_Dubai_Brent_Oil']])

# Price_Shenhua
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Shenhua'])
coaldata['Price_Shenhua'] = winsor_iqr.fit_transform(
    coaldata[['Price_Shenhua']])

# Price_All_Share
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_All_Share'])
coaldata['Price_All_Share'] = winsor_iqr.fit_transform(
    coaldata[['Price_All_Share']])

# Price_Mining
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Mining'])
coaldata['Price_Mining'] = winsor_iqr.fit_transform(coaldata[['Price_Mining']])

# Price_LNG_Japan_Korea_Marker_PLATTS
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_LNG_Japan_Korea_Marker_PLATTS'])
coaldata['Price_LNG_Japan_Korea_Marker_PLATTS'] = winsor_iqr.fit_transform(
    coaldata[['Price_LNG_Japan_Korea_Marker_PLATTS']])

# Price_Natural_Gas
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Natural_Gas'])
coaldata['Price_Natural_Gas'] = winsor_iqr.fit_transform(
    coaldata[['Price_Natural_Gas']])

# Price_Dutch_TTF
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Dutch_TTF'])
coaldata['Price_Dutch_TTF'] = winsor_iqr.fit_transform(
    coaldata[['Price_Dutch_TTF']])

# Price_Indian_en_exg_rate
winsor_iqr = Winsorizer(capping_method='iqr',
                        # IQR rule boundaries
                        tail='both',  # cap left, right or both tails
                        fold=1.5,
                        variables=['Price_Indian_en_exg_rate'])
coaldata['Price_Indian_en_exg_rate'] = winsor_iqr.fit_transform(
    coaldata[['Price_Indian_en_exg_rate']])


# ----------------------------AUTO EDA------------------------

# 1. Sweetviz

s = sv.analyze(coaldata)
s.show_html()

# 2. Autoviz

av = AutoViz_Class()
a = av.AutoViz(coaldata, dfte=coaldata, chart_format="svg",
               max_rows_analyzed=150000, max_cols_analyzed=30)

# D-TAle

d = dtale.show(coaldata)
d.open_browser()


# ---------------------- interpolation------------------------

coaldata = coaldata.interpolate(method='linear')
coaldata.head()


# --------------------- KPSS TEST ---------------------

results = {}

for column in coaldata.columns:
    series = coaldata[column].dropna()
    kpss_stat, p_value, lags, crit_values = ts.kpss(series, regression='c')
    results[column] = {
        'KPSS Statistic': kpss_stat,
        'p-value': p_value,
        'lags': lags,
        'Critical Value 10%': crit_values['10%'],
        'Critical Value 5%': crit_values['5%'],
        'Critical Value 2.5%': crit_values['2.5%'],
        'Critical Value 1%': crit_values['1%']
    }

# Convert the results dictionary to a DataFrame
results_df = pd.DataFrame(results).T

# Display the results DataFrame
print(results_df)

'''
H0 = the dataset is stationary
H1 = the dataset is non-stationary

p-value = 0.01 < 0.05 => so p low null go => reject null hypothesis
reject the null hypothesis and conclude that the time series is non-stationary.
The test statistic in the KPSS test is positive, and it is compared against critical values. 
KPSS Statistic is higher than the critical value(0.05). so, the null hypothesis is rejected.

Coal_RB_4800_FOB_London_Close_USD: 
KPSS Statistic = 1.531131, 
p-value = 0.010000
This series is likely non-stationary.

Coal_RB_5500_FOB_London_Close_USD: 
KPSS Statistic = 1.446811, 
p-value = 0.010000
This series is likely non-stationary.

Coal_RB_5700_FOB_London_Close_USD: 
KPSS Statistic = 1.347539, 
p-value = 0.010000
This series is likely non-stationary.

Coal_RB_6000_FOB_CurrentWeek_Avg_USD: 
KPSS Statistic = 1.266845, 
p-value = 0.010000
This series is likely non-stationary.

Coal_India_5500_CFR_London_Close_USD: 
KPSS Statistic = 1.521133, 
p-value = 0.010000
This series is likely non-stationary.

Price_WTI: 
KPSS Statistic = 2.658860,
p-value = 0.010000
This series is likely non-stationary.

Price_Brent_Oil: 
KPSS Statistic = 2.802898, 
p-value = 0.010000
This series is likely non-stationary.

Price_Dubai_Brent_Oil: 
KPSS Statistic = 2.970070, 
p-value = 0.010000
This series is likely non-stationary.

Price_ExxonMobil: 
KPSS Statistic = 5.034559,
p-value = 0.010000
This series is likely non-stationary.

Price_Shenhua: 
KPSS Statistic = 4.607248, 
p-value = 0.010000
This series is likely non-stationary.

Price_All_Share: 
KPSS Statistic = 3.947453, 
p-value = 0.010000
This series is likely non-stationary.

Price_Mining: 
KPSS Statistic = 0.549221, 
p-value = 0.030581
This series is close to the threshold but still suggests non-stationarity at a more lenient level.

Price_LNG_Japan_Korea_Marker_PLATTS: 
KPSS Statistic = 1.173768, 
p-value = 0.010000
This series is likely non-stationary.

Price_ZAR_USD: 
KPSS Statistic = 3.010557, 
p-value = 0.010000
This series is likely non-stationary.

Price_Natural_Gas: 
KPSS Statistic = 0.959332, 
p-value = 0.010000
This series is likely non-stationary.

Price_ICE: 
KPSS Statistic = 1.065565, 
p-value = 0.010000
This series is likely non-stationary.

Price_Dutch_TTF: 
KPSS Statistic = 1.188385, 
p-value = 0.010000
This series is likely non-stationary.

Price_Indian_en_exg_rate: 
KPSS Statistic = 1.702612, 
p-value = 0.010000
This series is likely non-stationary.

'''

# --------------------- ADF TEST --------------------------


results1 = {}

for column in coaldata.columns:
    series = coaldata[column].dropna()  # Ensure no NaN values in the series
    adf_stat, p_value, used_lag, n_obs, crit_values, icbest = adfuller(series)
    results1[column] = {
        'ADF Statistic': adf_stat,
        'p-value': p_value,
        'Used Lag': used_lag,
        'Number of Observations': n_obs,
        'Critical Values': crit_values,
        'IC Best': icbest
    }

# Convert the results dictionary to a DataFrame
results_adf = pd.DataFrame(results1).T

# Display the results DataFrame
print(results_adf)

'''
H0 = the dataset is non-stationary
H1 = the dataset is stationary

Coal_RB_4800_FOB_London_Close_USD:
ADF Statistic: -1.548438
p-value: 0.509428
Interpretation: The p-value is high, suggesting that we fail to reject the null hypothesis. This series might be non-stationary.

Coal_RB_5500_FOB_London_Close_USD:
ADF Statistic: -1.669817
p-value: 0.446727
Interpretation: The p-value is high, then the series is likely non-stationarity.

Coal_RB_5700_FOB_London_Close_USD:
ADF Statistic: -1.470938
p-value: 0.547836
Interpretation: The p-value is high, then the series is likely non-stationarity.

Coal_RB_6000_FOB_CurrentWeek_Avg_USD:
ADF Statistic: -1.382277
p-value: 0.590743
Interpretation: The p-value is high, then the series is likely non-stationarity.

Coal_India_5500_CFR_London_Close_USD:
ADF Statistic: -1.597799
p-value: 0.484699
Interpretation: The p-value is high, then the series is likely non-stationarity.

Price_WTI:
ADF Statistic: -1.979154
p-value: 0.295787
Interpretation: The p-value is high, then the series is likely non-stationarity.

Price_Brent_Oil:
ADF Statistic: -1.972388    
p-value: 0.29881   
Interpretation: The p-value is high, then the series is likely non-stationarity.

Price_Dubai_Brent_Oil:                  
ADF Statistic: -2.115846    
p-value: 0.238191  
Interpretation: The p-value is high, then the series is likely non-stationarity.

Price_ExxonMobil:                     
ADF Statistic:  -0.995986   
p-value: 0.754726 
Interpretation: The p-value is high, then the series is likely non-stationarity.
 
Price_Shenhua:                        
ADF Statistic:  -0.107393   
p-value: 0.948697 
Interpretation: The p-value is high, then the series is likely non-stationarity.
     
Price_All_Share:                      
ADF Statistic:  -1.857262   
p-value: 0.352426 
Interpretation: The p-value is high, then the series is likely non-stationarity.
    
Price_Mining:                         
ADF Statistic:  -3.426669   
p-value: 0.010083
Interpretation:  The p-value is less than 0.05, reject the null hypothesis and the series is likely to be stationary.
      
Price_LNG_Japan_Korea_Marker_PLATTS:  
ADF Statistic:  -1.994199   
p-value: 0.289122 
Interpretation: The p-value is high, then the series is likely non-stationarity.
    
Price_ZAR_USD:                        
ADF Statistic:  -1.568167   
p-value: 0.499562
Interpretation: The p-value is high, then the series is likely non-stationarity.
     
Price_Natural_Gas:                    
ADF Statistic:  -1.742692   
p-value: 0.409261 
Interpretation: The p-value is high, then the series is likely non-stationarity.
    
Price_ICE:                            
ADF Statistic:  -2.093937   
p-value:  0.24696 
Interpretation: The p-value is high, then the series is likely non-stationarity.
     
Price_Dutch_TTF:                      
ADF Statistic:  -1.55631   
p-value: 0.505495 
Interpretation: The p-value is high, then the series is likely non-stationarity.
    
Price_Indian_en_exg_rate:             
ADF Statistic:  -1.69379   
p-value: 0.434315 
Interpretation: The p-value is high, then the series is likely non-stationarity.    

'''
# --------------------------- Model Building ----------------------
'''
Identify 10 Non-Stationary Data Models:

Autoregressive Integrated Moving Average (ARIMA)
Seasonal ARIMA (SARIMA)
Exponential Smoothing State Space Model (ETS)
Holt-Winters Exponential Smoothing
Prophet
Long Short-Term Memory (LSTM) Neural Network
Gated Recurrent Units (GRU) Neural Network
Vector Autoregression (VAR)
Dynamic Time Warping (DTW)
Bayesian Structural Time Series (BSTS)

'''



# Data Partition
Train = coaldata.head(977)
Test = coaldata.tail(130)

Test.to_csv('test_data.csv')
import os
os.getcwd()

# --------------- Auto Correlation Function PLOT-------------------------
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(coaldata.Coal_RB_4800_FOB_London_Close_USD)
tsa_plots.plot_acf(coaldata.Coal_RB_5500_FOB_London_Close_USD)
tsa_plots.plot_acf(coaldata.Coal_RB_5700_FOB_London_Close_USD)
tsa_plots.plot_acf(coaldata.Coal_RB_6000_FOB_CurrentWeek_Avg_USD)
tsa_plots.plot_acf(coaldata.Coal_India_5500_CFR_London_Close_USD)
tsa_plots.plot_acf(coaldata.Price_WTI)
tsa_plots.plot_acf(coaldata.Price_Brent_Oil)
tsa_plots.plot_acf(coaldata.Price_Dubai_Brent_Oil)
tsa_plots.plot_acf(coaldata.Price_ExxonMobil)
tsa_plots.plot_acf(coaldata.Price_Shenhua)
tsa_plots.plot_acf(coaldata.Price_All_Share)
tsa_plots.plot_acf(coaldata.Price_Mining)
tsa_plots.plot_acf(coaldata.Price_LNG_Japan_Korea_Marker_PLATTS)
tsa_plots.plot_acf(coaldata.Price_ZAR_USD)
tsa_plots.plot_acf(coaldata.Price_Natural_Gas)
tsa_plots.plot_acf(coaldata.Price_ICE)
tsa_plots.plot_acf(coaldata.Price_Dutch_TTF)
tsa_plots.plot_acf(coaldata.Price_Indian_en_exg_rate)

# --------------- Partial Auto Correlation Function PLOT-------------------------

tsa_plots.plot_pacf(coaldata.Coal_RB_4800_FOB_London_Close_USD)
tsa_plots.plot_pacf(coaldata.Coal_RB_5500_FOB_London_Close_USD)
tsa_plots.plot_pacf(coaldata.Coal_RB_5700_FOB_London_Close_USD)
tsa_plots.plot_pacf(coaldata.Coal_RB_6000_FOB_CurrentWeek_Avg_USD)
tsa_plots.plot_pacf(coaldata.Coal_India_5500_CFR_London_Close_USD)
tsa_plots.plot_pacf(coaldata.Price_WTI)
tsa_plots.plot_pacf(coaldata.Price_Brent_Oil)
tsa_plots.plot_pacf(coaldata.Price_Dubai_Brent_Oil)
tsa_plots.plot_pacf(coaldata.Price_ExxonMobil)
tsa_plots.plot_pacf(coaldata.Price_Shenhua)
tsa_plots.plot_pacf(coaldata.Price_All_Share)
tsa_plots.plot_pacf(coaldata.Price_Mining)
tsa_plots.plot_pacf(coaldata.Price_LNG_Japan_Korea_Marker_PLATTS)
tsa_plots.plot_pacf(coaldata.Price_ZAR_USD)
tsa_plots.plot_pacf(coaldata.Price_Natural_Gas)
tsa_plots.plot_pacf(coaldata.Price_ICE)
tsa_plots.plot_pacf(coaldata.Price_Dutch_TTF)
tsa_plots.plot_pacf(coaldata.Price_Indian_en_exg_rate)

# 1. -------------------ARIMA ----------------

from statsmodels.tsa.arima.model import ARIMA
test_arima = pd.read_csv(r"C:\Users\Hriteek Kumar Nayak\test_data.csv",  index_col = 0)

'1. ------Coal_RB_4800_FOB_London_Close_USD:-------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model1 = ARIMA(Train.Coal_RB_4800_FOB_London_Close_USD, order = (30, 1, 15))
res1 = model1.fit()
print(res1.summary())


# Evaluate forecasts
from sklearn.metrics import mean_absolute_percentage_error

# Forecast on training data
train_pred1 = res1.fittedvalues

# Forecast on testing data
test_forecast1 = res1.forecast(steps=len(Test.Coal_RB_4800_FOB_London_Close_USD))


# Calculate MAPE for training data
mape_train1 = mean_absolute_percentage_error(Train.Coal_RB_4800_FOB_London_Close_USD, train_pred1)

# Calculate MAPE for testing data
mape_test1 = mean_absolute_percentage_error(Test.Coal_RB_4800_FOB_London_Close_USD, test_forecast1)

print(f"MAPE for training data: {mape_train1}")
print(f"MAPE for testing data: {mape_test1}")

# plot forecasts against actual outcomes
from matplotlib import pyplot 
pyplot.plot(Test.Coal_RB_4800_FOB_London_Close_USD, color = 'black')
pyplot.plot(test_forecast1, color = 'red')
pyplot.show()


'2. ------Coal_RB_5500_FOB_London_Close_USD:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model2 = ARIMA(Train.Coal_RB_5500_FOB_London_Close_USD, order = (30, 1, 15))
res2 = model2.fit()
print(res2.summary())

# Forecast on training data
train_pred2 = res2.fittedvalues

# Forecast on testing data
test_forecast2 = res2.forecast(steps=len(Test.Coal_RB_5500_FOB_London_Close_USD))

# Calculate MAPE for training data
mape_train2 = mean_absolute_percentage_error(Train.Coal_RB_5500_FOB_London_Close_USD, train_pred2)

# Calculate MAPE for testing data
mape_test2 = mean_absolute_percentage_error(Test.Coal_RB_5500_FOB_London_Close_USD, test_forecast2)

print(f"MAPE for training data: {mape_train2}")
print(f"MAPE for testing data: {mape_test2}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Coal_RB_5500_FOB_London_Close_USD, color = 'black')
pyplot.plot(test_forecast2, color = 'red')
pyplot.show()


'3. ------Coal_RB_5700_FOB_London_Close_USD:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model3 = ARIMA(Train.Coal_RB_5700_FOB_London_Close_USD, order = (30, 1, 15))
res3 = model3.fit()
print(res3.summary())

# Forecast on training data
train_pred3 = res3.fittedvalues

# Forecast on testing data
test_forecast3 = res3.forecast(steps=len(Test.Coal_RB_5700_FOB_London_Close_USD))

# Calculate MAPE for training data
mape_train3 = mean_absolute_percentage_error(Train.Coal_RB_5700_FOB_London_Close_USD, train_pred3)

# Calculate MAPE for testing data
mape_test3 = mean_absolute_percentage_error(Test.Coal_RB_5700_FOB_London_Close_USD, test_forecast3)

print(f"MAPE for training data: {mape_train3}")
print(f"MAPE for testing data: {mape_test3}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Coal_RB_5700_FOB_London_Close_USD, color = 'black')
pyplot.plot(test_forecast3, color = 'red')
pyplot.show()


'4. ------Coal_RB_6000_FOB_CurrentWeek_Avg_USD:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model4 = ARIMA(Train.Coal_RB_6000_FOB_CurrentWeek_Avg_USD, order = (30, 1, 15))
res4 = model4.fit()
print(res4.summary())

# Forecast on training data
train_pred4 = res4.fittedvalues

# Forecast on testing data
test_forecast4 = res4.forecast(steps=len(Test.Coal_RB_6000_FOB_CurrentWeek_Avg_USD))

# Calculate MAPE for training data
mape_train4 = mean_absolute_percentage_error(Train.Coal_RB_6000_FOB_CurrentWeek_Avg_USD, train_pred4)

# Calculate MAPE for testing data
mape_test4 = mean_absolute_percentage_error(Test.Coal_RB_6000_FOB_CurrentWeek_Avg_USD, test_forecast4)

print(f"MAPE for training data: {mape_train4}")
print(f"MAPE for testing data: {mape_test4}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Coal_RB_6000_FOB_CurrentWeek_Avg_USD, color = 'black')
pyplot.plot(test_forecast4, color = 'red')
pyplot.show()


'5. ------Coal_India_5500_CFR_London_Close_USD:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model5 = ARIMA(Train.Coal_India_5500_CFR_London_Close_USD, order = (30, 1, 15))
res5 = model5.fit()
print(res5.summary())

# Forecast on training data
train_pred5 = res5.fittedvalues

# Forecast on testing data
test_forecast5 = res5.forecast(steps=len(Test.Coal_India_5500_CFR_London_Close_USD))

# Calculate MAPE for training data
mape_train5 = mean_absolute_percentage_error(Train.Coal_India_5500_CFR_London_Close_USD, train_pred5)

# Calculate MAPE for testing data
mape_test5 = mean_absolute_percentage_error(Test.Coal_India_5500_CFR_London_Close_USD, test_forecast5)

print(f"MAPE for training data: {mape_train5}")
print(f"MAPE for testing data: {mape_test5}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Coal_India_5500_CFR_London_Close_USD, color = 'black')
pyplot.plot(test_forecast5, color = 'red')
pyplot.show()


'6. ------Price_WTI:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model6 = ARIMA(Train.Price_WTI, order = (30, 1, 15))
res6 = model6.fit()
print(res6.summary())

# Forecast on training data
train_pred6 = res6.fittedvalues

# Forecast on testing data
test_forecast6 = res6.forecast(steps=len(Test.Price_WTI))

# Calculate MAPE for training data
mape_train6 = mean_absolute_percentage_error(Train.Price_WTI, train_pred6)

# Calculate MAPE for testing data
mape_test6 = mean_absolute_percentage_error(Test.Price_WTI, test_forecast6)

print(f"MAPE for training data: {mape_train6}")
print(f"MAPE for testing data: {mape_test6}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_WTI, color = 'black')
pyplot.plot(test_forecast5, color = 'red')
pyplot.show()


'7. ------Price_Brent_Oil:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model7 = ARIMA(Train.Price_Brent_Oil, order = (30, 1, 15))
res7 = model7.fit()
print(res7.summary())

# Forecast on training data
train_pred7 = res7.fittedvalues

# Forecast on testing data
test_forecast7 = res7.forecast(steps=len(Test.Price_Brent_Oil))

# Calculate MAPE for training data
mape_train7 = mean_absolute_percentage_error(Train.Price_Brent_Oil, train_pred7)

# Calculate MAPE for testing data
mape_test7 = mean_absolute_percentage_error(Test.Price_Brent_Oil, test_forecast7)

print(f"MAPE for training data: {mape_train7}")
print(f"MAPE for testing data: {mape_test7}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Brent_Oil, color = 'black')
pyplot.plot(test_forecast5, color = 'red')
pyplot.show()


'8. ------Price_Dubai_Brent_Oil:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model8 = ARIMA(Train.Price_Dubai_Brent_Oil, order = (30, 1, 15))
res8 = model8.fit()
print(res8.summary())

# Forecast on training data
train_pred8 = res8.fittedvalues

# Forecast on testing data
test_forecast8 = res8.forecast(steps=len(Test.Price_Dubai_Brent_Oil))

# Calculate MAPE for training data
mape_train8 = mean_absolute_percentage_error(Train.Price_Dubai_Brent_Oil, train_pred8)

# Calculate MAPE for testing data
mape_test8 = mean_absolute_percentage_error(Test.Price_Dubai_Brent_Oil, test_forecast8)

print(f"MAPE for training data: {mape_train8}")
print(f"MAPE for testing data: {mape_test8}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Dubai_Brent_Oil, color = 'black')
pyplot.plot(test_forecast5, color = 'red')
pyplot.show()


'9. ------Price_ExxonMobil:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model9 = ARIMA(Train.Price_ExxonMobil, order = (30, 1, 15))
res9 = model9.fit()
print(res9.summary())

# Forecast on training data
train_pred9 = res9.fittedvalues

# Forecast on testing data
test_forecast9 = res9.forecast(steps=len(Test.Price_ExxonMobil))

# Calculate MAPE for training data
mape_train9 = mean_absolute_percentage_error(Train.Price_ExxonMobil, train_pred9)

# Calculate MAPE for testing data
mape_test9 = mean_absolute_percentage_error(Test.Price_ExxonMobil, test_forecast9)

print(f"MAPE for training data: {mape_train9}")
print(f"MAPE for testing data: {mape_test9}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_ExxonMobil, color = 'black')
pyplot.plot(test_forecast9, color = 'red')
pyplot.show()


'10. ------Price_Shenhua:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model10 = ARIMA(Train.Price_Shenhua, order = (30, 1, 15))
res10 = model10.fit()
print(res10.summary())

# Forecast on training data
train_pred10 = res10.fittedvalues

# Forecast on testing data
test_forecast10 = res10.forecast(steps=len(Test.Price_Shenhua))

# Calculate MAPE for training data
mape_train10 = mean_absolute_percentage_error(Train.Price_Shenhua, train_pred10)

# Calculate MAPE for testing data
mape_test10 = mean_absolute_percentage_error(Test.Price_Shenhua, test_forecast10)

print(f"MAPE for training data: {mape_train10}")
print(f"MAPE for testing data: {mape_test10}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Shenhua, color = 'black')
pyplot.plot(test_forecast10, color = 'red')
pyplot.show()


'11. ------Price_All_Share:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model11 = ARIMA(Train.Price_All_Share, order = (30, 1, 15))
res11 = model11.fit()
print(res11.summary())

# Forecast on training data
train_pred11 = res11.fittedvalues

# Forecast on testing data
test_forecast11 = res11.forecast(steps=len(Test.Price_All_Share))

# Calculate MAPE for training data
mape_train11 = mean_absolute_percentage_error(Train.Price_All_Share, train_pred11)

# Calculate MAPE for testing data
mape_test11 = mean_absolute_percentage_error(Test.Price_All_Share, test_forecast11)

print(f"MAPE for training data: {mape_train11}")
print(f"MAPE for testing data: {mape_test11}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_All_Share, color = 'black')
pyplot.plot(test_forecast10, color = 'red')
pyplot.show()


'12. ------Price_Mining:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model12 = ARIMA(Train.Price_Mining, order = (30, 1, 15))
res12 = model12.fit()
print(res12.summary())

# Forecast on training data
train_pred12 = res12.fittedvalues

# Forecast on testing data
test_forecast12 = res12.forecast(steps=len(Test.Price_Mining))

# Calculate MAPE for training data
mape_train12= mean_absolute_percentage_error(Train.Price_Mining, train_pred12)

# Calculate MAPE for testing data
mape_test12= mean_absolute_percentage_error(Test.Price_Mining, test_forecast12)

print(f"MAPE for training data: {mape_train12}")
print(f"MAPE for testing data: {mape_test12}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Mining, color = 'black')
pyplot.plot(test_forecast10, color = 'red')
pyplot.show()


'13. ------Price_Mining:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model13 = ARIMA(Train.Price_LNG_Japan_Korea_Marker_PLATTS, order = (30, 1, 15))
res13 = model13.fit()
print(res13.summary())

# Forecast on training data
train_pred13 = res13.fittedvalues

# Forecast on testing data
test_forecast13 = res13.forecast(steps=len(Test.Price_LNG_Japan_Korea_Marker_PLATTS))

# Calculate MAPE for training data
mape_train13= mean_absolute_percentage_error(Train.Price_LNG_Japan_Korea_Marker_PLATTS, train_pred13)

# Calculate MAPE for testing data
mape_test13 = mean_absolute_percentage_error(Test.Price_LNG_Japan_Korea_Marker_PLATTS, test_forecast13)

print(f"MAPE for training data: {mape_train13}")
print(f"MAPE for testing data: {mape_test13}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_LNG_Japan_Korea_Marker_PLATTS, color = 'black')
pyplot.plot(test_forecast10, color = 'red')
pyplot.show()


'14. ------Price_ZAR_USD:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model14 = ARIMA(Train.Price_ZAR_USD, order = (30, 1, 15))
res14 = model14.fit()
print(res14.summary())

# Forecast on training data
train_pred14 = res14.fittedvalues

# Forecast on testing data
test_forecast14 = res14.forecast(steps=len(Test.Price_ZAR_USD))

# Calculate MAPE for training data
mape_train14= mean_absolute_percentage_error(Train.Price_ZAR_USD, train_pred14)

# Calculate MAPE for testing data
mape_test14 = mean_absolute_percentage_error(Test.Price_ZAR_USD, test_forecast14)

print(f"MAPE for training data: {mape_train14}")
print(f"MAPE for testing data: {mape_test14}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_ZAR_USD, color = 'black')
pyplot.plot(test_forecast10, color = 'red')
pyplot.show()


'15. ------Price_Natural_Gas:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model15 = ARIMA(Train.Price_Natural_Gas, order = (30, 1, 15))
res15 = model15.fit()
print(res15.summary())

# Forecast on training data
train_pred15= res15.fittedvalues

# Forecast on testing data
test_forecast15 = res15.forecast(steps=len(Test.Price_Natural_Gas))

# Calculate MAPE for training data
mape_train15 = mean_absolute_percentage_error(Train.Price_Natural_Gas, train_pred15)

# Calculate MAPE for testing data
mape_test15 = mean_absolute_percentage_error(Test.Price_Natural_Gas, test_forecast15)

print(f"MAPE for training data: {mape_train15}")
print(f"MAPE for testing data: {mape_test15}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Natural_Gas, color = 'black')
pyplot.plot(test_forecast10, color = 'red')
pyplot.show()


'16. ------Price_ICE:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model16 = ARIMA(Train.Price_ICE, order = (30, 1, 15))
res16 = model16.fit()
print(res16.summary())

# Forecast on training data
train_pred16 = res16.fittedvalues

# Forecast on testing data
test_forecast16 = res16.forecast(steps=len(Test.Price_ICE))

# Calculate MAPE for training data
mape_train16 = mean_absolute_percentage_error(Train.Price_ICE, train_pred16)

# Calculate MAPE for testing data
mape_test16 = mean_absolute_percentage_error(Test.Price_ICE, test_forecast16)

print(f"MAPE for training data: {mape_train16}")
print(f"MAPE for testing data: {mape_test16}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_ICE, color = 'black')
pyplot.plot(test_forecast16, color = 'red')
pyplot.show()


'17. ------Price_Dutch_TTF:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model17 = ARIMA(Train.Price_Dutch_TTF, order = (30, 1, 15))
res17 = model17.fit()
print(res17.summary())

# Forecast on training data
train_pred17 = res17.fittedvalues

# Forecast on testing data
test_forecast17 = res17.forecast(steps=len(Test.Price_Dutch_TTF))

# Calculate MAPE for training data
mape_train17 = mean_absolute_percentage_error(Train.Price_Dutch_TTF, train_pred17)

# Calculate MAPE for testing data
mape_test17 = mean_absolute_percentage_error(Test.Price_Dutch_TTF, test_forecast17)

print(f"MAPE for training data: {mape_train17}")
print(f"MAPE for testing data: {mape_test17}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Dutch_TTF, color = 'black')
pyplot.plot(test_forecast17, color = 'red')
pyplot.show()


'18. ------Price_Indian_en_exg_rate:------'

#----Training Data-----
# ARIMA with AR = 30, MA = 6
model18 = ARIMA(Train.Price_Indian_en_exg_rate, order = (30, 1, 15))
res18 = model18.fit()
print(res18.summary())

# Forecast on training data
train_pred18 = res18.fittedvalues

# Forecast on testing data
test_forecast18 = res18.forecast(steps=len(Test.Price_Indian_en_exg_rate))

# Calculate MAPE for training data
mape_train18 = mean_absolute_percentage_error(Train.Price_Indian_en_exg_rate, train_pred18)

# Calculate MAPE for testing data
mape_test18 = mean_absolute_percentage_error(Test.Price_Indian_en_exg_rate, test_forecast18)

print(f"MAPE for training data: {mape_train18}")
print(f"MAPE for testing data: {mape_test18}")

# plot forecasts against actual outcomes
pyplot.plot(Test.Price_Indian_en_exg_rate, color = 'black')
pyplot.plot(test_forecast18, color = 'red')
pyplot.show()



# 2. -------------------SARIMA ----------------

from statsmodels.tsa.statespace.sarimax import SARIMAX

target_columns = ['Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua',
                  'Price_All_Share', 'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD',
                  'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate']

# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    # Ensure the data is univariate and drop NaNs
    uni_train = Train[target_column].dropna()
    uni_test = Test[target_column].dropna()

    # Check for empty datasets
    if uni_train.empty or uni_test.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    try:
        # Fit the SARMA model on the training data
        p, d, q = 1, 0, 1  # AR, differencing, MA components for non-seasonal part
        P, D, Q, s = 1, 0, 1, 12  # Seasonal AR, differencing, MA, seasonal period (monthly data assumed)

        # The SARMA model assumes d and D are zero, focusing on AR and MA components
        model = SARIMAX(uni_train, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False)
        model_fit = model.fit(disp=False)

        # Forecast the values for the training period
        train_forecast = model_fit.predict(start=uni_train.index[0], end=uni_train.index[-1])

        # Forecast the values for the test period
        forecast = model_fit.get_forecast(steps=len(uni_test))
        forecast_values = forecast.predicted_mean

        # Calculate the Mean Absolute Percentage Error (MAPE)
        train_mape = mean_absolute_percentage_error(uni_train, train_forecast)
        test_mape = mean_absolute_percentage_error(uni_test, forecast_values)
        
        print(f'MAPE for {target_column} (Train): {train_mape:.2f}')
        print(f'MAPE for {target_column} (Test): {test_mape:.2f}')

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(uni_train.index, uni_train, label='Train')
        plt.plot(uni_test.index, uni_test, label='Test', color='red')
        plt.plot(uni_test.index, forecast_values, label='Forecast', color='blue')
        plt.legend(loc='best')
        plt.title(f'Forecast vs Actuals for {target_column}')
        plt.show()
    except np.linalg.LinAlgError:
        print(f"Numerical issue encountered for {target_column}. Try different parameters or data preprocessing.")
    except ValueError as e:
        print(f"ValueError for {target_column}: {e}")
    except Exception as e:
        print(f"Unexpected error for {target_column}: {e}")


# 3. -----------------SES (Simple Exponential Smoothing)-------------------

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Ensure coaldata has a DatetimeIndex
coaldata.index = pd.to_datetime(coaldata.index, errors='coerce')

# Split the data using datetime slicing
train_end_date = coaldata.index[977]
train_data = coaldata.loc[:train_end_date]
test_data = coaldata.loc[train_end_date:]

for target_column in target_columns:
    print(f"Forecasting for {target_column}")
    
    # Check if the column exists
    if target_column not in coaldata.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue
    
    # Extract the target column and split into train and test sets
    endog = coaldata[target_column].dropna()
    
    # Check for valid datetime index
    if not isinstance(endog.index, pd.DatetimeIndex):
        print(f"Index type for {target_column} is not DatetimeIndex.")
        continue
    
    train_series = endog.loc[train_data.index]
    test_series = endog.loc[test_data.index]
    
    # Check for empty datasets
    if train_series.empty or test_series.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue
    
    # Fit the Exponential Smoothing model
    try:
        model = ExponentialSmoothing(train_series, seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        
        # Forecast for the test period
        train_forecast_values = model_fit.fittedvalues
        forecast_values = model_fit.forecast(steps=len(test_series))
        
        # Align forecast index with test_data index
        forecast_values.index = test_series.index
        
        # Calculate MAPE for both train and test data
        train_mape = mean_absolute_percentage_error(train_series, train_forecast_values)
        test_mape = mean_absolute_percentage_error(test_series, forecast_values)
        
        print(f'MAPE for {target_column} (Train): {train_mape:.2f}')
        print(f'MAPE for {target_column} (Test): {test_mape:.2f}')
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(train_series.index, train_series, label='Train', color = 'blue')
        plt.plot(test_series.index, test_series, label='Test', color='red')
        plt.plot(forecast_values.index, forecast_values, label='Forecast', color='black')
        plt.legend(loc='best')
        plt.title(f'Forecast vs Actuals for {target_column}')
        plt.show()
    
    except ValueError as e:
        print(f"ValueError for {target_column}: {e}")
    except Exception as e:
        print(f"Unexpected error for {target_column}: {e}")



# 4. --------------------------PMD ARIMA ------------------------------


from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error

# Split the data using datetime slicing
train_end_index = 977
train_data = coaldata.iloc[:train_end_index]
test_data = coaldata.iloc[train_end_index:]

target_columns = [
    'Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua',
    'Price_All_Share', 'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD',
    'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate'
]

# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    if target_column not in coaldata.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue

    # Extract data
    endog = coaldata[target_column].dropna()
    
    # Split data into training and testing datasets
    train_series = endog[:train_end_index]
    test_series = endog[train_end_index:]

    # Check for empty datasets
    if train_series.empty or test_series.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    # Fit ARIMA model using auto_arima
    try:
        model = auto_arima(train_series, seasonal=True, m=12, stepwise=True, 
                           suppress_warnings=True, error_action='ignore')
        
        # Print the summary of the model
        print(model.summary())

        # Forecast for the test period
        y_pred, conf_int = model.predict(n_periods=len(test_series), return_conf_int=True)
        
        # Calculate MAPE for train and test data
        train_pred = model.predict_in_sample()
        train_mape = mean_absolute_percentage_error(train_series, train_pred)
        test_mape = mean_absolute_percentage_error(test_series, y_pred)
        print(f'MAPE for {target_column} - Train: {train_mape:.2f}, Test: {test_mape:.2f}')

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(train_series.index, train_series, label='Train')
        plt.plot(test_series.index, test_series, label='Test', color='red')
        plt.plot(test_series.index, y_pred, label='Forecast', color='green')

        # Plot confidence intervals
        plt.fill_between(test_series.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3, label='95% Confidence Interval')

        plt.legend()
        plt.title(f'ARIMA Model Forecast vs Actuals for {target_column}')
        plt.show()
    except ValueError as e:
        print(f"ValueError for {target_column}: {e}")
    except Exception as e:
        print(f"Unexpected error for {target_column}: {e}")

# 5. -----------------------Long Short-Term Memory (LSTM)---------------------

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to create sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Parameters
n_steps = 12  # Number of time steps for sequences
epochs = 50
batch_size = 32

# Ensure the index of coaldata is of type DatetimeIndex
coaldata.index = pd.to_datetime(coaldata.index, errors='coerce')

# Define train and test start and end indices
train_end_index = 977
test_start_index = train_end_index
test_end_index = len(coaldata)

# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    if target_column not in coaldata.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue

    # Extract data
    endog = coaldata[target_column].dropna()

    # Split data into training and testing datasets
    train_data = endog.iloc[:train_end_index]
    test_data = endog.iloc[test_start_index:test_end_index]

    if train_data.empty or test_data.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_test, y_test = create_sequences(test_scaled, n_steps)

    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Fit model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Make predictions on training data
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_pred = scaler.inverse_transform(y_train_pred)

    # Make predictions on testing data
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    # Calculate MAPE for training data
    y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1))
    train_mape = mean_absolute_percentage_error(y_train_inverse, y_train_pred)
    print(f'MAPE for {target_column} (Train): {train_mape:.2f}')

    # Calculate MAPE for testing data
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_mape = mean_absolute_percentage_error(y_test_inverse, y_test_pred)
    print(f'MAPE for {target_column} (Test): {test_mape:.2f}')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_data.index, train_data, label='Train')
    plt.plot(test_data.index[n_steps:], y_test_inverse, label='Test', color='red')
    plt.plot(test_data.index[n_steps:], y_test_pred, label='Forecast', color='green')
    plt.legend()
    plt.title(f'Forecast vs Actuals for {target_column}')
    plt.show()

# 6. ---------------------Recurrent Neural Network (RNN)--------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Function to create sequences for RNN
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Parameters
n_steps = 12  # Number of time steps
epochs = 50
batch_size = 32

# Ensure the index of coaldata is of type DatetimeIndex
coaldata.index = pd.to_datetime(coaldata.index, errors='coerce')

# Define train and test start and end indices
train_end_index = 977
test_start_index = train_end_index
test_end_index = len(coaldata)

# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    if target_column not in coaldata.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue

    # Extract and scale data
    endog = coaldata[target_column].dropna().values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(endog.reshape(-1, 1))

    # Split data into train and test sets
    train_data = scaled_data[:train_end_index]
    test_data = scaled_data[test_start_index:test_end_index]

    if len(train_data) == 0 or len(test_data) == 0:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    # Create sequences
    X_train, y_train = create_sequences(train_data, n_steps)
    X_test, y_test = create_sequences(test_data, n_steps)

    # Reshape data for RNN input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the RNN model
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Inverse transform to get actual values
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MAPE
    train_mape = mean_absolute_percentage_error(y_train_actual, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test_actual, y_test_pred)
    print(f'MAPE for {target_column} - Train: {train_mape:.2f}, Test: {test_mape:.2f}')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(endog[:train_end_index], label='Train')
    plt.plot(np.arange(train_end_index + n_steps, train_end_index + n_steps + len(y_test_actual)), y_test_actual, label='Test', color='red')
    plt.plot(np.arange(train_end_index + n_steps, train_end_index + n_steps + len(y_test_actual)), y_test_pred, label='Forecast', color='green')
    plt.legend()
    plt.title(f'Forecast vs Actuals for {target_column}')
    plt.show()
    
# 7. ------------------------Holt-Winters seasonal model-----------------------

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Define function to calculate MAPE
    def calculate_mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)


    # Loop through each target column
    for target_column in target_columns:
        print(f"Forecasting for {target_column}")

        if target_column not in coaldata.columns:
            print(f"Column {target_column} not found in the dataset.")
            continue

        # Extract data
        endog = coaldata[target_column].dropna()
        if not isinstance(endog.index, pd.DatetimeIndex):
            print(f"Index type for {target_column} is not DatetimeIndex.")
            continue

        train_data = endog[:train_end_index]
        test_data = endog[test_start_index:test_end_index]

        if train_data.empty or test_data.empty:
            print(f"No data available for {target_column} in the specified date range.")
            continue

        # Fit Holt-Winters Seasonal Model
        model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()

        # Forecast for the test period
        y_pred = model_fit.forecast(steps=len(test_data))

        # Calculate MAPE for train and test data
        train_mape = calculate_mape(train_data, model_fit.fittedvalues)
        test_mape = calculate_mape(test_data, y_pred)
        print(f'MAPE for {target_column} - Train: {train_mape:.2f}, Test: {test_mape:.2f}')

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(train_data.index, train_data, label='Train')
        plt.plot(test_data.index, test_data, label='Test', color='red')
        plt.plot(test_data.index, y_pred, label='Forecast', color='green')
        plt.legend()
        plt.title(f'Holt-Winters Seasonal Model Forecast vs Actuals for {target_column}')
        plt.show()


# 8. ------------------------Holt’s Linear Trend Model-------------------------

from statsmodels.tsa.holtwinters import Holt

# Define function to calculate MAPE
def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    if target_column not in coaldata.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue

    # Extract data
    endog = coaldata[target_column].dropna()
    if not isinstance(endog.index, pd.DatetimeIndex):
        print(f"Index type for {target_column} is not DatetimeIndex.")
        continue

    # Split data into training and test sets
    train_data = endog[:train_end_index]
    test_data = endog[test_start_index:test_end_index]

    if train_data.empty or test_data.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    # Fit Holt’s Linear Trend Model
    model = Holt(train_data).fit()

    # Forecast for the test period
    y_pred = model.forecast(steps=len(test_data))

    # Calculate MAPE for train and test data
    train_mape = calculate_mape(train_data, model.fittedvalues)
    test_mape = calculate_mape(test_data, y_pred)
    print(f'MAPE for {target_column} - Train: {train_mape:.2f}, Test: {test_mape:.2f}')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_data.index, train_data, label='Train')
    plt.plot(test_data.index, test_data, label='Test', color='red')
    plt.plot(test_data.index, y_pred, label='Forecast', color='green')
    plt.legend()
    plt.title(f'Holt’s Linear Trend Model Forecast vs Actuals for {target_column}')
    plt.show()
  
# 9. -----------GPU MOdel---------

    # Loop through each target column

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense
    from sklearn.preprocessing import MinMaxScaler


    # Ensure the index of coaldata is of type DatetimeIndex
    coaldata.index = pd.to_datetime(coaldata.index, errors='coerce')

    # Example split index; adjust as needed
    train_end_index = 977  # This should be an actual date index for splitting
    train_data = coaldata.iloc[:train_end_index]
    test_data = coaldata.iloc[train_end_index:]

    target_columns = [
        'Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua',
        'Price_All_Share', 'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD',
        'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate'
    ]

    # Define function to calculate MAPE
    def calculate_mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)

    # Function to create sequences for GRU
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Parameters
    seq_length = 10  # Sequence length for GRU

    # Loop through each target column
    for target_column in target_columns:
        print(f"Forecasting for {target_column}")

        if target_column not in coaldata.columns:
            print(f"Column {target_column} not found in the dataset.")
            continue

        # Prepare data
        endog = coaldata[target_column].dropna()
        if not isinstance(endog.index, pd.DatetimeIndex):
            print(f"Index type for {target_column} is not DatetimeIndex.")
            continue

        # Normalize the data
        scaler = MinMaxScaler()
        endog_scaled = scaler.fit_transform(endog.values.reshape(-1, 1))

        # Create training and test datasets
        train_data_scaled = endog_scaled[:train_end_index]
        test_data_scaled = endog_scaled[train_end_index:]

        if train_data_scaled.size == 0 or test_data_scaled.size == 0:
            print(f"No data available for {target_column} in the specified date range.")
            continue

        # Create sequences
        X_train, y_train = create_sequences(train_data_scaled, seq_length)
        X_test, y_test = create_sequences(test_data_scaled, seq_length)

        # Reshape for GRU [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build the GRU model
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(GRU(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Inverse transform to get actual values
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_train_pred = scaler.inverse_transform(y_train_pred)

        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_test_pred = scaler.inverse_transform(y_test_pred)

        # Calculate MAPE
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        print(f'MAPE for {target_column} (Train): {train_mape:.2f}')
        print(f'MAPE for {target_column} (Test): {test_mape:.2f}')

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(endog.index[:len(train_data_scaled)], scaler.inverse_transform(train_data_scaled.reshape(-1, 1)), label='Train')
        plt.plot(endog.index[len(train_data_scaled) + seq_length:len(train_data_scaled) + seq_length + len(y_test)], y_test, label='Test', color='red')
        plt.plot(endog.index[len(train_data_scaled) + seq_length:len(train_data_scaled) + seq_length + len(y_test_pred)], y_test_pred, label='Forecast', color='green')
        plt.legend()
        plt.title(f'GRU Model Forecast vs Actuals for {target_column}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()


###############################################################################
# ------------ SELECTING RNN MODEL And Applying Random Forest Regression-------
###############################################################################

from sklearn.ensemble import RandomForestRegressor


# List of external factors and coal prices
external_factors = [
    'Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua',
    'Price_All_Share', 'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD',
    'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate'
]

original_coal_prices = [
    'Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD', 'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 'Coal_India_5500_CFR_London_Close_USD'
]

# Assuming coaldata is a Pandas DataFrame
X = coaldata[external_factors]

# Define train and test data
train_X = X.head(977)
test_X = X.tail(130)

# Initialize a dictionary to store MAPE values
mape_values = {}

# Loop through each coal price column to fit the model and calculate MAPE
for coal_price in original_coal_prices:
    y = coaldata[coal_price]
    train_y = y.head(977)
    test_y = y.tail(130)

    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(train_X, train_y)

    # Make predictions
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)

    # Calculate MAPE for training and testing data
    train_mape = mean_absolute_percentage_error(train_y, train_predictions)
    test_mape = mean_absolute_percentage_error(test_y, test_predictions)

    # Store the MAPE values
    mape_values[coal_price] = {
        'Training MAPE': train_mape,
        'Testing MAPE': test_mape
    }

    print(f"{coal_price} - Training MAPE: {train_mape:.2f}")
    print(f"{coal_price} - Testing MAPE: {test_mape:.2f}")












