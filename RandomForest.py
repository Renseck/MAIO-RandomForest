# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:37:09 2023

@author: rens_
"""
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Read the data into a dataframe
df = pd.read_csv("Rural_NL10644-AQ-METEO.csv", sep=";")
# Split off unnecessarry columns but store them in singular variables (for speed)
# May even delete these variables altogether if they turn out to be useless
siteNum = df["site"].iloc[0]
df = df.drop(["site"], axis =  1)
siteName = df["name"].iloc[0]
df = df.drop(["name"], axis = 1)
siteLat = df["lat"].iloc[0]
df = df.drop(["lat"], axis = 1)
siteLon = df["lon"].iloc[0]
df = df.drop(["lon"], axis = 1)
siteOrg = df["organisation"].iloc[0]
df = df.drop(["organisation"], axis = 1)
siteType = df["type"].iloc[0]
df = df.drop(["type"], axis = 1)
siteEu = df["site_eu"].iloc[0]
df = df.drop(["site_eu"], axis = 1)
siteTypeAlt = df["type_alt"].iloc[0]
df = df.drop(["type_alt"], axis = 1)
siteTypeAirbase = df["type_airbase"].iloc[0]
df = df.drop(["type_airbase"], axis = 1)
siteMunicipality = df["municipality.name"].iloc[0]
df = df.drop(["municipality.name"], axis = 1)
siteProvince = df["province.name"].iloc[0]
df = df.drop(["province.name"], axis = 1)
siteArea = df["air.quality.area"].iloc[0]
df = df.drop(["air.quality.area"], axis = 1)

df["date"] = pd.to_datetime(df["date"])
df["date_end"] = pd.to_datetime(df["date_end"])

for col in df.columns:
    # If the column is filled with only nans
    if df[col].isna().sum() == len(df[col]):
        df[col].fillna(0, inplace = True)
    else:
        df[col].fillna(df[col].mean(), inplace = True)
        # This part in particular needs to be more clever
    
start_index = 0
end_index = df["date_end"][df["date_end"] == pd.to_datetime("2018-01-01 00:00:00")].index[0]
df_train = df[start_index:end_index]
df_pred = df[end_index:]
    
# Read in the data for training (x_train, y_train for 2015-2017) and for prediction (x_pred for 2018)
#x_train(n_data, n_variables)  = 2d-array with explanatory variables 
#y_train(n_data)                      = 1d-array with observations
#x_pred(n_data2, n_variables)= 2d-array with explanatory variables 
target = "o3" 

X = df_train[["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]]
y = df_train[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
x_pred = df_pred[["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]]
y_hat = df_pred[target] # For comparison

# Define the Random Forest model
model = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, random_state=42)

# Fit the Random Forest model on the whole dataset
model.fit(x_train, y_train)

# Determine the R2 score of the model
score = model.score(x_train, y_train)
print('R2 score: {:6.3f}' .format(score))

# Make a prediction for x_pred for 2018
y_pred = model.predict(x_pred)

# Plot data and Calculate statistics by comparing y_pred with observation for 2018
#==========================================
plt.figure(figsize = (10,7))
plt.title("Ozone concentration")

plt.plot(df_pred["date"], y_pred, label = "Prediction")
plt.plot(df_pred["date"], y_hat.values, label = "Observation", alpha = 0.5)

plt.xlabel("Date")
plt.xticks(rotation = 45)
plt.ylabel("$\mu$g $m^{-3}$")
plt.legend()
