# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:37:09 2023

@author: rens_
"""
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Site():
    def __init__(self, siteNum, siteName, siteLat, siteLon, siteOrg, siteType, 
                 siteEu, siteTypeAlt, siteTypeAirbase, siteMunicipality, siteProvince,
                 siteArea):
        self.siteNum = siteNum
        self.siteName = siteName
        self.siteLat = siteLat
        self.siteLon = siteLon,
        self.siteOrg = siteOrg
        self.siteType = siteType
        self.siteEu = siteEu
        self.siteTypeAlt = siteTypeAlt
        self.siteTypeAirbase = siteTypeAirbase
        self.siteMunicipality = siteMunicipality
        self.siteProvince = siteProvince
        self.siteArea = siteArea

def hypertune(grid, x_data, y_data, scoring = None):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf, param_grid = grid,
                               cv = 5, n_jobs = -1, scoring = scoring, verbose = 10)
    grid_search.fit(x_data, y_data)
    
    print("Best parameters: {}".format(grid_search.best_params_))
    return grid_search.best_estimator_

# Read the data into a dataframe
df = pd.read_csv("Rural_NL10644-AQ-METEO.csv", sep=";")
# Split off unnecessarry columns but store them in singular variables (for speed)
# May even delete these variables altogether if they turn out to be useless
siteKeys = ["site", "name", "lat", "lon", "organisation", "type", "site_eu", "type_alt",
            "type_airbase", "municipality.name", "province.name", "air.quality.area"]
siteValues = []
for key in siteKeys:
    siteValues.append(df[key].iloc[0]) 
    
site = Site(*siteValues)
df = df.drop(siteKeys, axis = 1)

df["date"] = pd.to_datetime(df["date"])
df["date_end"] = pd.to_datetime(df["date_end"])

for col in df.drop(["date", "date_end"], axis = 1):
    # If the column is filled with only nans
    if df[col].isna().sum() == len(df[col]):
        df[col].fillna(0, inplace = True)
    else:
        df[col] = df[col].interpolate(method = 'linear')
    
start_index = 0
end_index = df["date_end"][df["date_end"] == pd.to_datetime("2018-01-01 00:00:00")].index[0]
df_train = df[start_index:end_index]
df_pred = df[end_index:]

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30],
    'max_features': [3],
    'min_samples_leaf': [3],
    'min_samples_split': [6],
    'n_estimators': [30, 40, 50]
}

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
tuned_model = hypertune(param_grid, x_train, y_train, scoring = "r2")

# Fit the Random Forest model on the whole dataset
model.fit(x_train, y_train) 

# Determine the R2 score of the model
score = model.score(x_train, y_train)
print('R2 score: {:6.3f}' .format(score))

# Make a prediction for x_pred for 2018
y_pred = model.predict(x_pred)
y_pred_tuned = tuned_model.predict(x_pred)

# Plot data and Calculate statistics by comparing y_pred with observation for 2018
#==========================================
plt.figure(figsize = (10,7))
plt.title(r"Ozone concentration ($n_{{estimators}} = {{{n_estimators}}}$)".format(n_estimators = model.n_estimators))

plt.plot(df_pred["date"], y_pred, label = "Prediction")
plt.plot(df_pred["date"], y_pred_tuned, label = "Prediction tuned", alpha = 0.5)
plt.plot(df_pred["date"], y_hat.values, label = "Observation", alpha = 0.5)

plt.xlabel("Date")
plt.xticks(rotation = 45)
plt.ylabel("Concentration $[\mu$g $m^{-3}]$")
plt.legend()
plt.show()

corr = np.corrcoef(y_pred, y_hat.values)[0,1]
print(f"Correlation: {corr:6.3f}")
