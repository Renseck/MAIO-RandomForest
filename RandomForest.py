# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:37:09 2023

@author: rens_
"""
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

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
                               cv = 5, n_jobs = -1, scoring = scoring, verbose = 0)
    grid_search.fit(x_data, y_data)
    
    print("Best parameters: {}".format(grid_search.best_params_))
    return grid_search.best_estimator_

def clean_data(dataframe, verbose = True):
    before = dataframe.isna().sum()
    
    nan_stretches = {}
    for col in dataframe.columns:
        nan_stretch = []
        for idx, value in enumerate(dataframe[col].isna()):
            if value:
                nan_stretch.append(idx)
            elif nan_stretch:
                if len(nan_stretch) > 1:
                    nan_stretches[col] = nan_stretches.get(col, []) + nan_stretch
                nan_stretch = []

        # Handle the case where the NaN stretch ends at the last index
        if len(nan_stretch) > 1:
            nan_stretches[col] = nan_stretches.get(col, []) + nan_stretch

    # Create a list of indices with adjacent neighbors
    valid_indices = set()
    for indices in nan_stretches.values():
        valid_indices.update(indices)
        valid_indices.update(idx - 1 for idx in indices)
        valid_indices.update(idx + 1 for idx in indices)

    # Filter out singular indices, because we're going to interpolate these after we evict the 
    # multi-length nan holes
    nan_stretches = {col: [idx for idx in indices if idx in valid_indices] for col, indices in nan_stretches.items()}
    
    nan_indices = list(chain(*nan_stretches.values()))
    dataframe = dataframe.drop(nan_indices, axis = 0)
    intermediate = dataframe.isna().sum()
    
    # Now that those pesky big holes are gone, we can peacefully interpolate the small holes
    dataframe.loc[:, ~dataframe.columns.isin(["date", "date_end"])] = dataframe.loc[:, ~dataframe.columns.isin(["date", "date_end"])].interpolate(method = "linear")
    after = dataframe.isna().sum()
    
    if verbose:
        result = pd.concat([before, intermediate, after], axis = 1).rename(columns = {0: "Before", 1: "Hole deletion", 2: "Interpolation"}).replace(np.nan, 0).astype(int)
        print("Data cleanup readout:\n")
        print("Number of NaNs in data after each column title: \n", result)

    return dataframe

# %%% The start of the actual stuff

# Read the data into a dataframe
df = pd.read_csv("Rural_NL10644-AQ-METEO.csv", sep=";")
target = "o3"
relevant_cols = ["date", "date_end", "wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh", target]

# Split off unnecessarry columns but store them in singular variables (for speed)
siteKeys = ["site", "name", "lat", "lon", "organisation", "type", "site_eu", "type_alt",
            "type_airbase", "municipality.name", "province.name", "air.quality.area"]
siteValues = []
for key in siteKeys:
    siteValues.append(df[key].iloc[0]) 
    
site = Site(*siteValues)
df = df.drop(siteKeys, axis = 1)

df["date"] = pd.to_datetime(df["date"])
df["date_end"] = pd.to_datetime(df["date_end"])

df = clean_data(df[relevant_cols])
    
start_index = 0
end_index = df["date_end"][df["date_end"] == pd.to_datetime("2018-01-01 00:00:00")].index[0]
df_train = df[start_index:end_index]
df_pred = df[end_index:]

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [15, 20, 25],
    'max_features': ["auto", "sqrt"],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 5, 8],
    'n_estimators': [30, 40, 50]
}

# Read in the data for training (x_train, y_train for 2015-2017) and for prediction (x_pred for 2018)
#x_train(n_data, n_variables)  = 2d-array with explanatory variables 
#y_train(n_data)                      = 1d-array with observations
#x_pred(n_data2, n_variables)= 2d-array with explanatory variables 

X = df_train[["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]]
y = df_train[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
x_pred = df_pred[["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]]
y_hat = df_pred[target].values # For comparison

# Define the Random Forest model
model = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, random_state=42)
tuned_model = hypertune(param_grid, x_train, y_train, scoring = "r2")

# Fit the Random Forest model on the whole dataset
model.fit(x_train, y_train) 

# Determine the R2 score of the models
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
tuned_train_score = tuned_model.score(x_train, y_train)
tuned_test_score = tuned_model.score(x_test, y_test)
print('R2 train score: {:6.3f}' .format(train_score))
print('R2 test score: {:6.3f}' .format(test_score))
print("Tuned R2 train score: {:6.3f}".format(tuned_train_score))
print("Tuned R2 test score: {:6.3f}".format(tuned_test_score))

# Make a prediction for x_pred for 2018
y_pred = model.predict(x_pred)
y_pred_tuned = tuned_model.predict(x_pred)

# Plot data and Calculate statistics by comparing y_pred with observation for 2018
#==========================================
plt.figure(figsize = (10,7))
plt.title(r"Ozone concentration ($n_{{estimators}} = {{{n_estimators}}}$)".format(n_estimators = model.n_estimators))

plt.plot(df_pred["date"], y_pred, label = "Prediction")
plt.plot(df_pred["date"], y_pred_tuned, label = "Prediction tuned", alpha = 0.5)
plt.plot(df_pred["date"], y_hat, label = "Observation", alpha = 0.5)

plt.xlabel("Date")
plt.xticks(rotation = 45)
plt.ylabel("Concentration $[\mu$g $m^{-3}]$")
plt.legend()
plt.show()

corr = np.corrcoef(y_pred, y_hat)[0,1]
print(f"Correlation: {corr:6.3f}")
