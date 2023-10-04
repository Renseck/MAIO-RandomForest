# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:37:09 2023

@author: rens_
"""
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from tabulate import tabulate

full_variable_names = {"wd": "Wind direction",
              "ws": "Wind speed", 
              "t": "Temperature",
              "q": "Global radiation",
              "hourly_rain": "Precipitation (hourly)",
              "p": "Pressure",
              "n": "Cloud cover",
              "rh": "Relative humidity"}

class Site():
    # This is just used to store some site-specific information, in case it ever
    # becomes useful for plotting or something.
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

def clean_data(dataframe, verbose = True):
    """
    Remove NaNs of len > 1; interpolate rest

    Parameters
    ----------
    dataframe : DATAFRAME
       Pandas DataFrame
    verbose : BOOL, optional
        controls printing of output

    Returns
    -------
    dataframe : DATAFRAME
        Pandas DataFrame with all Nans removed

    """
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

def hypertune(grid, x_data, y_data, scoring = None):
    """
    Tune hyperparameters of sklearn randomforestregressor.

    Parameters
    ----------
    grid : DICT
        Parameters (with ranges) to be optimized.
    x_data : DATAFRAME
        X-data to be trained.
    y_data : DATAFRAME
        Y-data to be trained.
    scoring : STR, optional
        Scoring scheme to use. The default is None.

    Returns
    -------
    sklearn randomforestregressor
        Tuned random forest regressor model.

    """
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf, param_grid = grid,
                               cv = 5, n_jobs = -1, scoring = scoring, verbose = 0)
    grid_search.fit(x_data, y_data)
    
    print("Best parameters: {}".format(grid_search.best_params_))
    return grid_search.best_estimator_

# Credit to https://www.kaggle.com/code/dedecu/cross-correlation-time-lag-with-pandas/notebook
def df_derived_by_shift(df,lag=0,NON_DER=[]):
    """
    Generate time-shifted columns.

    Parameters
    ----------
    df : DATAFRAME
        Input DataFrame with columns to-be-shifted.
    lag : INT, optional
        Number of time shifts (up to and including argument). The default is 0.
    NON_DER : LIST, optional
        Columns to be excluded from shifting. The default is [].

    Returns
    -------
    df : DATAFRAME
        DataFrame with included time shifted columns.

    """
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1).reindex(df.index)
    return df

def show_cross_correlation(df, target):
    """
    Shows maximum correlation for time shifted variables and target.

    Parameters
    ----------
    df : DATAFRAME
        Input dataframe containing variables and target.
    target : STRING
        Target variable (column name).

    Returns
    -------
    correls : LIST
        List of maximally correlated variable per "category".

    """
    df_new = df_derived_by_shift(df, 6, ["date", "date_end", target])
    correlation = df_new.corr()
    meteo_vars = ["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]
    
    correls = []
    print("Value after underscore shows the amount of time-steps. I.e., 1 means shifted by 1 step/hour.")
    for meteo_var in meteo_vars:
        correl = correlation.loc[target].filter(regex = meteo_var)
        max_correl = correl.abs().idxmax()
        #print(f"{correl}")
        print(f"Max correlation between {meteo_var} and {target}: {max_correl}")
        correls.append(max_correl)
        
    return correls

def add_shifts(df, train_cols, target):
    """
    Add time shifted variables to DataFrame.

    Parameters
    ----------
    df : DATAFRAME
        Input DataFrame containing variables and target.
    train_cols : LIST
        List of training columns.
    target : STRING
        Name of target.

    Returns
    -------
    df : DATAFRAME
        DataFrame with shifted columns added in.
    train_cols : LIST
        List of "new" training columns.

    """
    max_correl_shifts = show_cross_correlation(df, target)
    
    for new_col in max_correl_shifts:
        splitup = new_col.split("_")
        if len(splitup) == 1:
            pass
        elif len(splitup) == 2:
            df[new_col] = df[splitup[0]].shift(int(splitup[-1]))
            train_cols.append(new_col)
        else:
            df[new_col] = df[splitup[0] + "_" + splitup[1]].shift(int(splitup[-1]))
            train_cols.append(new_col)    
            
    df = df.dropna()    

    return df, train_cols

def run(df, num_trees, features, plot = True):
    """
    Single model run

    Parameters
    ----------
    df : DATAFRAME
        DataFrame containing training and target data.
    num_trees : INT
        Number of trees for RandomForestRegressor model.
    features : LIST
        What features to train on.
    plot : BOOL, optional
        Show plots or not. The default is True.

    Returns
    -------
    y_pred_train : NP.ARRAY
        Array of predicted data in training period.
    y_pred : NP.ARRAY
        Array of predicted data in prediction period.

    """
    start_index = 0
    end_index = df["date_end"][df["date_end"] == pd.to_datetime("2018-01-01 00:00:00")].index[0]
    df_train = df[start_index:end_index]
    df_pred = df[end_index:]
    
    X = df_train[features]
    y = df_train[target]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    x_pred = df_pred[features]
    y_hat = df_pred[target].values # For comparison

    # Define the Random Forest model
    model = RandomForestRegressor(n_estimators=num_trees, min_samples_leaf=5, random_state=42)

    # Fit the Random Forest model on the whole dataset
    model.fit(x_train, y_train) 

    # Determine the R2 score of the models
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    # print('R2 train score: {:6.3f}' .format(train_score))
    # print('R2 test score: {:6.3f}' .format(test_score))
    # (Severe) overfitting?

    # Make a prediction for x_pred for 2018
    y_pred_train = model.predict(X)
    y_pred = model.predict(x_pred)

    # Plot data and calculate statistics by comparing y_pred with observation data
    #==========================================
    if plot:
        # Plot training period:
        plt.figure(figsize = (10,7))
        plt.title(r"Ozone concentration ($n_{{estimators}} = {{{n_estimators}}}$) (Training)".format(n_estimators = model.n_estimators))
    
        plt.plot(df_train["date"], y_pred_train, label = "Model")
        plt.plot(df_train["date"], y, label = "Observation", alpha = 0.5)
        plt.xlabel("Date")
        plt.xticks(rotation = 45)
        plt.ylabel("Concentration $[\mu$g $m^{-3}]$")
        plt.legend()
        plt.show()
        print(f"[TRAINING (trees = {model.n_estimators})] R2: {r2_score(y, y_pred_train):6.3f}")
        print(f"[TRAINING (trees = {model.n_estimators})] RMSE: {mean_squared_error(y, y_pred_train, squared = False):6.3f}")
        print(f"[TRAINING (trees = {model.n_estimators})] Correlation: {np.corrcoef(y, y_pred_train)[0][1]:6.3f}")
    
        # Plot prediction period (2018)
        plt.figure(figsize = (10,7))
        plt.title(r"Ozone concentration ($n_{{estimators}} = {{{n_estimators}}}$) (Prediction)".format(n_estimators = model.n_estimators))
    
        plt.plot(df_pred["date"], y_pred, label = "Model")
        plt.plot(df_pred["date"], y_hat, label = "Observation", alpha = 0.5)
    
        plt.xlabel("Date")
        plt.xticks(rotation = 45)
        plt.ylabel("Concentration $[\mu$g $m^{-3}]$")
        plt.legend()
        plt.show()
        print(f"[PREDICTION (trees = {model.n_estimators})] R2: {r2_score(y_hat, y_pred):6.3f}")
        print(f"[PREDICTION (trees = {model.n_estimators})] RMSE: {mean_squared_error(y_hat, y_pred, squared = False):6.3f}")
        print(f"[PREDICTION (trees = {model.n_estimators})] Correlation: {np.corrcoef(y_hat, y_pred)[0][1]:6.3f}")

    return y_pred_train, y_pred

def run_singles(df, num_trees, features):
    """
    Many runs of single features in [features].

    Parameters
    ----------
    df : DATAFRAME
        DataFrame containing training and target data.
    num_trees : INT
        Number of trees for RandomForestRegressor model.
    features : LIST
        What features to train on.
    plot : BOOL, optional
        Show plots or not. The default is True.

    Returns
    -------
    results_dict : DICT
        Predicted data trained on only given key.

    """
    start_index = 0
    end_index = df["date_end"][df["date_end"] == pd.to_datetime("2018-01-01 00:00:00")].index[0]
    df_train = df[start_index:end_index]
    df_pred = df[end_index:]

    X = df_train[features]
    y = df_train[target]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    x_pred = df_pred[features]
    y_hat = df_pred[target].values # For comparison

    results_dict = {"y_pred_train": {}, "y_pred": {}}
    for feature in features:
        feature_name = feature
        y_pred_train, y_pred = run(df, num_trees, [feature], plot = False)
        results_dict["y_pred_train"][feature] = y_pred_train
        results_dict["y_pred"][feature] = y_pred

    fig, axd = plt.subplot_mosaic([["wd", "ws", "t", "q"],
                                  ["hourly_rain", "p", "n", "rh"]], figsize = (11,8))

    table = []
    for key in axd.keys():
        axd[key].set_title(full_variable_names[key])
        axd[key].plot(df_train["date"], results_dict["y_pred_train"][key], label = "Model")
        axd[key].plot(df_train["date"], y, label = "Observation", alpha = 0.5)
        axd[key].tick_params("x", labelrotation = 45)
        
        rsquared = r2_score(y, results_dict["y_pred_train"][key])
        rmse = mean_squared_error(y, results_dict["y_pred_train"][key], squared = False)
        corr = np.corrcoef(y, results_dict["y_pred_train"][key])[0][1]
        table.append([full_variable_names[key], rsquared, rmse, corr])
        
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    axd["q"].legend(bbox_to_anchor = (1.05, 1.0))
    fig.suptitle("Effect of single variables on modelled data (training set)")
    plt.show()
    
    # Sort table from highest to lowest R2 score.
    table.sort(key = lambda x: x[1], reverse = True)
    print(tabulate(table, headers = ["Variable", "R^2", "RMSE", "Correlation"]))
    
    return results_dict

def bonus_run(df, train_cols, target):
    """
    This run includes time-shifted variables, based on which time-shift has the highest correlation
    to the observed target variable. The model is then trained on these additional shifted variables.

    Parameters
    ----------
    df : DATAFRAME
        Input dataframe containing (to-be-shifted) variables and target.
    train_cols : LIST
        List of training column names.
    target : STRING
        Name of target variable.

    Returns
    -------
    None.

    """
    df, train_pred_cols = add_shifts(df, train_cols, target)
    y_pred_train, y_train = run(df, 30, train_pred_cols, True)
    
    
# %%% The start of the actual stuff

# Read the data into a dataframe
df = pd.read_csv("data\\Rural_NL10644-AQ-METEO.csv", sep=";")
target = "o3"
train_pred_cols = ["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]
relevant_cols = ["date", "date_end"] + train_pred_cols + [target]

# Split off unnecessarry columns but store them in the Site class
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

X = df_train[train_pred_cols]
y = df_train[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
x_pred = df_pred[train_pred_cols]
y_hat = df_pred[target].values # For comparison    

run(df, 30, features = train_pred_cols, plot = True)
run(df, 20, features = train_pred_cols, plot = True)
run(df, 40, features = train_pred_cols, plot = True)
run_singles(df, 30, train_pred_cols)

bonus_run(df, train_pred_cols, target)
