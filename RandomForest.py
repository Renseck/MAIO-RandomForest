# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:37:09 2023

@author: rens_
"""
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.special import erf 
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from tabulate import tabulate
import concurrent.futures

DATA_PATH = "data"
RESULTS_PATH = "results"

full_variable_names = {"wd": "Wind direction",
                       "ws": "Wind speed",
                       "t": "Temperature",
                       "q": "Global radiation",
                       "hourly_rain": "Precipitation (hourly)",
                       "p": "Pressure",
                       "n": "Cloud cover",
                       "rh": "Relative humidity",
                       "o3": "Ozone"}

ylabels = {"wd": "Degrees",
           "ws": "m/s",
           "t": "$^{\circ}$C",
           "q": "J/cm$^2$",
           "hourly_rain": "mm",
           "p": "hPa",
           "n": "1/8",
           "rh": "%",
           "o3": "Concentration $[\mu$g $m^{-3}]$"}


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


def clean_data(dataframe, target, verbose=True):
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
    # With help from ChatGPT (only the NaN deletion part)
    before = dataframe.isna().sum()
    nan_stretches = {}
    for col in dataframe.columns:
        nan_stretch = []
        for idx, value in enumerate(dataframe[col].isna()):
            if value:
                nan_stretch.append(idx)
            elif nan_stretch:
                if len(nan_stretch) > 1:
                    nan_stretches[col] = nan_stretches.get(
                        col, []) + nan_stretch
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
    nan_stretches = {col: [idx for idx in indices if idx in valid_indices]
                     for col, indices in nan_stretches.items()}

    nan_indices = list(chain(*nan_stretches.values()))
    dataframe = dataframe.drop(nan_indices, axis=0)
    intermediate = dataframe.isna().sum()

    # Now that those pesky big holes are gone, we can peacefully interpolate the small holes
    dataframe.loc[:, ~dataframe.columns.isin(["date", "date_end"])] = dataframe.loc[:, ~dataframe.columns.isin([
        "date", "date_end"])].interpolate(method="linear")
    after = dataframe.isna().sum()

    if verbose:
        result = pd.concat([before, intermediate, after], axis=1).rename(columns={
            0: "Before", 1: "Hole deletion", 2: "Interpolation"}).replace(np.nan, 0).astype(int)
        print("Data cleanup readout:\n")
        print("Number of NaNs in data after each column title: \n", result)

    return dataframe


def show_data(df_train, df_pred):
    meteo_vars = list(full_variable_names.keys())

    fig, axd = plt.subplot_mosaic([["wd", "wd", "n", "n"], ["ws", "t", "q", "hourly_rain"],
                                  ["p", "p", "rh", "rh"], ["o3", "o3", "o3", "o3"]], figsize=(11, 8))

    for meteo_var in meteo_vars:
        axd[meteo_var].set_title(full_variable_names[meteo_var])
        axd[meteo_var].plot(df_train["date"], df_train[meteo_var], label="Training data")
        axd[meteo_var].plot(df_pred["date"], df_pred[meteo_var], label="Prediction data")
        axd[meteo_var].set_ylabel(ylabels[meteo_var])
        axd[meteo_var].tick_params("x", labelrotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0, "Date", ha="center", va="center", rotation=0)
    axd["wd"].legend(bbox_to_anchor=(0.9, 1.3))
    plt.savefig(os.path.join(RESULTS_PATH, "data_showcase.jpg"))
    plt.show()


def hypertune(grid, x_data, y_data, scoring=None):
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
    grid_search = GridSearchCV(estimator=rf, param_grid=grid,
                               cv=5, n_jobs=-1, scoring=scoring, verbose=0)
    grid_search.fit(x_data, y_data)

    print("Best parameters: {}".format(grid_search.best_params_))
    return grid_search.best_estimator_


def df_derived_by_shift(df, lag=0, NON_DER=[]):
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
    # Credit to https://www.kaggle.com/code/dedecu/cross-correlation-time-lag-with-pandas/notebook
    df = df.copy()
    if not lag:
        return df
    cols = {}
    for i in range(1, lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i += 1
        df = pd.concat([df, dfn], axis=1).reindex(df.index)
    return df


def show_cross_correlation(df, target, time_shift):
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
    df_new = df_derived_by_shift(df, time_shift, ["date", "date_end", target])
    time_shift_list = ["Base variable"] + \
        ["Time shift +{}".format(i+1) for i in range(time_shift)]
    correlation = df_new.corr()
    meteo_vars = ["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh"]

    correls = []
    table = []
    print("\nValue after underscore shows the amount of time-steps. I.e., 1 means shifted by 1 step/hour.")
    for meteo_var in meteo_vars:
        regex = "^" + meteo_var + "(_[0-9])?$"
        correl = correlation.loc[target].filter(regex=regex)
        table.append([meteo_var] + correl.to_list())
        max_correl = correl.abs().idxmax()
        # print(f"{correl}")
        print(
            f"Max correlation between {meteo_var} and {target}: {max_correl}")
        correls.append(max_correl)

    print("\n" + tabulate(table, headers=time_shift_list, tablefmt="github", floatfmt="6.3f"))
    return correls


def add_shifts(df, train_cols, target):
    """
    Add time shifted variables to DataFrame, as well as extra variables.

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
    max_correl_shifts = show_cross_correlation(df, target, time_shift=6)

    for new_col in max_correl_shifts:
        splitup = new_col.split("_")
        if len(splitup) == 1:
            pass
        elif len(splitup) == 2:
            df[new_col] = df[splitup[0]].shift(int(splitup[-1]))
            train_cols.append(new_col)
        else:
            df[new_col] = df[splitup[0] + "_" +
                             splitup[1]].shift(int(splitup[-1]))
            train_cols.append(new_col)

    df = df.dropna()

    train_cols.append("hour_of_day")
    train_cols.append("day_of_year")
    train_cols.append("season")

    return df, train_cols


def run(df, num_trees, features, target, plot=True, filename_addition=""):
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
    filename_addition: STR, optional
        Adds a little extra to a filename for extra distinction.

    Returns
    -------
    y_pred_train : NP.ARRAY
        Array of predicted data in training period.
    y_pred : NP.ARRAY
        Array of predicted data in prediction period.
    model : RandomForestRegressor()
        Model containing parameters.

    """
    start_index = 0
    end_index = df["date_end"][df["date_end"] ==
                               pd.to_datetime("2018-01-01 00:00:00")].index[0]
    df_train = df[start_index:end_index]
    df_pred = df[end_index:]

    X = df_train[features]
    y = df_train[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    x_pred = df_pred[features]
    y_hat = df_pred[target].values  # For comparison

    # Define the Random Forest model
    model = RandomForestRegressor(
        n_estimators=num_trees, min_samples_leaf=5, random_state=42)

    # Fit the Random Forest model on the whole dataset
    model.fit(x_train, y_train)

    # Determine the R2 score of the models
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    # (Severe) overfitting?

    # Make a prediction for x_pred for 2018
    y_pred_train = model.predict(X)
    y_pred = model.predict(x_pred)

    # Plot data and calculate statistics by comparing y_pred with observation data
    # ==========================================
    if plot:
        # Calculate statistical measures
        rsquared_train = r2_score(y, y_pred_train)
        rmse_train = mean_squared_error(y, y_pred_train, squared=False)
        corr_train = np.corrcoef(y, y_pred_train)[0][1]
        model_mean_train = np.mean(y_pred_train)
        model_stdev_train = np.std(y_pred_train)

        # Plot training period:
        fig, axd = plt.subplot_mosaic([["top"], ["bottom"]], figsize=(11, 7))
        fig.text(0.005, 0.5, "Ozone concentration $[\mu$g $m^{-3}]$",
                 ha="center", va="center", rotation=90)
        axd["top"].set_title(r"Ozone concentration ($n_{{estimators}} = {{{n_estimators}}}$) (Training)".format(
            n_estimators=model.n_estimators))
        axd["top"].plot(df_train["date"], y_pred_train, label="Model")
        axd["top"].plot(df_train["date"], y, label="Observation", alpha=0.5)
        label_text = "$R^2$ = {rsquared:6.3f} \nRMSE = {rmse:6.3f} \ncorr = {corr:6.3f}".format(
            rsquared=rsquared_train, rmse=rmse_train, corr=corr_train)
        axd["top"].text(df_train["date"].iloc[0], 170, label_text, color="black", bbox=dict(
            facecolor="none", edgecolor="gray", boxstyle="round"))
        axd["top"].legend()

        # And the same here for the next subplot
        rsquared_pred = r2_score(y_hat, y_pred)
        rmse_pred = mean_squared_error(y_hat, y_pred, squared=False)
        corr_pred = np.corrcoef(y_hat, y_pred)[0][1]
        model_mean_pred = np.mean(y_pred)
        model_stdev_pred = np.std(y_pred)

        axd["bottom"].set_title(r"Ozone concentration ($n_{{estimators}} = {{{n_estimators}}}$) (Prediction)".format(
            n_estimators=model.n_estimators))
        axd["bottom"].plot(df_pred["date"], y_pred, label="Model")
        axd["bottom"].plot(df_pred["date"], y_hat, label="Observation", alpha=0.5)
        axd["bottom"].set_xlabel("Date")
        label_text = "$R^2$ = {rsquared:6.3f} \nRMSE = {rmse:6.3f} \ncorr = {corr:6.3f}".format(
            rsquared=rsquared_pred, rmse=rmse_pred, corr=corr_pred)
        axd["bottom"].text(df_pred["date"].iloc[0], 157, label_text, color="black", bbox=dict(
            facecolor="none", edgecolor="gray", boxstyle="round"))
        axd["bottom"].legend()
        print(
            f"[TRAINING (trees = {model.n_estimators})] Train score, test score: {train_score:6.3f}, \t {test_score:6.3f}")
        print(f"[TRAINING (trees = {model.n_estimators})] R2: {rsquared_train:6.3f}")
        print(f"[TRAINING (trees = {model.n_estimators})] RMSE: {rmse_train:6.3f}")
        print(f"[TRAINING (trees = {model.n_estimators})] Correlation: {corr_train:6.3f}")
        print(f"[TRAINING (trees = {model.n_estimators})] Obs mean - model mean: {np.mean(y) - model_mean_train:6.3f}")
        print(
            f"[TRAINING (trees = {model.n_estimators})] Obs stdev - model stdev  : {np.std(y) - model_stdev_train:6.3f}\n")

        print(
            f"[PREDICTION (trees = {model.n_estimators})] R2: {rsquared_pred:6.3f}")
        print(f"[PREDICTION (trees = {model.n_estimators})] RMSE: {rmse_pred:6.3f}")
        print(
            f"[PREDICTION (trees = {model.n_estimators})] Correlation: {corr_pred:6.3f}")
        print(
            f"[PREDICTION (trees = {model.n_estimators})] Obs mean - model mean: {np.mean(y_hat) - model_mean_pred:6.3f}")
        print(
            f"[PREDICTION (trees = {model.n_estimators})] Obs stdev - model stdev  : {np.std(y_hat) - model_stdev_pred:6.3f}")

        filename = f"{target}_{model.n_estimators}_combined_results" + \
            filename_addition
        plt.savefig(os.path.join(
            RESULTS_PATH, filename + ".jpg"))
        plt.show()

    return y_pred_train, y_pred, model


def run_singles(df, num_trees, features, target):
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
    end_index = df["date_end"][df["date_end"] ==
                               pd.to_datetime("2018-01-01 00:00:00")].index[0]
    df_train = df[start_index:end_index]

    X = df_train[features]
    y = df_train[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # For comparison ^

    results_dict = {"y_pred_train": {}, "y_pred": {}}
    for feature in features:
        y_pred_train, y_pred, model = run(
            df, num_trees, [feature], target, plot=False)
        results_dict["y_pred_train"][feature] = y_pred_train
        results_dict["y_pred"][feature] = y_pred

    fig, axd = plt.subplot_mosaic([["wd", "ws", "t", "q"],
                                  ["hourly_rain", "p", "n", "rh"]], figsize=(11, 8), sharey=True)

    table = []
    for key in axd.keys():
        axd[key].set_title(full_variable_names[key])
        axd[key].plot(df_train["date"],
                      results_dict["y_pred_train"][key], label="Model")
        axd[key].plot(df_train["date"], y, label="Observation", alpha=0.5)
        axd[key].tick_params("x", labelrotation=45)

        rsquared = r2_score(y, results_dict["y_pred_train"][key])
        rmse = mean_squared_error(
            y, results_dict["y_pred_train"][key], squared=False)
        corr = np.corrcoef(y, results_dict["y_pred_train"][key])[0][1]
        mean_diff = np.mean(y) - np.mean(results_dict["y_pred_train"][key])
        std_diff = np.std(y) - np.std(results_dict["y_pred_train"][key])
        table.append([full_variable_names[key], rsquared, rmse, corr, mean_diff, std_diff])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    axd["n"].legend(bbox_to_anchor=(0.2, 1.1))
    fig.suptitle("Effect of single variables on modelled data (training set)")
    fig.text(0.5, 0.025, "Date", ha="center", va="center", rotation=0)
    fig.text(0.01, 0.5, "Ozone concentration $[\mu$g $m^{-3}]$",
             ha="center", va="center", rotation=90)
    plt.savefig(os.path.join(
        RESULTS_PATH, f"{target}_singles_comparison_Training.jpg"))
    plt.show()

    # Sort table from highest to lowest R2 score.
    table.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(table, headers=[
          "Variable", "R^2", "RMSE", "Correlation", "Mean diff", "Stdev diff"], tablefmt="github", floatfmt="6.3f"))

    return results_dict

def multithread_run(tree, fixed_args):
    """
    Multithreading helper function for the run() function

    Parameters
    ----------
    tree : INT
        Number of trees to run the model with.
    fixed_args : LIST
        List of fixed arguments to go in run().

    Returns
    -------
    y_pred_train : NP.ARRAY
        Array of predicted data in training period.
    y_pred : NP.ARRAY
        Array of predicted data in prediction period.
    model : RandomForestRegressor()
        Model containing parameters.

    """
    df, features, target = fixed_args
    y_pred_train, y_pred, model = run(df, tree, features, target, plot = False)
    return y_pred_train, y_pred, model

def r2_fitfunction(x, a, b, c, d):
    # Function to fit the r2 scores to, so we can get the asymptotic max value.
    return (a)/(b*x + c) + d

def trees_vs_r2(df, max_trees, features, target):
    start_index = 0
    end_index = df["date_end"][df["date_end"] ==
                               pd.to_datetime("2018-01-01 00:00:00")].index[0]
    df_train = df[start_index:end_index]
    df_pred = df[end_index:]
    
    X = df_train[features]
    y = df_train[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_hat = df_pred[target].values  # For comparison
    
    
    fixed_args = [df, train_pred_cols, target]
    trees = range(1, max_trees + 1)
    # Do some multithreading magic to cut down on computation time
    with concurrent.futures.ThreadPoolExecutor() as exec:
        max_workers = exec._max_workers
        print(f"Starting multithread execution of tree-varying model run, maximum of {max_workers} workers.\n")
        futures = tqdm([exec.submit(multithread_run, tree, fixed_args) for tree in trees])
        results = [future.result() for future in futures]
        
    train_scores = []
    pred_scores = []
    
    for result in results:
        y_pred_train, y_pred, model = result
        rsquared_train = r2_score(y, y_pred_train)
        rsquared_pred = r2_score(y_hat, y_pred)
        
        train_scores.append(rsquared_train)
        pred_scores.append(rsquared_pred)
        
    train_popt, _ = curve_fit(r2_fitfunction, np.array(trees), np.array(train_scores))
    pred_popt, _ = curve_fit(r2_fitfunction, np.array(trees), np.array(pred_scores))
    plt.figure(figsize = (11,7))
    plt.title("Relation between number of trees and $R^2$")
    plt.plot(trees, train_scores, label = "Training")
    plt.hlines(train_popt[-1], min(trees), max(trees), color = "blue", linestyle = "dashed", alpha = 0.7)
    plt.plot(trees, pred_scores, label = "Prediction")
    plt.hlines(pred_popt[-1], min(trees), max(trees), color = "orange", linestyle = "dashed", alpha = 0.7)
    plt.vlines(30, 0.6, 0.95, color = "grey", linestyle = "dashed", alpha = 0.8)
    plt.scatter(30, train_scores[29], alpha = 0.7)
    plt.scatter(30, pred_scores[29], alpha = 0.7 )
    plt.xlabel("Number of trees")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, f"{target}_trees_vs_r2_{max_trees}.jpg"))
    plt.show()

    return train_scores, pred_scores

def bonus_run(df, train_cols, target, plot=True, filename_addition=""):
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
    print("\nTHIS IS A BONUS RUN TO SEE IF ADDING\nTIMESHIFTED VARIABLES HELPS:")
    df, train_pred_cols = add_shifts(df, train_cols, target)
    y_pred_train, y_pred, model = run(
        df, 30, train_pred_cols, target, plot, filename_addition)
    return y_pred_train, y_pred, model


# %%% The start of the actual stuff

# Read the data into a dataframe
df = pd.read_csv(os.path.join(
    DATA_PATH, "Rural_NL10644-AQ-METEO.csv"), sep=";")
target = "o3"
train_pred_cols = ["wd", "ws", "t", "q", "hourly_rain", "p", "n", "rh", "month",
                   "day_of_month", "hour_of_day", "day_of_year", "season"]
relevant_cols = ["date", "date_end"] + train_pred_cols + [target]

# Split off unnecessarry columns but store them in the Site class
siteKeys = ["site", "name", "lat", "lon", "organisation", "type", "site_eu", "type_alt",
            "type_airbase", "municipality.name", "province.name", "air.quality.area"]
siteValues = []
for key in siteKeys:
    siteValues.append(df[key].iloc[0])

site = Site(*siteValues)
df = df.drop(siteKeys, axis=1)

df["date"] = pd.to_datetime(df["date"])
df["date_end"] = pd.to_datetime(df["date_end"])
# Add extra data that the regressor can actually read, because it can't deal with the
# timestamp data from 2 lines up.
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["day_of_month"] = df["date"].dt.day
df["hour_of_day"] = df["date"].dt.hour
df["season"] = df["date"].dt.month % 12 // 3 + 1

# Just remove negative values from the target variable.
print(f"Number of negative values in {target}: {len(df[df[target] < 0][target])}")
df = df.drop(df[df[target] < 0][target].index, axis=0)
df = df.reset_index(drop=True)

df = clean_data(df[relevant_cols], target)

start_index = 0
end_index = df["date_end"][df["date_end"] ==
                           pd.to_datetime("2018-01-01 00:00:00")].index[0]
df_train = df[start_index:end_index]
df_pred = df[end_index:]

X = df_train[train_pred_cols]
y = df_train[target]

# Show_case training data here
show_data(df_train, df_pred)

# Start splitting and training the models
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
x_pred = df_pred[train_pred_cols]
y_hat = df_pred[target].values  # For comparison

y_pred_train30, y_pred30, model30 = run(df, 30, features=train_pred_cols, target=target, plot=True)
y_pred_train1, y_pred1, model1 = run(df, 1, features=train_pred_cols, target=target, plot=True)
y_pred_train300, y_pred300, model300 = run(df, 300, features=train_pred_cols, target=target, plot=True)
results_dict = run_singles(df, 30, train_pred_cols, target)

train_scores, pred_scores = trees_vs_r2(df, 100, train_pred_cols, target)

y_pred_train_bonus, y_pred_bonus, model_bonus = bonus_run(df, train_pred_cols, target,
                                                              plot=True, filename_addition="_bonus")

# Final bit of plotting
feat_importance30 = pd.Series(model30.feature_importances_, index=X.columns)
feat_importance1 = pd.Series(model1.feature_importances_, index=X.columns)
feat_importance300 = pd.Series(model300.feature_importances_, index=X.columns)

df_importances = pd.concat([feat_importance30, feat_importance1, feat_importance300],
                           axis=1).rename(columns={0: "Default", 1: "Small", 2: "Large"})
df_importances["Feature"] = list(full_variable_names.values())[:-1] \
                            + ["Month", "Day of month", "Hour of day", "Day of year", "Season"]
ax = df_importances.sort_values("Default", ascending=False).plot(
    x="Feature", y=["Default", "Small", "Large"], kind="bar", figsize=(11, 7))
_ = plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.tight_layout()
ax.set_ylabel("Importance")
ax.set_xlabel("Feature")
ax.set_title("Normalised feature importances")
plt.savefig(os.path.join(RESULTS_PATH, "model_feature_importances.png"))
plt.show()
