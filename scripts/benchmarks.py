# Standard and GIS Modules
import os
import sys
import numpy as np
import pandas as pd
import time

# ignore linalg warnings from MGWR package
import warnings

warnings.filterwarnings("ignore")

# gwr:
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sprf.spatial_random_forest import SpatialRandomForest
from sprf.geographical_random_forest import GeographicalRandomForest

from models import *


def get_folds(nr_samples, nr_folds=10):
    fold_inds = np.random.permutation(nr_samples)
    num_per_fold = nr_samples // nr_folds
    train_inds, test_inds = [], []
    for i in range(nr_folds):
        # print("start, end", i*num_per_fold)
        if i < nr_folds - 1:
            test_inds_fold = np.arange(
                i * num_per_fold, (i + 1) * num_per_fold, 1
            )
        else:
            test_inds_fold = np.arange(i * num_per_fold, nr_samples)
        test_inds.append(fold_inds[test_inds_fold])
        train_inds.append(np.delete(fold_inds, test_inds_fold))
    return train_inds, test_inds


def prepare_data(data, target, lon="x", lat="y"):
    """Assumes that all other columns are used as covariates"""
    # covariates = [col for col in data.columns if col not in [lon, lat, target]]
    # return data[covariates], data[target], data[[lon, lat]]
    return data.rename(
        columns={target: "label", lon: "x_coord", lat: "y_coord"}
    )


def add_metrics(test_pred, test_y, res_dict_init, method, runtime):
    res_dict = res_dict_init.copy()
    res_dict["Method"] = method
    res_dict["RMSE"] = mean_squared_error(test_pred, test_y, squared=False)
    res_dict["MAE"] = mean_absolute_error(test_pred, test_y)
    res_dict["R-Squared"] = r2_score(test_y, test_pred)
    res_dict["Runtime"] = runtime
    return res_dict


def cross_validation(data):
    nr_folds = 5
    train_inds, test_inds = get_folds(len(data), nr_folds=nr_folds)
    res_df = []

    # dataset specific information
    target = dataset_target[DATASET]
    x_coord_name = dataset_x.get(DATASET, "x")
    y_coord_name = dataset_y.get(DATASET, "y")

    # model params --> TODO: grid search
    max_depth = 10
    spatial_neighbors = len(data) // 5  # one fifth of the dataset
    print("Number of neighbors considered for spatial RF:", spatial_neighbors)

    for fold in range(nr_folds):
        res_dict_init = {"fold": fold, "max_depth": max_depth}
        train_data = data.iloc[train_inds[fold]]
        test_data = data.iloc[test_inds[fold]]
        train_data_renamed = prepare_data(
            train_data, target, x_coord_name, y_coord_name
        )
        test_data_renamed = prepare_data(
            test_data, target, x_coord_name, y_coord_name
        )
        feat_cols = [
            col
            for col in train_data_renamed.columns
            if "coord" not in col and col != "label"
        ]
        # print(
        #     train_x.shape, train_y.shape, train_coords.shape, test_x.shape,
        #     test_y.shape, test_coords.shape
        # )
        for model_function, name in zip(model_function_names, model_names):
            tic = time.time()
            test_pred = model_function(
                train_data_renamed.copy(),
                test_data_renamed.copy(),
                feat_cols=feat_cols,
            )
            runtime = time.time() - tic
            res_df.append(
                add_metrics(
                    test_pred,
                    test_data_renamed["label"],
                    res_dict_init,
                    name,
                    runtime,
                )
            )
            print(name, res_df[-1]["R-Squared"])

    # Finalize results
    res_df = pd.DataFrame(res_df)
    return res_df


os.makedirs("outputs", exist_ok=True)

dataset_target = {
    "plants": "richness_species_vascular",
    "meuse": "zinc",
    "atlantic": "Rate",
    "deforestation": "deforestation_quantile",
    "california_housing": "median_house_value",
}

model_function_names = [
    linear_regression,
    rf_coordinates,
    rf_global,
    rf_spatial,
    my_gwr,
    kriging,
    sarm
    # rf_geographical,
]
model_names = [
    "linear regression",
    "RF (coordinates)",
    "RF",
    "spatial RF",
    "GWR",
    "Kriging",
    "SAR"
    # "geographical RF",
]

datasets = [
    "meuse",
    "plants",
    "atlantic",
    "deforestation",
    "california_housing",
]

for DATASET in datasets:
    print("\nDATASET", DATASET, "\n")

    dataset_x = {}  # per default: x
    dataset_y = {}  # per default: y
    data_path = os.path.join("data", DATASET + ".csv")

    data = pd.read_csv(data_path)
    print("Number of samples", len(data))

    results = cross_validation(data)
    results.to_csv(
        os.path.join("outputs", f"results_{DATASET}_folds.csv"), index=False
    )

    results_grouped = (
        results.groupby(["Method"])
        .mean()
        .drop(["fold", "max_depth"], axis=1)
        .sort_values("RMSE")
    )
    results_grouped.to_csv(
        os.path.join("outputs", "real_jan_22", f"results_{DATASET}.csv")
    )

    print(results_grouped)
    print("--------------")
