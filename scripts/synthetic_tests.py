# Standard and GIS Modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sprf.spatial_random_forest import SpatialRandomForest
from sprf.geographical_random_forest import GeographicalRandomForest
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


def non_linear_function_simple(feat_arr, weights):
    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 0)
    function_zoo = [
        np.sin,
        np.exp,
        lambda x: x ** 2,
        lambda x: x,
        np.cos,
        lambda x: np.log(x ** 2),
    ]
    feature_transformed = np.zeros(feat_arr.shape)
    for i in range(feat_arr.shape[1]):
        feature_transformed[:, i] = (
            function_zoo[i](feat_arr[:, i]) * weights[:, i]
        )
    return np.sum(feature_transformed, axis=1)


def non_linear_function(feat_arr, weights):
    feature_transformed = np.zeros(feat_arr.shape)
    a, b, c, d, e = (
        feat_arr[:, 0],
        feat_arr[:, 1],
        feat_arr[:, 2],
        feat_arr[:, 3],
        feat_arr[:, 4],
    )
    # first term: a**2 * b
    feature_transformed[:, 0] = a ** 2 * b * weights[:, 0]
    feature_transformed[:, 1] = b * c * d * weights[:, 1]
    feature_transformed[:, 2] = e ** 3 * c * weights[:, 2]
    feature_transformed[:, 3] = d ** 2 * 2 * b * weights[:, 3]
    feature_transformed[:, 4] = e * a * d * weights[:, 4]

    return np.sum(feature_transformed, axis=1)


def linear_regression(train_data, test_data):
    regr = LinearRegression()
    regr.fit(train_data[feat_cols], train_data["label"])
    test_pred = regr.predict(test_data[feat_cols])
    return test_pred


def rf_coordinates(train_data, test_data):
    regr = RandomForestRegressor(max_depth=max_depth)
    regr.fit(
        train_data[["x_coord", "y_coord"] + feat_cols], train_data["label"]
    )
    test_pred = regr.predict(test_data[["x_coord", "y_coord"] + feat_cols])
    return test_pred


def rf_global(train_data, test_data):
    regr = RandomForestRegressor(max_depth=max_depth)
    regr.fit(train_data[feat_cols], train_data["label"])
    test_pred = regr.predict(test_data[feat_cols])
    return test_pred


def rf_spatial(train_data, test_data):
    n_estim = 100 if nr_data > 200 else 50
    regr = SpatialRandomForest(
        n_estimators=n_estim, neighbors=500, max_depth=max_depth
    )
    regr.tune_neighbors(
        train_data[feat_cols],
        train_data["label"],
        train_data[["x_coord", "y_coord"]],
    )
    print("spatial rf tuned:", regr.neighbors)
    regr.fit(
        train_data[feat_cols],
        train_data["label"],
        train_data[["x_coord", "y_coord"]],
    )
    test_pred = regr.predict(
        test_data[feat_cols], test_data[["x_coord", "y_coord"]]
    )
    return test_pred


def rf_geographical(train_data, test_data):
    n_estim = 20  # lower number of estimators to reduce runtime
    regr = GeographicalRandomForest(
        n_estimators=n_estim, neighbors=500, max_depth=max_depth
    )
    regr.tune_neighbors(
        train_data[feat_cols],
        train_data["label"],
        train_data[["x_coord", "y_coord"]],
    )
    print("geo rf tuned:", regr.neighbors)
    regr.fit(
        train_data[feat_cols],
        train_data["label"],
        train_data[["x_coord", "y_coord"]],
    )
    test_pred = regr.predict(
        test_data[feat_cols], test_data[["x_coord", "y_coord"]]
    )
    return test_pred


def my_gwr(train_data, test_data):
    try:
        train_coords = np.array(train_data[["x_coord", "y_coord"]])
        train_y = np.expand_dims(train_data["label"].values, 1)
        train_x = np.array(train_data[feat_cols])
        # bandwidth selection
        gwr_selector = Sel_BW(
            train_coords, train_y, train_x, fixed=True, kernel="exponential"
        )
        gwr_bw = gwr_selector.search(criterion="AICc")
        # create and train model
        gwr_model = GWR(
            train_coords,
            train_y,
            train_x,
            gwr_bw,
            kernel="exponential",
            fixed=True,
        )
        gwr_results = gwr_model.fit()

        test_coords = np.array(test_data[["x_coord", "y_coord"]])
        test_x = np.array(test_data[feat_cols])
        # predict
        test_pred = gwr_model.predict(
            test_coords, test_x, gwr_results.scale, gwr_results.resid_response
        ).predictions
        return test_pred
    except:
        print("GWR not possible")
        return np.zeros(len(test_data))


def add_results(score, name):
    results_list.append(
        {
            "nr_data": nr_data,
            "noise": noise_level,
            "locality": locality,
            "data mode": mode,
            "model": name,
            "time": time_diff,
            "R2 score": score,
        }
    )
    print(name, round(score, 3))


# parameters and models to include
noise_level_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
locality_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
model_function_names = [
    linear_regression,
    rf_coordinates,
    rf_global,
    rf_spatial,
    my_gwr,
    rf_geographical,
]
model_names = [
    "linear regression",
    "RF (coordinates)",
    "RF",
    "spatial RF",
    "GWR",
    "geographical RF",
]

# MAIN PARAMETERS
nr_feats = 5
max_depth = 30

# save results
results_list = []

weights = np.array([-0.95, 0.38, 0.66, -0.43, 0.22])

for nr_data in [100, 500, 1000, 5000]:
    print("\n ======== DATA SAMPLES", nr_data)

    # MAKE MAIN DATA
    train_cutoff = int(nr_data * 0.9)
    feat_cols = ["feat_" + str(i) for i in range(nr_feats)]
    synthetic_data = pd.DataFrame(
        np.random.rand(nr_data, 2 + nr_feats) * 2 - 1,
        columns=["x_coord", "y_coord"] + feat_cols,
    )

    # simulate spatial variation of features (varying per weight)
    spatial_variation = np.zeros((nr_data, nr_feats))
    for i in range(nr_feats):
        spatial_variation[:, i] = 0.5 * (
            np.sin(synthetic_data["x_coord"].values * np.pi * 1.5 + i)
            + np.cos(synthetic_data["y_coord"].values * np.pi * 1.5 + i)
        )

    for noise_level in noise_level_range:
        for locality in locality_range:
            # spatially dependent but linear
            spatially_dependent_weights = weights + locality * spatial_variation

            for mode in ["linear", "non-linear (simple)", "non-linear"]:
                print("--------", noise_level, locality, mode)
                # apply linear or non_linear function
                if mode == "linear":
                    synthetic_data["label"] = np.sum(
                        spatially_dependent_weights
                        * synthetic_data[feat_cols].values,
                        axis=1,
                    )
                elif "simple" in mode:
                    synthetic_data["label"] = non_linear_function_simple(
                        synthetic_data[feat_cols].values,
                        spatially_dependent_weights,
                    )
                else:
                    synthetic_data["label"] = non_linear_function(
                        synthetic_data[feat_cols].values,
                        spatially_dependent_weights,
                    )

                synthetic_data["label"] = synthetic_data[
                    "label"
                ] + np.random.normal(0, noise_level, nr_data)

                train_data, test_data = (
                    synthetic_data[:train_cutoff],
                    synthetic_data[train_cutoff:],
                )

                for model_function, name in zip(
                    model_function_names, model_names
                ):
                    tic = time.time()
                    test_pred = model_function(
                        train_data.copy(), test_data.copy()
                    )
                    score = r2_score(test_pred, test_data["label"])
                    time_diff = time.time() - tic
                    add_results(score, name)

        results = pd.DataFrame(results_list)
        results.to_csv("synthetic_data_results.csv", index=False)
        print("Saved intermediate results")
