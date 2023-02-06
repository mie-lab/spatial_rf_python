# Standard and GIS Modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score, mean_squared_error
from models import *


def create_data(nr_data, nr_feats=5, rho=0.6, weight_matrix_cutoff=20):

    x_coords = np.random.rand(nr_data, 2) * 2 - 1

    all_feats = np.zeros((nr_data, nr_feats))

    for feat in range(nr_feats):
        att = np.random.uniform(-1, 1, nr_data)

        w = get_weights_as_array(x_coords, weight_matrix_cutoff)
        # compute I - rho*W
        m = np.identity(len(x_coords)) - rho * w
        # invert and multiply with x_j
        att_hat = np.matmul(np.linalg.inv(m), att)
        # scale to -1 to 1
        att_hat = (att_hat - np.min(att_hat)) / (
            np.max(att_hat) - np.min(att_hat)
        ) * 2 - 1
        all_feats[:, feat] = att_hat
    return np.hstack([x_coords, all_feats])


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


def non_linear_1(feat_arr, weights):
    feature_transformed = np.zeros(feat_arr.shape)
    a, b, c, d, e = (
        feat_arr[:, 0],
        feat_arr[:, 1],
        feat_arr[:, 2],
        feat_arr[:, 3],
        feat_arr[:, 4],
    )
    # first term: a**2 * b
    feature_transformed[:, 0] = a ** 2 * np.sin(b) * weights[:, 0]
    feature_transformed[:, 1] = np.sin(b) * np.log(c ** 2) * d * weights[:, 1]
    feature_transformed[:, 2] = e ** 3 * np.log(c ** 2) * weights[:, 2]
    feature_transformed[:, 3] = d ** 2 * np.cos(b) * weights[:, 3]
    feature_transformed[:, 4] = e * a * d * weights[:, 4]

    return np.sum(feature_transformed, axis=1)


def non_linear_2(feat_arr, weights):
    feature_transformed = np.zeros(feat_arr.shape)
    a, b, c, d, e = (
        feat_arr[:, 0],
        feat_arr[:, 1],
        feat_arr[:, 2],
        feat_arr[:, 3],
        feat_arr[:, 4],
    )
    # first term: a**2 * b
    feature_transformed[:, 0] = a ** 2 * np.sin(b) * weights[:, 0]
    feature_transformed[:, 1] = np.sin(b) * d * weights[:, 1]
    feature_transformed[:, 2] = e * np.log(c ** 2) * weights[:, 2]
    feature_transformed[:, 3] = d ** 2 * np.cos(b) * weights[:, 3]
    feature_transformed[:, 4] = e * a ** 2 * d * weights[:, 4]

    return np.sum(feature_transformed, axis=1)


def non_linear_3(feat_arr, weights):
    feature_transformed = np.zeros(feat_arr.shape)
    a, b, c, d, e = (
        feat_arr[:, 0],
        feat_arr[:, 1],
        feat_arr[:, 2],
        feat_arr[:, 3],
        feat_arr[:, 4],
    )
    # first term: a**2 * b
    feature_transformed[:, 0] = a ** 2 * weights[:, 0]
    feature_transformed[:, 1] = b * a * weights[:, 1]
    feature_transformed[:, 2] = np.log(c ** 2) * weights[:, 2]
    feature_transformed[:, 3] = c * e * weights[:, 3]
    feature_transformed[:, 4] = np.sin(d) * weights[:, 4]

    return np.sum(feature_transformed, axis=1)


# parameters and models to include
np.random.seed(42)
noise_level_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
locality_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
model_function_names = [
    linear_regression,
    rf_coordinates,
    rf_global,
    rf_spatial,
    my_gwr,
    kriging,
    sarm,
    slx
    # rf_geographical,
]
model_names = [
    "linear regression",
    "RF (coordinates)",
    "RF",
    "spatial RF",
    "GWR",
    "Kriging",
    "SAR",
    "SLX"
    # "geographical RF",
]

# MAIN PARAMETERS
nr_feats = 5
max_depth = 30
n_estim = 150
w_cutoff = 20
rho = 0.75
noise_type = "uniformly distributed"
# 'heterogeneous - same', 'heterogeneous - different'

# save results
results_list = []

weights = np.array([-0.95, 0.38, 0.66, -0.43, 0.22])

for nr_data in [100, 500, 1000, 5000]:
    print("\n ======== DATA SAMPLES", nr_data)

    # MAKE MAIN DATA
    train_cutoff = int(nr_data * 0.9)
    feat_cols = ["feat_" + str(i) for i in range(nr_feats)]
    # V1: X random uniform
    # synthetic_data_array = np.random.rand(nr_data, 2 + nr_feats) * 2 - 1
    # V2: with spatial lag
    synthetic_data_array = create_data(
        nr_data, nr_feats=nr_feats, rho=rho, weight_matrix_cutoff=w_cutoff
    )
    print(synthetic_data_array.shape)

    synthetic_data = pd.DataFrame(
        synthetic_data_array, columns=["x_coord", "y_coord"] + feat_cols,
    )
    print(synthetic_data.head(5))
    # Double check Moran's I
    w = get_weights_as_array(synthetic_data_array[:, :2], w_cutoff)
    for t in range(5):
        print(
            "Moran's I of coefficient",
            t,
            morans_i(synthetic_data_array[:, t + 2], w),
        )

    # simulate spatial variation of features (varying per weight)
    spatial_variation = np.zeros((nr_data, nr_feats))
    for i in range(nr_feats):
        spatial_variation[:, i] = 0.5 * (
            np.sin(synthetic_data["x_coord"].values * np.pi * 2 + i)
            + np.cos(synthetic_data["y_coord"].values * np.pi * 2 + i)
        )

    for noise_level in noise_level_range:
        for locality in locality_range:
            # spatially dependent but linear
            spatially_dependent_weights = weights + locality * spatial_variation

            for mode in ["non-linear 1", "non-linear 2", "non-linear 3"]:
                print("--------", noise_level, locality, mode)
                # apply linear or non_linear function
                # if mode == "linear":
                #     synthetic_data["label"] = np.sum(
                #         spatially_dependent_weights
                #         * synthetic_data[feat_cols].values,
                #         axis=1,
                #     )
                # elif "simple" in mode:
                #     synthetic_data["label"] = non_linear_function_simple(
                #         synthetic_data[feat_cols].values,
                #         spatially_dependent_weights,
                #     )
                if mode == "non-linear 1":
                    synthetic_data["label"] = non_linear_1(
                        synthetic_data[feat_cols].values,
                        spatially_dependent_weights,
                    )
                elif mode == "non-linear 2":
                    synthetic_data["label"] = non_linear_2(
                        synthetic_data[feat_cols].values,
                        spatially_dependent_weights,
                    )
                else:
                    synthetic_data["label"] = non_linear_3(
                        synthetic_data[feat_cols].values,
                        spatially_dependent_weights,
                    )

                if noise_type == "uniformly distributed":
                    noise = np.random.normal(0, noise_level, nr_data)
                elif noise_type == "heterogeneous - different":
                    spatial_variation_different = noise_level * (
                        0.5
                        * (
                            synthetic_data["x_coord"].values
                            + synthetic_data["y_coord"].values
                        )
                        + 1
                    )
                    noise = np.random.normal(
                        0,
                        spatial_variation_different,
                        len(spatial_variation_different),
                    )
                elif noise_type == "heterogeneous - same":
                    # e.g. high noise level (0.5), spatial variation is from
                    # sin and cos so it's between -1 and 1, so we make + 1
                    # so on average we multiply by 1, but varying variance
                    # between 0.5 * 0 and 0.5 * 2
                    spatially_dependent_noise = noise_level * (
                        spatial_variation[:, 0] + 1  # without locality level!
                    )
                    noise = np.random.normal(
                        0, spatially_dependent_noise, nr_data
                    )
                else:
                    raise RuntimeError("Noise must be one of above")

                synthetic_data["label"] = synthetic_data["label"] + noise

                train_data, test_data = (
                    synthetic_data[:train_cutoff],
                    synthetic_data[train_cutoff:],
                )

                for model_function, name in zip(
                    model_function_names, model_names
                ):
                    tic = time.time()
                    test_pred = model_function(
                        train_data.copy(),
                        test_data.copy(),
                        # train_data.copy(), # for overfitting test
                        feat_cols=feat_cols,
                        max_depth=max_depth,
                        nr_data=nr_data,
                        n_estim=n_estim,
                        w_cutoff=w_cutoff,
                    )
                    # compute metrics
                    score = r2_score(test_data["label"], test_pred)
                    rmse = mean_squared_error(
                        test_data["label"], test_pred, squared=False
                    )
                    # train_data["label"]) # for overfitting test
                    time_diff = time.time() - tic
                    # add to results
                    results_list.append(
                        {
                            "nr_data": nr_data,
                            "noise": noise_level,
                            "locality": locality,
                            "data mode": mode,
                            "model": name,
                            "time": time_diff,
                            "R2 score": score,
                            "RMSE": rmse,
                        }
                    )
                    print(name, round(rmse, 3))

        results = pd.DataFrame(results_list)
        results["noise_type"] = noise_type
        noise_name = "_".join(noise_type.split(" "))
        results.to_csv(f"synthetic_data_results_{noise_name}.csv", index=False)
        print("Saved intermediate results")
