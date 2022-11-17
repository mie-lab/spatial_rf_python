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
    covariates = [col for col in data.columns if col not in [lon, lat, target]]
    return data[covariates], data[target], data[[lon, lat]]


def add_metrics(test_pred, test_y, res_dict_init, method, runtime):
    res_dict = res_dict_init.copy()
    res_dict["Method"] = method
    res_dict["MSE"] = mean_squared_error(test_pred, test_y)
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
        train_x, train_y, train_coords = prepare_data(
            train_data, target, x_coord_name, y_coord_name
        )
        test_x, test_y, test_coords = prepare_data(
            test_data, target, x_coord_name, y_coord_name
        )
        # print(
        #     train_x.shape, train_y.shape, train_coords.shape, test_x.shape,
        #     test_y.shape, test_coords.shape
        # )

        # Method 1: global RF
        tic = time.time()
        rf = RandomForestRegressor(max_depth=max_depth)
        rf.fit(train_x, train_y)
        test_pred = rf.predict(test_x)
        runtime = time.time() - tic
        res_df.append(
            add_metrics(
                test_pred, test_y, res_dict_init, "global_rf", runtime
            )
        )

        # Method 2: global RF with coordinates
        tic = time.time()
        rf = RandomForestRegressor(max_depth=max_depth)
        rf.fit(train_x.join(train_coords), train_y)
        test_pred = rf.predict(test_x.join(test_coords))
        runtime = time.time() - tic
        res_df.append(
            add_metrics(test_pred, test_y, res_dict_init, "coord_rf", runtime)
        )

        # Method 3: linear regression
        tic = time.time()
        lr = LinearRegression()
        lr.fit(train_x, train_y)
        test_pred = lr.predict(test_x)
        runtime = time.time() - tic
        res_df.append(
            add_metrics(
                test_pred, test_y, res_dict_init, "linear_regression", runtime
            )
        )

        # Method 4: Spatial RF:
        tic = time.time()
        sp = SpatialRandomForest(
            max_depth=max_depth, neighbors=spatial_neighbors
        )
        sp.fit(train_x, train_y, train_coords)
        test_pred = sp.predict(test_x, test_coords)
        runtime = time.time() - tic
        res_df.append(
            add_metrics(
                test_pred, test_y, res_dict_init, "spatial_rf", runtime
            )
        )

        # Method 5: Geographical RF (one RF per sample)
        tic = time.time()
        geo_rf = GeographicalRandomForest(
            n_estimators=10, neighbors=spatial_neighbors, max_depth=max_depth
        )
        geo_rf.fit(train_x, train_y, train_coords)
        test_pred = geo_rf.predict(test_x, test_coords)
        runtime = time.time() - tic
        res_df.append(
            add_metrics(test_pred, test_y, res_dict_init, "geo_rf", runtime)
        )

        # Method 6: GWR:
        try:
            tic = time.time()
            train_coords = np.array(train_coords)
            train_y = np.expand_dims(np.array(train_y), 1)
            train_x = np.array(train_x)
            # bandwidth selection
            # print(train_coords)
            # print()
            # print(train_y)
            # print()
            # print(train_x)
            # print("---------")
            gwr_selector = Sel_BW(
                train_coords,
                train_y,
                train_x,
                fixed=True,
                kernel="exponential"
            )
            gwr_bw = gwr_selector.search(criterion="AICc")
            # create and train model
            model = GWR(
                train_coords,
                train_y,
                train_x,
                gwr_bw,
                kernel="exponential",
                fixed=True
            )
            gwr_results = model.fit()
            # predict
            test_pred = model.predict(
                np.asarray(test_coords), test_x, gwr_results.scale,
                gwr_results.resid_response
            ).predictions
            runtime = time.time() - tic
            res_df.append(
                add_metrics(test_pred, test_y, res_dict_init, "GWR", runtime)
            )
        except:
            print("GWR NOT POSSIBLE FOR", DATASET)

    # Finalize results
    res_df = pd.DataFrame(res_df)
    return res_df


dataset_target = {
    "plants": "richness_species_vascular",
    "atlantic": "Rate",
    "deforestation": "deforestation_quantile",
    "meuse": "zinc",
    "california_housing": "median_house_value"
}

for DATASET in ["meuse", "california_housing"]:
    # "plants", "atlantic", "deforestation"]:
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

    results_grouped = results.groupby(
        ["Method"]
    ).mean().drop(["fold", "max_depth"], axis=1).sort_values("MSE")
    results_grouped.to_csv(os.path.join("outputs", f"results_{DATASET}.csv"))

    print(results_grouped)
    print("--------------")
