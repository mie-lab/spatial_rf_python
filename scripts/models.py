import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sprf.spatial_random_forest import SpatialRandomForest
from sprf.geographical_random_forest import GeographicalRandomForest

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pykrige.rk import RegressionKriging
import spreg
import libpysal


def linear_regression(train_data, test_data, feat_cols=[], **kwargs):
    regr = LinearRegression()
    regr.fit(train_data[feat_cols], train_data["label"])
    test_pred = regr.predict(test_data[feat_cols])
    return test_pred


def rf_coordinates(train_data, test_data, feat_cols=[], max_depth=30, **kwargs):
    regr = RandomForestRegressor(max_depth=max_depth)
    regr.fit(
        train_data[["x_coord", "y_coord"] + feat_cols], train_data["label"]
    )
    test_pred = regr.predict(test_data[["x_coord", "y_coord"] + feat_cols])
    return test_pred


def rf_global(train_data, test_data, feat_cols=[], max_depth=30, **kwargs):
    regr = RandomForestRegressor(max_depth=max_depth)
    regr.fit(train_data[feat_cols], train_data["label"])
    test_pred = regr.predict(test_data[feat_cols])
    return test_pred


def rf_spatial(
    train_data, test_data, feat_cols=[], max_depth=30, nr_data=500, *kwargs
):
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


def rf_geographical(
    train_data, test_data, feat_cols=[], max_depth=30, **kwargs
):
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


def kriging(train_data, test_data, feat_cols=[], **kwargs):
    krig = RegressionKriging(RandomForestRegressor(), verbose=False)
    krig.fit(
        train_data[feat_cols].values,
        train_data[["x_coord", "y_coord"]].values,
        train_data["label"].values,
    )
    test_pred = krig.predict(
        test_data[feat_cols].values,
        test_data[["x_coord", "y_coord"]].values.astype(float),
    )
    return test_pred


def sarm(train_data, test_data, feat_cols=[], **kwargs):
    X = train_data[feat_cols].values
    Y = train_data["label"].values
    try:
        w = libpysal.weights.DistanceBand(
            train_data[["x_coord", "y_coord"]].values,
            threshold=0.5,
            binary=False,
        )
        model = spreg.GM_Lag(Y, X, w=w)
        test_pred = np.matmul(test_data[feat_cols].values, model.betas[1:-1])
    except:
        test_pred = np.zeros(len(test_data)) + np.mean(Y)
    return test_pred


def my_gwr(train_data, test_data, feat_cols=[], **kwargs):
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
