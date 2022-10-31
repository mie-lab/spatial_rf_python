from multiprocessing.sharedctypes import Value
import warnings
import numpy as np

from sklearn.ensemble import RandomForestRegressor


class GeographicalRandomForest:
    """
    Geographical Random Forest according to Georganos et al.
    
    Parameters
    ----------
    sample_by : str, optional {neighbors, distance}
        Sampling strategy. The spatial random forest consists of trees that are
        fitted on a spatial subset of samples. These spatial subsets can either
        be sampled by defining a distance-radius, or by specifying a fixed
        number of spatial neighbors. By default "neighbors", see notes below.
    neighbors : int, optional
        Number of neighbors to use for spatial fit, by default 500 samples.
        Only relevant if sample_by=neighbors.
    max_distance : int, optional
        Maximum distance of samples to belong to the same decision tree. Only
        relevant if sample_by=distance. By default 150000m
    """

    def __init__(
        self,
        sample_by: str = "neighbors",
        neighbors: int = 500,
        max_distance: float = 150000,
        **random_forest_arguments
    ):
        self.sample_by = sample_by
        if sample_by == "distance" and max_distance == 150000:
            warnings.warn(
                "It seems that you have selected the 'distance'-sampling mode,\
                     but the parameter max_distance is still the default. Make\
                     sure to adapt the max_distance parameter to your dataset."
            )
        self.max_distance = max_distance
        self.neighbors = neighbors
        self.random_forest_arguments = random_forest_arguments

    def fit(self, x_train, y_train, coords_train):
        # convert to arrays
        x_train, y_train, coords_train = (
            np.array(x_train),
            np.array(y_train),
            np.array(coords_train),
        )
        assert (
            len(coords_train.shape) == 2 and coords_train.shape[1] == 2
        ), "coords test must have len 2 in dimension 1"

        # init RFs
        self.random_forests = [
            RandomForestRegressor(**self.random_forest_arguments)
            for _ in range(len(x_train))
        ]

        # make distance matrix n x n
        dist = np.zeros((len(coords_train), len(coords_train)))
        for i, coord1 in enumerate(coords_train):
            for j, coord2 in enumerate(coords_train[i:]):
                dist[i, j + i] = np.linalg.norm(coord1 - coord2)
        # mirror distance matrix
        dist = dist + dist.T

        # save the train coordinates because they are needed for prediction
        self.rf_coords_train = coords_train

        # fit one random forest per sample
        for core_ind in range(len(x_train)):
            dist_to_others = dist[core_ind]
            if self.sample_by == "distance":
                samples_to_fit = np.where(dist_to_others < self.max_distance
                                          )[0]
            elif self.sample_by == "neighbors":
                sorted_inds = np.argsort(dist_to_others)
                samples_to_fit = sorted_inds[:self.neighbors]
            else:
                raise NotImplementedError(
                    "sample mode must be one of 'neighbors', 'distance'!"
                )
            x_train_subset = x_train[samples_to_fit]
            y_train_subset = y_train[samples_to_fit]
            self.random_forests[core_ind].fit(x_train_subset, y_train_subset)

    def predict(self, x_test, coords_test):
        x_test = np.array(x_test)
        coords_test = np.array(coords_test)

        # predict with a the closest random forest for each sample
        predictions = []
        for i in range(len(x_test)):
            dist_to_train_points = np.linalg.norm(
                self.rf_coords_train - coords_test[i], axis=1
            )
            closest_rf = np.argmin(dist_to_train_points)
            y_pred = self.random_forests[closest_rf].predict(
                x_test[i].reshape(1, -1)
            )
            predictions.append(y_pred)
        return np.array(predictions)
