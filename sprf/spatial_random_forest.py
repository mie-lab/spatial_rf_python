import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor


class SpatialRandomForest:

    def __init__(
        self,
        nr_estimators=20,
        neighbors=500,
        sample_mode="cluster",
        sample_by="neighbors",
        min_points_distance=100,
        max_distance=150000,
        **decision_tree_args,
    ):
        self.estimators = [
            DecisionTreeRegressor(**decision_tree_args)
            for _ in range(nr_estimators)
        ]
        self.nr_estimators = nr_estimators
        self.sample_mode = sample_mode
        self.sample_by = sample_by
        # only relevant if sample_by == distance
        self.max_distance = max_distance
        self.min_points_distance = min_points_distance
        # only relevant if sample_by == "neighbors"
        self.neighbors = neighbors
        # init core points
        self.estimator_core_points = []

    def _sample_core_points(self, coords):
        if self.sample_mode == "cluster":
            kmeans = KMeans(self.nr_estimators)
            kmeans.fit(coords)
            core_points = kmeans.cluster_centers_
        # TODO: elif sample_mode == "grid":
        elif self.sample_mode == "random":
            core_points = coords[np.random.permutation(len(coords)
                                                       )[:self.nr_estimators]]
        else:
            raise NotImplementedError(
                "sample mode must be one of cluster, random"
            )
        return core_points

    def _sample_point_clouds(self, coords):
        point_clouds = []
        for core_point in self.estimator_core_points:
            dist_to_others = np.sqrt(np.sum((coords - core_point)**2, axis=1))
            if self.sample_by == "neighbors":
                point_clouds.append(
                    np.argsort(dist_to_others)[:self.neighbors]
                )
            elif self.sample_by == "distance":
                point_with_lower_dist = np.where(
                    dist_to_others < self.max_distance
                )[0]
                if len(point_with_lower_dist) > self.min_points_distance:
                    point_clouds.append(point_with_lower_dist)
            else:
                raise NotImplementedError(
                    "sample mode must be one of 'neighbors', 'distance'!"
                )
        return point_clouds

    def fit(self, x_train, y_train, coords_train):
        coords_train = np.array(coords_train)
        assert len(coords_train.shape) == 2 and coords_train.shape[
            1] == 2, "coords test must have len 2 in dimension 1"

        # sample core points
        self.estimator_core_points = self._sample_core_points(coords_train)
        point_clouds = self._sample_point_clouds(coords_train)
        if len(point_clouds) < self.nr_estimators:
            warnings.warn(
                f"Some point clouds had less than {self.min_points_distance} points and are therefore ignored.\
                     Consider increasing the parameter 'max_distance' to include more points"
            )
            # correct number of estimators
            self.nr_estimators = len(point_clouds)
            self.estimators = self.estimators[:self.nr_estimators]
        # correct estimator core points: Use center of gravity of each point clouds
        self.estimator_core_points = np.array(
            [
                np.mean(coords_train[cloud_inds], axis=0)
                for cloud_inds in point_clouds
            ]
        )
        # fit each point cloud to an estimator
        for i, sample_inds in enumerate(point_clouds):
            x_train_subset = x_train[sample_inds]
            y_train_subset = y_train[sample_inds]
            self.estimators[i].fit(x_train_subset, y_train_subset)

    def predict(self, x_test, coords_test=None, weighted=True):
        assert (
            coords_test is not None
        ) or weighted == False, "If weighted=True, then coords_test is required."
        y_pred = np.zeros((len(x_test), self.nr_estimators))
        for i, estimator in enumerate(self.estimators):
            y_pred[:, i] = estimator.predict(x_test)
        # If no spatial weighting: Simply return average of estimators
        if not weighted:
            return np.mean(y_pred, axis=1)
        coords_test = np.array(coords_test)
        assert len(coords_test.shape) == 2 and coords_test.shape[
            1] == 2, "coords test must have len 2 in dimension 1"
        dist_to_core_points = np.array(
            [
                np.sqrt(np.sum((coords_test - core_point)**2, axis=1))
                for core_point in self.estimator_core_points
            ]
        ).swapaxes(1, 0)

        # turn into probabilies
        if np.any(dist_to_core_points == 0):
            weights = np.array(
                [0 if dist != 0 else 1 for dist in dist_to_core_points]
            )
        else:
            weights = 1 / dist_to_core_points
            weights = weights / np.expand_dims(np.sum(weights, axis=1), 1)
        return np.sum(y_pred * weights, axis=1)

    def _sample_by_distance_old(
        coords_train, nr_clouds=20, radius=150000, min_points=400
    ):
        """Coords: coordinate array of shape N x 2"""
        # make distance matrix n x n
        dist = np.zeros((len(coords_train), len(coords_train)))
        for i, coord1 in enumerate(coords_train):
            for j, coord2 in enumerate(coords_train[i:]):
                dist[i, j + i] = np.linalg.norm(coord1 - coord2)
        # mirror distance matrix
        dist = dist + dist.T
        # make point clouds
        point_clouds = []
        for core_ind in np.random.permutation(len(dist)):
            dist_to_others = dist[core_ind]
            inds = np.where(dist_to_others < radius)[0]
            if len(inds) > min_points:
                point_clouds.append(inds)
            #             print("Cloud for core ind", core_ind, "has members", len(inds))
            if len(point_clouds) > nr_clouds:
                break
        return point_clouds
