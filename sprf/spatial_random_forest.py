import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor


class SpatialRandomForest:
    """
    Spatial Random Forest implementation, following the sklearn style

    Parameters
    ----------
    n_estimators : int, optional
        Number of base estimators (decision trees), by default 20
    sample_mode : str, optional {cluster, random}
        Trees are rooted either in the centers of clusters of the dataset, or in
        random locations, by default "cluster"
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
    min_points_distance : int, optional
        Minimum points for fitting a decision tree, i.e. if the distance is
        set too low, decision trees would be fit on an insufficient number of
        points. Only relevant if sample_by=distance, by default 100
    **kwargs: dict
        Any arguments that are passed to the sklearn DecisionTreeRegressor

    Notes
    ----------
    - Only regression is implemented so far.
    - In contrast to other spatial RF papers, we do not build on tree per
     sample, but rather a fixed set of <n_estimators> trees on spatial subsets
     of the data. 
    - The spatial subsets are either coosen by a fixed number of neighbors or
     by a radius (spatial distance)
    - Projected coordinates are assumed!

    Example
    ----------
    sp = SpatialRandomForest(max_depth=20, neighbors=50)
    sp.fit(train_x, train_y, train_coords)
    pred_y = sp.predict(test_x, test_coords)
    """

    def __init__(
        self,
        n_estimators: int = 20,
        sample_mode: str = "cluster",
        sample_by: str = "neighbors",
        neighbors: int = 500,
        max_distance: float = 150000,
        min_points_distance: int = 100,
        **decision_tree_args,
    ):
        self.estimators = [
            DecisionTreeRegressor(**decision_tree_args)
            for _ in range(n_estimators)
        ]
        if sample_by == "distance" and max_distance == 150000:
            warnings.warn(
                "It seems that you have selected the 'distance'-sampling mode,\
                     but the parameter max_distance is still the default. Make\
                     sure to adapt the max_distance parameter to your dataset."
            )
        self.n_estimators = n_estimators
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
        """
        Sample indices of points that form the centers of each spatial tree.
        coords: 2D Array of shape (N, 2) where N is the number of samples
        Returns: 2D Array of shape (N, 2) which is a subset / another set of
            coordinates
        """
        if self.sample_mode == "cluster":
            # cluster coordinates with kmeans use centers as core points
            kmeans = KMeans(self.n_estimators)
            kmeans.fit(coords)
            core_points = kmeans.cluster_centers_
        # TODO: elif sample_mode == "grid":
        elif self.sample_mode == "random":
            # select random coordinates from the train data as core points
            core_points = coords[np.random.permutation(len(coords)
                                                       )[:self.n_estimators]]
        else:
            raise NotImplementedError(
                "sample mode must be one of cluster, random"
            )
        return core_points

    def _sample_point_clouds(self, coords):
        """
        Assign samples to their spatial decision tree.
        coords: 2D Array of shape (N, 2) where N is the number of samples
        Returns: List of lists with indices of samples belonging to each tree
        """
        point_clouds = []
        for core_point in self.estimator_core_points:
            # Compute distance of the core point to all coordinates
            dist_to_others = np.sqrt(np.sum((coords - core_point)**2, axis=1))
            if self.sample_by == "neighbors":
                # add fixed number of closest samples
                point_clouds.append(
                    np.argsort(dist_to_others)[:self.neighbors]
                )
            elif self.sample_by == "distance":
                # filter by distance
                point_with_lower_dist = np.where(
                    dist_to_others < self.max_distance
                )[0]
                # only use point clouds that are large enough! --> cannot fit a
                # decision tree on 5 points
                if len(point_with_lower_dist) > self.min_points_distance:
                    point_clouds.append(point_with_lower_dist)
            else:
                raise NotImplementedError(
                    "sample mode must be one of 'neighbors', 'distance'!"
                )
        return point_clouds

    def fit(self, x_train, y_train, coords_train):
        """
        Fit spatial random forest to a dataset.

        Parameters
        ----------
        x_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``.
        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers in regression).
        coords_train: array-like of shape (n_samples, 2) with spatial
            coordinates of each sample. Geographic coordinates are assumed to be
            projected!
        """
        # convert to arrays
        x_train, y_train, coords_train = (
            np.array(x_train),
            np.array(y_train),
            np.array(coords_train),
        )
        assert (
            len(coords_train.shape) == 2 and coords_train.shape[1] == 2
        ), "coords test must have len 2 in dimension 1"

        # sample core points
        self.estimator_core_points = self._sample_core_points(coords_train)
        # assign samples to their core points
        # (one sample can be in several point clouds!)
        point_clouds = self._sample_point_clouds(coords_train)
        if len(point_clouds) < self.n_estimators:
            warnings.warn(
                f"Some point clouds had less than {self.min_points_distance}\
                     points and are therefore ignored.\
                     Consider increasing the parameter 'max_distance' to\
           include more points (recommended), or decrease 'min_points_distance'"
            )
            # correct number of estimators
            self.n_estimators = len(point_clouds)
            self.estimators = self.estimators[:self.n_estimators]
        # correct core points: Use center of gravity of each point clouds
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
        """
        Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        x_test : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. 
        coords_test: array-like of shape (n_samples, 2), optional
            Coordinates are only required if weighted=True, i.e. if the tree-
            wise outputs should be weighted and combined by their distance
        weighted: bool, optional
            Whether the tree-wise predictions should be aggregated based on
            their spatial distance to the test sample (similar to inverse
            distance weighting).

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        # convert to arrays
        x_test = np.array(x_test)
        if coords_test is not None:
            coords_test = np.array(coords_test)
        assert (coords_test is not None) or weighted == False, (
            "If weighted=True, then coords_test is required."
        )
        # predict output with each base estimator
        y_pred = np.zeros((len(x_test), self.n_estimators))
        for i, estimator in enumerate(self.estimators):
            y_pred[:, i] = estimator.predict(x_test)
        # If no spatial weighting: Simply return average of estimators
        if not weighted:
            return np.mean(y_pred, axis=1)
        # if weighted: check that test coords are alright
        coords_test = np.array(coords_test)
        assert (
            len(coords_test.shape) == 2 and coords_test.shape[1] == 2
        ), "coords test must have len 2 in dimension 1"
        # compute distance of test samples to all core points
        dist_to_core_points = np.array(
            [
                np.sqrt(np.sum((coords_test - core_point)**2, axis=1))
                for core_point in self.estimator_core_points
            ]
        ).swapaxes(1, 0)

        # turn into probabilies
        if np.any(dist_to_core_points == 0):
            # special if test sample is exactly equal to one of the core points
            weights = np.array(
                [0 if dist != 0 else 1 for dist in dist_to_core_points]
            )
        else:
            # normal situation: weight dependent on spatial distance
            weights = 1 / dist_to_core_points
            weights = weights / np.expand_dims(np.sum(weights, axis=1), 1)

        # prediction is weighted sum
        y_pred = np.sum(y_pred * weights, axis=1)
        return y_pred

    def _sample_by_distance_old(
        coords_train, nr_clouds=20, radius=150000, min_points=400
    ):
        """Deprecated"""
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
