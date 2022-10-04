import numpy as np
import pytest
import pandas as pd
from sprf import SpatialRandomForest


class TestSpatialRandomForest:
    """Test spatial random forest class"""

    x_train = np.random.rand(500, 10)
    y_train = np.random.rand(500)
    coords_train = np.random.rand(500, 2)
    x_test = np.random.rand(50, 10)
    y_test = np.random.rand(50)
    coords_test = np.random.rand(50, 2)

    def test_init_warning(self):
        with pytest.warns(UserWarning):
            sp = SpatialRandomForest(sample_by="distance")

    def test_fit(self):
        sp = SpatialRandomForest()
        sp.fit(self.x_train, self.y_train, self.coords_train)
        assert sp.n_estimators == 20

    def test_fit_equal(self):
        np.random.seed(42)
        sp1 = SpatialRandomForest()
        x_df = pd.DataFrame(self.x_train)
        sp1.fit(x_df, self.y_train, self.coords_train)
        y_pred_1 = sp1.predict(self.x_test, self.coords_test)

        np.random.seed(42)
        sp2 = SpatialRandomForest()
        sp2.fit(self.x_train, self.y_train, self.coords_train)
        y_pred_2 = sp2.predict(self.x_test, self.coords_test)
        assert np.all(y_pred_1 == y_pred_2)

    def test_predict_error(self):
        sp = SpatialRandomForest()
        sp.fit(self.x_train, self.y_train, self.coords_train)
        with pytest.raises(AssertionError):
            sp.predict(self.x_test)
