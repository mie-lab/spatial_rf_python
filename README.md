# Benchmarking regression models under spatial heterogeneity 

This repository accompanies our GIScience publication "Benchmarking regression models under spatial heterogeneity" (see reference below). In the code base, we provide 1) the script for reproducing our experiments on synthetic data, 2) the script for reproducing our benchmarking experiments on several real datasets and 3) an open-source Python implementation of spatial Random Forests. Each part is described in the following.

#### Installation

The required packages and our sprf package can be installed via pip in editable mode in a virtual environment with the following commands:
```
git clone https://github.com/mie-lab/spatial_rf_python.git
cd spatial_rf_python
python -m venv env
source env/bin/activate
pip install -e .
````

### 1) Experiments on synthetic datasets

To reproduce our analysis on synthetic data, run:
```
python scripts/synthetic_tests.py
```
All results will be saved in a single csv file named `synthetic_data_results.csv`.

### 2) Benchmarking on real datasets

We use five public data sets to validate our results and to benchmark different algorithms. The datasets are provided as csv fils in the [data](data) folder. They include
* A [plants](https://github.com/BlasBenito/spatialRF/blob/main/data/plant_richness_df.rda) dataset
* A [deforestation](https://github.com/FSantosCodes/GWRFC/tree/master/data) dataset
* A [mortality rate](https://www.dropbox.com/s/lrz6og0ld2m64df/Data_GWR.7z?dl=0) dataset from [here](https://zia207.github.io/geospatial-r-github.io/geographically-wighted-random-forest.html)

Please cite these sources if reusing their data.

Our code for benchmarking is provided as a [notebook](benchmarks.ipynb) and as a [script](scripts/benchmarks.py). To reproduce our experiments from the paper, run
```
python scripts/benchmarks.py
```
The results will be saved as csv files in a folder named `outputs`.

### 3) Spatial Random Forest implementation in Python

This repository further provides Python implementations of Spatial Random Forests. Different approaches have been proposed in the literature, including:
* Simply using coordinates as covariates
* *Geographical Random Forests*: Training one random forest per sample, using only the sample's neighbors as training data (similar to GWR)
* *Random Forest Spatial Interpolation*: Including the actual observations at the nearest locations of the prediction location as covariates

Here, we also implement a variant of Geographical Random Forests: Instead of training one Random Forest per sample, we train a fixed number of random forests on spatially distinct set of points. The prediction is then a weighted average of the tree-wise predictions, weighted by the distance of the test sample from the centers of each tree (see figure).

![sprf](assets/sprf.png){width=100}

#### Usage

The usage is analogous to other scikit-learn models, except that the coordinates must also be given as input:
```
from sprf import SpatialRandomForest
spatial_rf = SpatialRandomForest()
spatial_rf.fit(train_x, train_y, train_coords)
test_pred = spatial_rf.predict(test_x, test_coords)
```

The  `SpatialRandomForest` refers to the one explained above, with spatial decision trees. We also provide an implementation of the Geographical Random Forest by Georganos et al, which can be used in the same way:

```
from sprf import GeographicalRandomForest
geo_rf = GeographicalRandomForest()
geo_rf.fit(train_x, train_y, train_coords)
test_pred = geo_rf.predict(test_x, test_coords)
```



### Citation

If you use our work, please cite our paper with the following bibtex entry:

```bib
@inproceedings{wiedemann2023benchmarking,
  title={Benchmarking regression models under spatial heterogeneity},
  author={Wiedemann, Nina and Martin, Henry and Westerholt, René},
  booktitle={12th International Conference on Geographic Information Science (GIScience 2023)},
  year={2023},
}
```
