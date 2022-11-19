import numpy as np
from sklearn.metrics import r2_score


def tune_neighbors(
    model, x_train, y_train, coords_train, nr_check=10, eval_criterium=r2_score,
):
    x_train, y_train, coords_train = (
        np.array(x_train),
        np.array(y_train),
        np.array(coords_train),
    )
    max_neighbors = len(x_train)
    # split in train and val
    cutoff = int(len(x_train) * 0.9)
    rand_inds = np.random.permutation(max_neighbors)
    train_i, val_i = rand_inds[:cutoff], rand_inds[cutoff:]
    x_val = x_train[val_i]
    x_train = x_train[train_i]
    y_val = y_train[val_i]
    y_train = y_train[train_i]
    coords_val = coords_train[val_i]
    coords_train = coords_train[train_i]

    steps_to_check = np.linspace(0, max_neighbors, nr_check + 2).astype(int)
    best_neighbors = model.neighbors
    best_performance = -np.inf
    for neighbors in steps_to_check[1:-1]:
        model.neighbors = neighbors
        model.fit(x_train, y_train, coords_train)
        y_pred = model.predict(x_val, coords_val)
        performance = eval_criterium(y_pred, y_val)
        if performance > best_performance:
            best_neighbors = neighbors
            best_performance = performance

    # print("Found best bandwidth (neighbors) at ", best_neighbors)
    model.neighbors = best_neighbors
    return best_neighbors
