"""NestedGridSearchCV example: fit a simple regression model."""
from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_breast_cancer
from KRC import KernelRidgeClassifier
from grid_search import NestedGridSearchCV

data = load_breast_cancer()
X = data['data']
y = data['target']

estimator = KernelRidgeClassifier()

param_grid = {'alpha': np.logspace(-3, 3, 10),
              'gamma': np.logspace(-3, 3, 10),
              'kernel': ['poly', 'rbf']}

nested_cv = NestedGridSearchCV(estimator, param_grid,
                               'accuracy', cv=5, inner_cv=3)
nested_cv.fit(X, y)

if MPI.COMM_WORLD.Get_rank() == 0:
    for i, scores in enumerate(nested_cv.grid_scores_):
        scores.to_csv('grid-scores-%d.csv' % (i + 1), index=False)

    print(nested_cv.best_params_)
