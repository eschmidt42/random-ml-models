import math
from typing import List

import numpy as np
import sklearn.base as base

import src.decisiontree as dt


class GradientBoostedTreeTemplate(base.BaseEstimator):
    def __init__(
        self, n_trees: int, max_depth: int, rng: np.random.RandomState
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.rng = rng
        self.trees_: List[dt.DecisionTreeTemplate] = None
        self.features_: List[np.ndarray] = None

    def _fit_tree(
        self, X: np.ndarray, y: np.ndarray
    ) -> dt.DecisionTreeRegressor:
        tree = dt.DecisionTreeRegressor(self.max_depth)
        tree.fit(X, y)
        return tree

    def _get_start_estimate_(self, y: np.ndarray) -> float:
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError()


class GradientBoostedTreeRegressor(
    GradientBoostedTreeTemplate, base.RegressorMixin
):
    """
    Friedman 2001, Greedy Function Approximation: A Gradient Boosting Machine
    Algorithm 2 (LS_Boost)

    y = our continuous target
    M = number of boosts

    start_estimate = mean(y)
    for m = 1 to M do:
        dy = y - prev_estimate
        new_rho, new_estimator = arg min(rho, estimator) mse(dy, rho*estimator(x))
        new_estimate = prev_estimate + new_rho * new_estimator(x)
        prev_estimate = new_estimate
    """

    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        factor: float,
        rng: np.random.RandomState,
    ) -> None:
        super().__init__(n_trees, max_depth, rng)
        self.factor = factor

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.trees_ = []
        self.features_ = []

        self.start_estimate_ = np.mean(y)

        _y = y - self.start_estimate_

        for _ in range(self.n_trees):

            new_tree = self._fit_tree(X, _y)
            self.trees_.append(new_tree)

            dy = self.factor * new_tree.predict(X)
            _y = _y - dy

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = np.ones(X.shape[0]) * self.start_estimate_

        for tree in self.trees_:  # loop boosts
            dy = self.factor * tree.predict(X)
            y += dy

        return y


def bool_to_float(x: bool) -> float:
    if x == True:
        return 1.0
    elif x == False:
        return -1.0
    else:
        raise ValueError(f"{x=}, expected bool")


class GradientBoostedTreeClassifier(
    GradientBoostedTreeTemplate, base.ClassifierMixin
):
    """
    Friedman 2001, Greedy Function Approximation: A Gradient Boosting Machine
    Algorithm 5 (LK_TreeBoost)

    y = our binary target (assumed in paper as -1 or 1)
    M = number of boosts
    loss = log(1+ exp(-2*y*estimate))
    estimate = 1/2 * log (P(y==1)/P(y==-1)) # log odds

    training
    --------

    start_estimate = 1/2 log (P(y==1) / P(y==-1))
    for m = 1 to M do:
        # d loss / d estimate
        dy = 2*y / (1 + exp(2*y*prev_estimate))
        new_estimator = estimator(x,dy)
        # new_estimator leaf sensitive update
        gamma[m,leaf j] = (sum_i dy) / (sum_i abs(dy)*(2-abs(dy))) # looks weird, see comment below*
        new_estimate = prev_estimate + (sum_j gamma[m,leaf j] if x in leaf j)
        prev_estimate = new_estimate

    computing the probability using new_estimate:
    P(y==1 given trees) = 1 / (1 + exp(-2*new_estimate))

    *gamma[m,leaf j] comment:
    Derivation of new_estimate takes three steps:
    1) loss minimal for estimate:
        -> rho_m = arg min_rho (sum_i log(1+exp(-2y*(prev_estimate + rho*h(x)))) )
    2) translation for tree model: rho*h(x) is for a tree just some leaf value gamma
        -> arg min_gamma instead of arg min_rho
        -> gamma[m,leaf j] = arg min_gamma (sum_i log(1+exp(-2y*(prev_estimate + gamma))) )
    3) estimation of gamma in the previous equation, using Newton-Raphson:
        -> gamma_jm = (sum_i dy) / (sum_i abs(dy)*(2-abs(dy)))
        with d loss / d estimate = dy = 2*y / (1 + exp(2*y*prev_estimate))

    inference
    ---------

    y = start_estimate
    for m = 1 to M do:
        # determine leaf for each x
        leaf j = estimator[m](x)
        # for each x retrieve gamma
        y += gamma[m,leaf j]

    y = 1 / (1 + exp(-2*y)) # converting to probability
    """

    def __init__(
        self, n_trees: int, max_depth: int, rng: np.random.RandomState
    ) -> None:
        super().__init__(n_trees, max_depth, rng)

    def _bool_to_float(self, y: np.ndarray) -> np.ndarray:
        f = np.vectorize(bool_to_float)
        return f(y)

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.trees_ = []
        self.features_ = []
        self.gammas_ = []

        y = self._bool_to_float(y)
        ym = np.mean(y)
        self.start_estimate_ = 0.5 * math.log((1 + ym) / (1 - ym))

        _y = np.ones_like(y) * self.start_estimate_

        for _ in range(self.n_trees):

            dy = 2 * y / (1 + np.exp(2 * y * _y))

            new_tree = self._fit_tree(X, dy)
            self.trees_.append(new_tree)

            # collect node ids for each x
            nodes = [new_tree._get_leaf_node(x) for x in X]
            ids = np.array([n.id for n in nodes])

            # log node ids and corresponding update gamma
            gammas = {}
            for _id in ids:
                # dy only for current id
                dy_for_id = np.where(ids == _id, dy, 0)
                # gamma[m,leaf j] = (sum_i dy) / (sum_i abs(dy)*(2-abs(dy)))
                dy_id_filtered_abs = np.abs(dy_for_id)
                gamma = (
                    dy_for_id.sum()
                    / (dy_id_filtered_abs * (2 - dy_id_filtered_abs)).sum()
                )
                # store
                gammas[_id] = gamma
            # store
            self.gammas_.append(gammas)

            # update _y
            dy = np.array([gammas[_id] for _id in ids])
            _y = _y + dy

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = np.ones(X.shape[0]) * self.start_estimate_

        for m, tree in enumerate(self.trees_):  # loop boosts
            nodes = [tree._get_leaf_node(c) for c in X]
            ids = np.array([n.id for n in nodes])
            dy = np.array([self.gammas_[m][_id] for _id in ids])
            y += dy

        y = 1 / (1 + np.exp(-2.0 * y))

        return y
