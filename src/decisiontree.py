import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import numpy as np
import sklearn.base as base
from rich import print as rprint
from rich.tree import Tree


def entropy_func(c: int, n: int) -> float:
    return -(c * 1.0 / n) * math.log(c * 1.0 / n, 2)


def entropy_cal(c1: int, c2: int):
    """
    Returns entropy of a group of data (i.e. one side of the split)
    c1: count of one class
    c2: count of another class
    """
    if (
        c1 == 0 or c2 == 0
    ):  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1 + c2) + entropy_func(c2, c1 + c2)


def entropy_of_one_division(division: Iterable) -> Tuple[float, int]:
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0  # entropy
    n = len(division)
    classes = set(division)
    for c in classes:  # for each class, get entropy
        n_c = sum(division == c)
        # weighted avg
        s_c = (
            n_c * 1.0 / n * entropy_cal(sum(division == c), sum(division != c))
        )
        s += s_c
    return s, n


def get_entropy(y_predict: Iterable, y_real: Iterable) -> float:
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """
    n = len(y_real)
    if len(y_predict) != n:
        raise ValueError("They have to be the same length")

    # left hand side entropy
    s_true, n_true = entropy_of_one_division(y_real[y_predict])
    # right hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])
    # overall entropy, again weighted average
    s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false
    return s


@dataclass
class Node:
    index_col: int  # index of the column to use
    cutoff: float  # threshold for decision
    measure_name: str
    measure: float  # optimization value gini etc
    value: float  # value to use for predictions
    n_obs: int  # number of observations in node
    left = None  # will be of type Node
    right = None  # will be of type Node
    reason: str = None  # place for some comment


class DecisionTreeTemplate(base.BaseEstimator):
    """
    Mostly based on https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea
    """

    def __init__(self, max_depth: int) -> None:
        self.depth_ = 0
        self.max_depth = max_depth
        self.measure_name_ = None  # entropy, variance
        self.measure_func_ = None  # get_entropy, get_variance

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        self._fit(X, y)
        return self

    def all_same(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        par_node: dict = None,
        depth: int = 0,
    ):
        """
        x: Feature set
        y: target variable
        par_node: will be the tree generated for this x and y.
        depth: the depth of the current layer
        """
        if par_node is None and getattr(
            self, "tree", None
        ):  # base case 1: tree stops at previous level
            return Node(
                None, None, None, None, None, None, "previous level"
            )  # {"reason": "previous level"}
        elif len(y) == 0:  # base case 2: no data in this group
            return Node(
                None, None, None, None, None, None, "no data in group"
            )  # {"reason": "no data in group"}
        elif self.all_same(y):  # base case 3: all y is the same in this group
            return Node(
                None, None, None, None, y[0], len(y), "homogenous group"
            )  # {'val':y[0], "reason": "homogenous group"}
        elif depth >= self.max_depth:  # base case 4: max depth reached
            return Node(
                None, None, None, None, np.mean(y), len(y), "max depth reached"
            )  # {"val": np.mean(y), "reason": "max depth reached"}
        else:  # Recursively generate tree!
            # find one split given an information gain
            col, cutoff, measure = self.find_best_split_of_all(X, y)
            is_left = X[:, col] < cutoff
            is_right = ~is_left
            # left hand side data
            y_left = y[is_left]
            X_left = X[is_left]
            # right hand side data
            y_right = y[is_right]
            X_right = X[is_right]
            # logging
            par_node = Node(
                col, cutoff, self.measure_name_, measure, np.mean(y), len(y)
            )
            # generate tree for the left hand side data
            par_node.left = self._fit(X_left, y_left, {}, depth + 1)
            # right hand side tree
            par_node.right = self._fit(X_right, y_right, {}, depth + 1)
            self.depth_ += 1  # increase the depth since we call fit
            self.tree_ = par_node
            return par_node

    def find_best_split(
        self, col: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Union[int, float]]:
        """
        col: the column we split on
        y: target var
        """
        min_val = None

        for value in set(col):  # iterating through each value in the column
            y_predict = col < value  # separate y into 2 groups
            val = self.measure_func_(y_predict, y)  # get entropy of this split
            if min_val is None or (
                val <= min_val
            ):  # check if it's the best one so far
                min_val = val
                cutoff = value
        return min_val, cutoff

    def find_best_split_of_all(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[int, Union[int, float], float]:
        """
        Find the best split from all features
        returns: the column to split on, the cutoff value, and the actual measure
        """
        col = None
        min_measure = None
        cutoff = None
        for i, x in enumerate(X.T):  # iterating through each feature
            measure, cur_cutoff = self.find_best_split(
                x, y
            )  # find the best split of that feature
            if measure == 0:  # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, measure
            elif min_measure is None or (
                measure < min_measure
            ):  # check if it's best so far
                min_measure = measure
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_measure

    def all_same(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        results = np.zeros(X.shape[0])
        for i, c in enumerate(X):  # for each row in test data
            results[i] = self._get_leaf_node(c).value
        return results

    def _get_leaf_node(self, row: np.ndarray) -> Node:
        cur_layer = self.tree_  # get the tree we built in training
        is_leaf = lambda node: (node.left is None) and (node.right is None)
        has_val = lambda node: node is not None and node.value is not None
        while not is_leaf(cur_layer):  # if not leaf node
            # get the direction
            go_left = row[cur_layer.index_col] < cur_layer.cutoff
            if go_left:
                if has_val(cur_layer.left):
                    cur_layer = cur_layer.left
                else:
                    break
            else:
                if has_val(cur_layer.right):
                    cur_layer = cur_layer.right
                else:
                    break
        # if leaf node, return node
        return cur_layer


class DecisionTreeClassifier(DecisionTreeTemplate, base.ClassifierMixin):
    def __init__(self, max_depth: int) -> None:
        super().__init__(max_depth=max_depth)
        self.measure_name_ = "entropy"
        self.measure_func_ = get_entropy


def get_variance(y_predict: Iterable, y_real: Iterable) -> float:
    """
    Returns variance of a split
    """
    n = len(y_real)
    if len(y_predict) != n:
        raise ValueError("They have to be the same length")

    # left hand side variance
    var_true, n_true = (
        np.var(y_real[y_predict]),
        y_predict.sum(),
    )  # entropy_of_one_division(y_real[y_predict])
    # right hand side variance
    var_false, n_false = np.var(y_real[~y_predict]), (~y_predict).sum()
    # overall variance, again weighted average
    var = n_true / n * var_true + n_false / n * var_false
    return var


class DecisionTreeRegressor(DecisionTreeTemplate, base.RegressorMixin):
    def __init__(self, max_depth: int) -> None:
        super().__init__(max_depth=max_depth)
        self.measure_name_ = "variance"
        self.measure_func_ = get_variance


def walk_tree(
    decision_tree: Node, tree: Tree, parent: Node = None, is_left: bool = None
):

    arrow = (
        ""
        if parent is None
        else f"[magenta](< {parent.cutoff:.3f})[/magenta]"
        if is_left
        else f"[magenta](>= {parent.cutoff:.3f})[/magenta]"
    )
    is_leaf = (decision_tree.left is None) and (decision_tree.right is None)
    if is_leaf:  # base cases
        branch = tree.add(
            f"{arrow} 🍁 # obs: [cyan]{decision_tree.n_obs}[/cyan], value: [green]{decision_tree.value:.3f}[/green], leaf reason '{decision_tree.reason}'"
        )
        return None
    else:
        branch = tree.add(
            f"{arrow} col idx: {decision_tree.index_col}, threshold: [magenta]{decision_tree.cutoff:.3f}[/magenta]"
        )

        if decision_tree.left is not None:  # go left
            walk_tree(decision_tree.left, branch, decision_tree, True)

        if decision_tree.right is not None:  # go right
            walk_tree(decision_tree.right, branch, decision_tree, False)


def show_tree(decision_tree: DecisionTreeTemplate):
    tree = Tree(f"Represenation of 🌲 ({decision_tree})")
    walk_tree(decision_tree.tree_, tree)
    rprint(tree)
