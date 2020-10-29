"""Contains custom Python typings specific to this project."""
from abc import ABC
from typing import Any, Tuple

from nptyping import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# fmt: off

StructuredArray = NDArray[(Any,), Any]  # type: ignore
CSVData = NDArray[(Any, Any,), float]  # type: ignore
CSVHeader = Tuple[str, ...]  # type: ignore
Array2D = NDArray[(Any, Any,), Any]  # type: ignore

# fmt: on


class BaseRegressor(ABC, RegressorMixin, BaseEstimator):
    """Convenient type for a base regressor.

    This can be used as a parent class for creating custom regressors.
    """


class BaseClassifier(ABC, ClassifierMixin, BaseEstimator):
    """Convenient type for a base classifier.

    This can be used as a parent class for creating custom classifiers.
    """
