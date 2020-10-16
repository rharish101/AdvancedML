"""Contains custom Python typings specific to this project."""
from abc import ABC
from typing import Any, Tuple

from nptyping import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

# fmt: off

StructuredArray = NDArray[(Any,), Any]  # type: ignore
CSVData = NDArray[(Any, Any,), float]  # type: ignore
CSVHeader = Tuple[str, ...]  # type: ignore
Array2D = NDArray[(Any, Any,), Any]  # type: ignore

# fmt: on


class BaseRegressor(ABC, BaseEstimator, RegressorMixin):
    """Convenient type for a base regressor.

    This can be used as a parent class for creating custom regressors.
    """
