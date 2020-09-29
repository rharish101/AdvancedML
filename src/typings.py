"""Contains custom Python typings specific to this project."""

from typing import Any, Tuple

from nptyping import NDArray

# fmt: off

CSVData = NDArray[(Any, Any,), float]  # type: ignore
CSVHeader = Tuple[str, ...]  # type: ignore
Array2D = NDArray[(Any, Any,), Any]  # type: ignore

# fmt: on
