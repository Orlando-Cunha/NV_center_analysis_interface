from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class T1Result:
    x: np.ndarray
    y: np.ndarray
    error: np.ndarray

@dataclass
class RabiResult:
    x: np.ndarray
    y: np.ndarray
    error: np.ndarray

@dataclass
class RamseyResult:
    x: np.ndarray
    y: np.ndarray

@dataclass
class ODMRResult:
    x: np.ndarray
    y: np.ndarray
    y2: Optional[np.ndarray] = None  # For compatibility with interface, can be same as y
    image: Optional[np.ndarray] = None
    x_min: Optional[int] = None
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    filtered_y: Optional[np.ndarray] = None 