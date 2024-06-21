"""
Visualization in its native dimension.

Seeing is important. `datatour` - allows you to see your data in its
native dimension. Currently implemented as a `plotly` scatter plot projected
from its original dimension in the 2D on the screen with timeline animation
inspired by GrandTour and common sense.
"""

from importlib.metadata import PackageNotFoundError, version

from . import datatour
from .datatour import DataTour

try:
    __version__ = version("datatour")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Mykhailo Vladymyrov"
__email__ = "neworldemancer@gmail.com"

__all__ = ["datatour", "DataTour"]
