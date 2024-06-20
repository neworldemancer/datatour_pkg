"""Seeing is important. `datatour` - allows you to see your data in it's native dimension.
Currently implemented as a `plotly` scatter plot projected from it's original dimension in the 2D on the screen with timeline animation inspired by GrandTour and common sense.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("datatour")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Mykhailo Vladymyrov"
__email__ = "neworldemancer@gmail.com"
