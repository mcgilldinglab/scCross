
try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import data,  models, num, plot, metrics
from .utils import config, log


name = "sccross"
__version__ = version(name)
