
"""
    scCross is a dDeep Learning-Based Model for integration, cross-dataset cross-modality generation and matched muti-omics simulation of single-cell multi-omics data. Our model can also maintain in-silico perturbations in cross-modality generation and can use in-silico perturbations to find key genes.


"""
try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import data,  models, metrics
from .utils import config, log


name = "sccross"
__version__ = version(name)
