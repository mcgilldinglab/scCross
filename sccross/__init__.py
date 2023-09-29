
"""
    Part of the sccross' code is adapted from MIT licensed projects GLUE and SCDIFF2.
    Thanks for these projects:

    Author: Zhi-Jie Cao
    Project: GLUE
    Ref: Cao Z J, Gao G. Multi-omics single-cell data integration and regulatory inference with graph-linked embedding[J].
    Nature Biotechnology, 2022, 40(10): 1458-1466.

    Author: Jun Ding
    Project: SCDIFF2
    Ref: Ding, J., Aronow, B. J., Kaminski, N., Kitzmiller, J., Whitsett, J. A., & Bar-Joseph, Z.
    (2018). Reconstructing differentiation networks and their regulation from time series
    single-cell expression data. Genome research, 28(3), 383-395.

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
