r"""
Sphinx configuration
"""

import inspect
import sccross
import sphinx_autodoc_typehints
project = 'scCross'
version = sccross.__version__
release = sccross.__version__
author = "Xiu-hui Yang"
copyright = "Hao Wu Lab and Jun Ding Lab, 2023"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'nbsphinx'
]

templates_path = ['_templates']
html_static_path = ['_static']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_show_sourcelink = True
set_type_checking_flag = True
typehints_fully_qualified = True
napoleon_use_rtype = False
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
autodoc_default_options = {
    'autosummary': True
}

html_favicon = '_static/icon.ico'
html_theme = 'sphinx_rtd_theme'

intersphinx_mapping = dict(
    python=('https://docs.python.org/3/', None),
    numpy=('https://numpy.org/doc/stable/', None),
    scipy=('https://docs.scipy.org/doc/scipy/reference/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    sklearn=('https://scikit-learn.org/stable/', None),
    matplotlib=('https://matplotlib.org/stable/', None),
    seaborn=('https://seaborn.pydata.org/', None),
    networkx=('https://networkx.org/documentation/stable/', None),
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    scanpy=('https://scanpy.readthedocs.io/en/stable/', None),
    torch=('https://pytorch.org/docs/stable/', None),
    ignite=('https://pytorch.org/ignite/', None),
    plotly=('https://plotly.com/python-api-reference/', None)
)

qualname_overrides = {
    'anndata._core.anndata.AnnData': 'anndata.AnnData',
    'matplotlib.axes._axes.Axes': 'matplotlib.axes.Axes',
    'networkx.classes.graph.Graph': 'networkx.Graph',
    'networkx.classes.digraph.DiGraph': 'networkx.DiGraph',
    'networkx.classes.multigraph.MultiGraph': 'networkx.MultiGraph',
    'networkx.classes.multidigraph.MultiDiGraph': 'networkx.MultiDiGraph',
    'numpy.random.mtrand.RandomState': 'numpy.random.RandomState',
    'pandas.core.frame.DataFrame': 'pandas.DataFrame',
    'scipy.sparse.base.spmatrix': 'scipy.sparse.spmatrix',
    'seaborn.axisgrid.JointGrid': 'seaborn.JointGrid',
    'torch.device': 'torch.torch.device',
    'torch.nn.modules.module.Module': 'torch.nn.Module'
}

fa_orig = sphinx_autodoc_typehints.format_annotation
def format_annotation(annotation, fully_qualified=True):  # pylint: disable=unused-argument
    r"""
    Adapted from https://github.com/agronholm/sphinx-autodoc-typehints/issues/38#issuecomment-448517805
    """
    if inspect.isclass(annotation):
        full_name = f'{annotation.__module__}.{annotation.__qualname__}'
        override = qualname_overrides.get(full_name)
        if override is not None:
            return f':py:class:`~{override}`'
    return fa_orig(annotation)
sphinx_autodoc_typehints.format_annotation = format_annotation
