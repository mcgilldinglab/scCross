"""
    scCross is a dDeep Learning-Based Model for integration, cross-dataset cross-modality generation and matched muti-omics simulation of single-cell multi-omics data. Our model can also maintain in-silico perturbations in cross-modality generation and can use in-silico perturbations to find key genes.
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


import os
from typing import Mapping


import numpy as np
from anndata import AnnData

from ..utils import logged, Kws
from .utils import Model


from .sccross import (AUTO, SCCROSSModel,
                     configure_dataset)


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file
    """
    return Model.load(fname)


@logged
def fit_SCCROSS(
        adatas: Mapping[str, AnnData], model: type = SCCROSSModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None
) -> SCCROSSModel:

    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}

    fit_SCCROSS.logger.info("Pretraining SCCROSS model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")

    pretrain = model(adatas, **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, **pretrain_fit_kws)
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))


    fit_SCCROSS.logger.info("Fine-tuning SCCROSS model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune")

    finetune = model(adatas, **init_kws)
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile(**compile_kws)
    finetune.fit(adatas, **finetune_fit_kws)
    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    return finetune
