r"""
Integration models
"""

import os
from typing import Mapping


import numpy as np
from anndata import AnnData


from ..typehint import Kws
from ..utils import config, logged
from .base import Model
from .dx import integration_consistency
from .sccross import SCCROSSModel
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
