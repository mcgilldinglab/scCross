

import copy
import os
import uuid
from itertools import chain
from math import ceil
from typing import Any, List, Mapping, Optional, Tuple, Union

import h5py
import ignite

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset


from ..typehint import AnyArray, RandomState
from ..utils import config, get_chained_attr, get_rs, logged
from . import sc
from .base import Model
from .data import  DataLoader, Dataset
from .cross import CROSS, CROSSTrainer
from .nn import freeze_running_stats, get_default_numpy_dtype

AUTO = -1  # Flag for using automatically determined hyperparameters
DATA_CONFIG = Mapping[str, Any]


#---------------------------------- Utilities ----------------------------------

def select_encoder(prob_model: str) -> type:
    r"""
    Select encoder architecture

    Parameters
    ----------
    prob_model
        Data probabilistic model

    Return
    ------
    encoder
        Encoder type
    """
    if prob_model in ("Normal", "ZIN", "ZILN"):
        return sc.VanillaDataEncoder
    if prob_model in ("NB", "ZINB"):
        return sc.NBDataEncoder
    raise ValueError("Invalid `prob_model`!")


def select_decoder(prob_model: str) -> type:
    r"""
    Select decoder architecture

    Parameters
    ----------
    prob_model
        Data probabilistic model

    Return
    ------
    decoder
        Decoder type
    """
    if prob_model == "Normal":
        return sc.NormalDataDecoder
    if prob_model == "ZIN":
        return sc.ZINDataDecoder
    if prob_model == "ZILN":
        return sc.ZILNDataDecoder
    if prob_model == "NB":
        return sc.NBDataDecoder
    if prob_model == "ZINB":
        return sc.ZINBDataDecoder
    raise ValueError("Invalid `prob_model`!")


@logged
class AnnDataset(Dataset):

    r"""
    Dataset for :class:`anndata.AnnData` objects with partial pairing support.

    Parameters
    ----------
    *adatas
        An arbitrary number of configured :class:`anndata.AnnData` objects
    data_configs
        Data configurations, one per dataset
    mode
        Data mode, must be one of ``{"train", "eval"}``
    getitem_size
        Unitary fetch size for each __getitem__ call
    """

    def __init__(
            self, adatas: List[AnnData], data_configs: List[DATA_CONFIG],
            mode: str = "train", getitem_size: int = 1
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        if mode not in ("train", "eval"):
            raise ValueError("Invalid `mode`!")
        self.mode = mode
        self.adatas = adatas
        self.data_configs = data_configs

    @property
    def adatas(self) -> List[AnnData]:
        r"""
        Internal :class:`AnnData` objects
        """
        return self._adatas

    @property
    def data_configs(self) -> List[DATA_CONFIG]:
        r"""
        Data configuration for each dataset
        """
        return self._data_configs

    @adatas.setter
    def adatas(self, adatas: List[AnnData]) -> None:
        self.sizes = [adata.shape[0] for adata in adatas]
        if min(self.sizes) == 0:
            raise ValueError("Empty dataset is not allowed!")
        self._adatas = adatas

    @data_configs.setter
    def data_configs(self, data_configs: List[DATA_CONFIG]) -> None:
        if len(data_configs) != len(self.adatas):
            raise ValueError(
                "Number of data configs must match "
                "the number of datasets!"
            )
        self.data_idx, self.extracted_data = self._extract_data(data_configs)
        self.view_idx = pd.concat(
            [data_idx.to_series() for data_idx in self.data_idx]
        ).drop_duplicates().to_numpy()
        self.size = self.view_idx.size
        self.shuffle_idx, self.shuffle_pmsk = self._get_idx_pmsk(self.view_idx)
        self._data_configs = data_configs

    def _get_idx_pmsk(
            self, view_idx: np.ndarray, random_fill: bool = False,
            random_state: RandomState = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rs = get_rs(random_state) if random_fill else None
        shuffle_idx, shuffle_pmsk = [], []
        for data_idx in self.data_idx:
            idx = data_idx.get_indexer(view_idx)
            pmsk = idx >= 0
            n_true = pmsk.sum()
            n_false = pmsk.size - n_true
            idx[~pmsk] = rs.choice(idx[pmsk], n_false, replace=True) \
                if random_fill else idx[pmsk][np.mod(np.arange(n_false), n_true)]
            shuffle_idx.append(idx)
            shuffle_pmsk.append(pmsk)
        return np.stack(shuffle_idx, axis=1), np.stack(shuffle_pmsk, axis=1)

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        s = slice(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        shuffle_idx = self.shuffle_idx[s].T
        shuffle_pmsk = self.shuffle_pmsk[s]
        items = [
            torch.as_tensor(self._index_array(data, idx))
            for extracted_data in self.extracted_data
            for idx, data in zip(shuffle_idx, extracted_data)
        ]
        items.append(torch.as_tensor(shuffle_pmsk))
        return items

    @staticmethod
    def _index_array(arr: AnyArray, idx: np.ndarray) -> np.ndarray:
        if isinstance(arr, (h5py.Dataset, SparseDataset)):
            rank = scipy.stats.rankdata(idx, method="dense") - 1
            sorted_idx = np.empty(rank.max() + 1, dtype=int)
            sorted_idx[rank] = idx
            arr = arr[sorted_idx][rank]  # Convert to sequantial access and back
        else:
            arr = arr[idx]
        return arr.toarray() if scipy.sparse.issparse(arr) else arr

    def _extract_data(self, data_configs: List[DATA_CONFIG]) -> Tuple[
            List[pd.Index], Tuple[
                List[AnyArray], List[AnyArray], List[AnyArray],
                List[AnyArray], List[AnyArray]
            ]
    ]:
        if self.mode == "eval":
            return self._extract_data_eval(data_configs)
        return self._extract_data_train(data_configs)  # self.mode == "train"

    def _extract_data_train(self, data_configs: List[DATA_CONFIG]) -> Tuple[
            List[pd.Index], Tuple[
                List[AnyArray], List[AnyArray], List[AnyArray],
                List[AnyArray], List[AnyArray]
            ]
    ]:
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xalt = [
            self._extract_xalt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xbch = [
            self._extract_xbch(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xlbl = [
            self._extract_xlbl(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xdwt = [
            self._extract_xdwt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        return xuid, (x, xalt, xbch, xlbl, xdwt)

    def _extract_data_eval(self, data_configs: List[DATA_CONFIG]) -> Tuple[
            List[pd.Index], Tuple[
                List[AnyArray], List[AnyArray], List[AnyArray],
                List[AnyArray], List[AnyArray]
            ]
    ]:
        default_dtype = get_default_numpy_dtype()
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xalt = [
            self._extract_xalt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config, xalt_ in zip(self.adatas, data_configs, xalt)
        ]
        xbch = xlbl = [
            np.empty((adata.shape[0], 0), dtype=int)
            for adata in self.adatas
        ]
        xdwt = [
            np.empty((adata.shape[0], 0), dtype=default_dtype)
            for adata in self.adatas
        ]
        return xuid, (x, xalt, xbch, xlbl, xdwt)

    def _extract_x(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )
            x = adata.layers[use_layer]
        else:
            x = adata.X
        if x.dtype.type is not default_dtype:
            if isinstance(x, (h5py.Dataset, SparseDataset)):
                raise RuntimeError(
                    f"User is responsible for ensuring a {default_dtype} dtype "
                    f"when using backed data!"
                )
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.tocsr()
        return x

    def _extract_xalt(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        use_rep = data_config["use_rep"]
        rep_dim = data_config["rep_dim"]
        if use_rep:
            if use_rep not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_rep}' "
                    f"cannot be found in input data!"
                )
            xalt = adata.obsm[use_rep].astype(default_dtype)

            return xalt
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xbch(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_batch = data_config["use_batch"]
        batches = data_config["batches"]
        if use_batch:
            if use_batch not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_batch}' "
                    f"cannot be found in input data!"
                )
            return batches.get_indexer(adata.obs[use_batch])
        return np.zeros(adata.shape[0], dtype=int)

    def _extract_xlbl(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_cell_type = data_config["use_cell_type"]
        cell_types = data_config["cell_types"]
        if use_cell_type:
            if use_cell_type not in adata.obs:
                raise ValueError(
                    f"Configured cell type '{use_cell_type}' "
                    f"cannot be found in input data!"
                )
            return cell_types.get_indexer(adata.obs[use_cell_type])
        return -np.ones(adata.shape[0], dtype=int)

    def _extract_xdwt(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        use_dsc_weight = data_config["use_dsc_weight"]
        if use_dsc_weight:
            if use_dsc_weight not in adata.obs:
                raise ValueError(
                    f"Configured discriminator sample weight '{use_dsc_weight}' "
                    f"cannot be found in input data!"
                )
            xdwt = adata.obs[use_dsc_weight].to_numpy().astype(default_dtype)
            xdwt /= xdwt.sum() / xdwt.size
        else:
            xdwt = np.ones(adata.shape[0], dtype=default_dtype)
        return xdwt

    def _extract_xuid(self, adata: AnnData, data_config: DATA_CONFIG) -> pd.Index:
        use_uid = data_config["use_uid"]
        if use_uid:
            if use_uid not in adata.obs:
                raise ValueError(
                    f"Configured cell unique ID '{use_uid}' "
                    f"cannot be found in input data!"
                )
            xuid = adata.obs[use_uid].to_numpy()
        else:  # NOTE: Assuming random UUIDs never collapse with anything
            self.logger.debug("Generating random xuid...")
            xuid = np.array([uuid.uuid4().hex for _ in range(adata.shape[0])])
        if len(set(xuid)) != xuid.size:
            raise ValueError("Non-unique cell ID!")
        return pd.Index(xuid)

    def propose_shuffle(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rs = get_rs(seed)
        view_idx = rs.permutation(self.view_idx)
        return self._get_idx_pmsk(view_idx, random_fill=True, random_state=rs)

    def accept_shuffle(self, shuffled: Tuple[np.ndarray, np.ndarray]) -> None:
        self.shuffle_idx, self.shuffle_pmsk = shuffled

    def random_split(
            self, fractions: List[float], random_state: RandomState = None
    ) -> List["AnnDataset"]:
        r"""
        Randomly split the dataset into multiple subdatasets according to
        given fractions.

        Parameters
        ----------
        fractions
            Fraction of each split
        random_state
            Random state

        Returns
        -------
        subdatasets
            A list of splitted subdatasets
        """
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        view_idx = rs.permutation(self.view_idx)
        split_pos = np.round(cum_frac * view_idx.size).astype(int)
        split_idx = np.split(view_idx, split_pos[:-1])
        subdatasets = []
        for idx in split_idx:
            sub = copy.copy(self)
            sub.view_idx = idx
            sub.size = idx.size
            sub.shuffle_idx, sub.shuffle_pmsk = sub._get_idx_pmsk(idx)
            subdatasets.append(sub)
        return subdatasets




class SCCROSS(CROSS):



    def __init__(
            self,
            x2u: Mapping[str, sc.DataEncoder],
            u2z: Mapping[str, sc.DataEncoder],
            z2u: Mapping[str, sc.DataDecoder],
            u2x: Mapping[str, sc.DataDecoder],
            du: sc.Discriminator, du_gen: Mapping[str,sc.Discriminator_gen],prior: sc.Prior,

            u2c: Optional[sc.Classifier] = None
    ) -> None:
        super().__init__( x2u,u2z,z2u, u2x,  du,du_gen, prior)
        self.u2c = u2c.to(self.device) if u2c else None






DataTensors = Tuple[
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]


@logged
class SCCROSSTrainer(CROSSTrainer):



    BURNIN_NOISE_EXAG: float = 1.5

    def __init__(
            self, net: SCCROSS, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            lam_sup: float = None, normalize_u: bool = None,
            domain_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph,
            lam_align=lam_align, domain_weight=domain_weight,
            optim=optim, lr=lr, **kwargs
        )
        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_sup = lam_sup
        self.normalize_u = normalize_u
        self.freeze_u = False
        if net.u2c:
            self.required_losses.append("sup_loss")

    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each domain,
        followed by alternative input arrays for each domain,
        in the same order as domain keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xalt, xbch, xlbl, xdwt = data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K]

        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xalt = {
            k: xalt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }

        return x, xalt, xbch, xlbl, xdwt, xflag

    def compute_losses1(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xalt, xbch, xlbl, xdwt, xflag = data

        u, u1, z, l, x_gen, x_gen_cat, x_gen_flag_cat, usamp1 = {}, {}, {}, {}, {}, {}, {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xalt[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}

        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        for k in net.keys:
            z[k] = net.u2z[k](u[k].mean)
            u1[k] = net.z2u[k](z[k].mean)

            usamp1[k] = u1[k].rsample()
            x_gen[k] = net.u2x[k](
                usamp1[k], xbch[k], l[k]
            )
            x_gen_cat[k] = torch.cat([x_gen[k].sample(), x[k]])
            x_gen_flag_cat[k] = torch.cat([xflag[k], torch.ones_like(xflag[k])])
        dsc_gen_loss = {
            k: (F.cross_entropy(net.du_gen[k](x_gen_cat[k]), x_gen_flag_cat[k], reduction="none")).sum()
            for k in net.keys
        }

        du_gen_loss_sum = sum(dsc_gen_loss[k] for k in net.keys)



        u_cat = torch.cat([z[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0],))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only:
            return {"dsc_loss": self.lam_align * (dsc_loss+du_gen_loss_sum )}

        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        x_nll = {
            k: -net.u2x[k](
                usamp[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        means = sum(u[k].mean for k in net.keys) / len(net.keys)
        scale = sum(u[k].stddev for k in net.keys) / len(net.keys)
        temp_D = D.Normal(means, scale)
        z_kl = {
            k: D.kl_divergence(
                z[k], temp_D
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.domain_weight[k] * x_elbo[k] for k in net.keys)
        z_kl_sum = sum(self.domain_weight[k] * z_kl[k] for k in net.keys)

        vae_loss = self.lam_data * x_elbo_sum + 0.1 * z_kl_sum

        gen_loss = vae_loss - self.lam_align * (dsc_loss+du_gen_loss_sum)

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,

        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses



    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xalt, xbch, xlbl, xdwt, xflag = data
        x_p = {}
        xalt1 = {}

        for k in net.keys:
            x_p[k] = xalt[k][:, -50:]
            xalt1[k] = xalt[k][:, :-50]


        u ,u1,z, l,x_gen,x_gen_cat,x_gen_flag_cat,usamp1 = {}, {}, {},{},{},{},{},{}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xalt1[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}


        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        for k in net.keys:
            z[k] = net.u2z[k](u[k].mean)
            u1[k] = net.z2u[k](z[k].mean)
            usamp1[k] = u1[k].rsample()
            x_gen[k] = net.u2x[k](
                usamp1[k], xbch[k], l[k]
            )
            x_gen_cat[k] = torch.cat([x_gen[k].sample(),x[k]])
            x_gen_flag_cat[k] = torch.cat([torch.zeros_like(xflag[k]),torch.ones_like(xflag[k])])
        dsc_gen_loss = {
            k : (F.cross_entropy(net.du_gen[k](x_gen_cat[k]), x_gen_flag_cat[k], reduction="none")).sum()
            for k in net.keys
        }

        zsamp = {k: z[k].rsample() for k in net.keys}

        du_gen_loss_sum = sum(dsc_gen_loss[k] for k in net.keys)

        u_cat = torch.cat([z[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only:
            return {"dsc_loss": self.lam_align * (dsc_loss+du_gen_loss_sum)}


        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        x_u1_nll = {
            k: -net.u2x[k](
                usamp1[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }


        x_nll = {
            k: -net.u2x[k](
                usamp[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }

        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        means = sum(u[k].mean for k in net.keys) / len(net.keys)
        scale = sum(u[k].stddev for k in net.keys) / len(net.keys)
        temp_D = D.Normal(means, scale)
        z_kl = {
            k: D.kl_divergence(
                z[k], temp_D
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }


        cosk = {}
        for i in range(len(net.keys)-1):
            cosk[net.keys[i]] = zsamp[net.keys[i]] @ zsamp[net.keys[i+1]].T

        cosk_p = {}
        for i in range(len(net.keys) - 1):
            cosk_p[net.keys[i]] = x_p[net.keys[i]] @ x_p[net.keys[i+1]].T
        z_p_nll = {}
        for i in range(len(net.keys)-1):
            z_p_nll[net.keys[i]] = (cosk_p[net.keys[i]]-cosk[net.keys[i]]).pow_(2)

        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.domain_weight[k] * x_elbo[k] for k in net.keys)
        z_kl_sum = sum(self.domain_weight[k] * z_kl[k] for k in net.keys)
        x_u1_nll_sum = sum(self.domain_weight[k] * x_u1_nll[k] for k in net.keys)
        z_p_sum = sum(z_p_nll[k].sum(dim=1).mean() for k in net.keys[:-1])

        vae_loss = self.lam_data * (x_elbo_sum+x_u1_nll_sum) + 0.05*z_kl_sum +0.05*z_p_sum



        gen_loss = vae_loss - self.lam_align * (dsc_loss) - du_gen_loss_sum

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,

        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def compute_losses_first(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xalt, xbch, xlbl, xdwt, xflag = data
        x_p = {}
        for k in net.keys:
            x_p[k] = xalt[k][:,-50:]
            xalt[k] = xalt[k][:, :-50]
        u, z, l = {}, {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xalt[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}

        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        cosk = {}
        for i in range(len(net.keys)-1):
            cosk[net.keys[i]] = usamp[net.keys[i]] @ usamp[net.keys[i+1]].T

        cosk_p = {}
        for i in range(len(net.keys) - 1):
            cosk_p[net.keys[i]] = x_p[net.keys[i]] @ x_p[net.keys[i+1]].T
        x_p_nll = {}
        for i in range(len(net.keys) - 1):
            x_p_nll[net.keys[i]] = (cosk_p[net.keys[i]]-cosk[net.keys[i]]).pow_(2)

        x_nll = {
            k: -net.u2x[k](
                usamp[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.domain_weight[k] * x_elbo[k] for k in net.keys)
        x_p_sum = sum(x_p_nll[k].sum(dim=1).mean() for k in net.keys[:-1])
        vae_loss = self.lam_data * x_elbo_sum +0.0001*x_p_sum

        gen_loss = vae_loss

        losses = {
            "dsc_loss": torch.tensor(0.0, device=self.net.device), "vae_loss": vae_loss, "gen_loss": gen_loss,

        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = torch.tensor(0.0, device=self.net.device)
        return losses



    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch
        if self.safe_burnin:
            for i in range(2):
                losses = self.compute_losses(data, epoch, dsc_only=True)
                self.net.zero_grad(set_to_none=True)
                losses["dsc_loss"].backward()  # Already scaled by lam_align
                self.dsc_optim.step()

            # Generator step
            losses = self.compute_losses(data, epoch)
            self.net.zero_grad(set_to_none=True)
            losses["gen_loss"].backward()
            self.vae_optim.step()
            return losses
        else:
            losses = self.compute_losses_first(data, epoch)
            self.net.zero_grad(set_to_none=True)
            losses["gen_loss"].backward()
            self.vae_optim.step()
            return losses

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )


import scanpy
import gc
import itertools
import scipy.sparse as sp
def normalize_sparse(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def gen_tf_gene_table(genes, tf_list, dTD):
    """
    Adapted from:
    Author: Jun Ding
    Project: SCDIFF2
    Ref: Ding, J., Aronow, B. J., Kaminski, N., Kitzmiller, J., Whitsett, J. A., & Bar-Joseph, Z.
    (2018). Reconstructing differentiation networks and their regulation from time series
    single-cell expression data. Genome research, 28(3), 383-395.
    """
    gene_names = [g.upper() for g in genes]
    TF_names = [g.upper() for g in tf_list]
    tf_gene_table = dict.fromkeys(tf_list)

    for i, tf in enumerate(tf_list):
        tf_gene_table[tf] = np.zeros(len(gene_names))
        _genes = dTD[tf]

        _existed_targets = list(set(_genes).intersection(gene_names))
        _idx_targets = map(lambda x: gene_names.index(x), _existed_targets)

        for _g in _idx_targets:
            tf_gene_table[tf][_g] = 1

    del gene_names
    del TF_names
    del _genes
    del _existed_targets
    del _idx_targets

    gc.collect()

    return tf_gene_table


def getGeneSetMatrix(_name, genes_upper, gene_sets_path):
    """
    Adapted from:
    Author: Jun Ding
    Project: SCDIFF2
    Ref: Ding, J., Aronow, B. J., Kaminski, N., Kitzmiller, J., Whitsett, J. A., & Bar-Joseph, Z.
    (2018). Reconstructing differentiation networks and their regulation from time series
    single-cell expression data. Genome research, 28(3), 383-395.
    """
    if _name[-3:] == 'gmt':
        print(f"GMT file {_name} loading ... ")
        filename = _name
        filepath = os.path.join(gene_sets_path, f"{filename}")

        with open(filepath) as genesets:
            pathway2gene = {line.strip().split("\t")[0]: line.strip().split("\t")[2:]
                            for line in genesets.readlines()}

        print(len(pathway2gene))

        gs = []
        for k, v in pathway2gene.items():
            gs += v

        print(f"Number of genes in {_name} {len(set(gs).intersection(genes_upper))}")

        pathway_list = pathway2gene.keys()
        pathway_gene_table = gen_tf_gene_table(genes_upper, pathway_list, pathway2gene)
        gene_set_matrix = np.array(list(pathway_gene_table.values()))
        keys = pathway_gene_table.keys()

        del pathway2gene
        del gs
        del pathway_list
        del pathway_gene_table

        gc.collect()


    elif _name == 'TF-DNA':

        # get TF-DNA dictionary
        # TF->DNA
        def getdTD(tfDNA):
            dTD = {}
            with open(tfDNA, 'r') as f:
                tfRows = f.readlines()
                tfRows = [item.strip().split() for item in tfRows]
                for row in tfRows:
                    itf = row[0].upper()
                    itarget = row[1].upper()
                    if itf not in dTD:
                        dTD[itf] = [itarget]
                    else:
                        dTD[itf].append(itarget)

            del tfRows
            del itf
            del itarget
            gc.collect()

            return dTD

        from collections import defaultdict

        def getdDT(dTD):
            gene_tf_dict = defaultdict(lambda: [])
            for key, val in dTD.items():
                for v in val:
                    gene_tf_dict[v.upper()] += [key.upper()]

            return gene_tf_dict

        tfDNA_file = os.path.join(gene_sets_path, f"Mouse_TF_targets.txt")
        dTD = getdTD(tfDNA_file)
        dDT = getdDT(dTD)

        tf_list = list(sorted(dTD.keys()))
        tf_list.remove('TF')

        tf_gene_table = gen_tf_gene_table(genes_upper, tf_list, dTD)
        gene_set_matrix = np.array(list(tf_gene_table.values()))
        keys = tf_gene_table.keys()

        del dTD
        del dDT
        del tf_list
        del tf_gene_table

        gc.collect()

    else:
        gene_set_matrix = None

    return gene_set_matrix, keys



class AnnDataset1(Dataset):
    def __init__(self, data, label_name: str = None, second_filepath: str = None,
                 variable_gene_name: str = None):
        """
        Anndata dataset.
        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array
        """

        super().__init__()

        self.data = data

        genes = self.data.var.index.values
        self.genes_upper = [g.upper() for g in genes]
        if label_name is not None:
            self.clusters_true = self.data.obs[label_name].values
        else:
            self.clusters_true = None

        self.N = self.data.shape[0]
        self.G = len(self.genes_upper)

        self.secondary_data = None
        if second_filepath is not None:
            self.secondary_data = np.load(second_filepath)
            assert len(self.secondary_data) == self.N, "The other file have same length as the main"

        if variable_gene_name is not None:
            _idx = np.where(self.data.var[variable_gene_name].values)[0]
            self.exp_variable_genes = self.data.X[:, _idx]
            self.variable_genes_names = self.data.var.index.values[_idx]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        main = self.data[idx].X.flatten()

        if self.secondary_data is not None:
            secondary = self.secondary_data[idx].flatten()
            return main, secondary
        else:
            return main


#--------------------------------- Public API ----------------------------------

@logged
def configure_dataset(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_gs: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_cell_type: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        use_uid: Optional[str] = None
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    prob_model
        Probabilistic generative model used by the decoder,
        must be one of ``{"Normal", "ZIN", "ZILN", "NB", "ZINB"}``.
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    use_uid
        Unique cell ID used to mark paired cells across multiple datasets
        (key in ``adata.obsm``)

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
        adata.layers[use_layer][adata.layers[use_layer]<0] = 0
    else:
        data_config["use_layer"] = None
        adata.X[adata.X<0] = 0
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = pd.Index(
            adata.obs[use_cell_type]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    if use_uid:
        if use_uid not in adata.obs:
            raise ValueError("Invalid `use_uid`!")
        data_config["use_uid"] = use_uid
    else:
        data_config["use_uid"] = None
    adata.uns[config.ANNDATA_KEY] = data_config
    scanpy.pp.neighbors(adata, key_added='gcn',use_rep=use_rep,n_neighbors=20)

    adj_adata = adata.obsp['gcn_connectivities']
    adj_adata = normalize_sparse(adj_adata)

    adj_adata.setdiag(1)

    adata.obsm[use_rep] = adj_adata* adata.obsm[use_rep]


    if use_gs:
        # scanpy.pp.highly_variable_genes(adata)

        gene = adata
        if adata.obs['domain'][0] != 'scRNA-seq':
            gene = adata.uns['gene']
            gene.obs['cell_type'] = adata.obs['cell_type']

        expression_only = AnnDataset1(gene, label_name='cell_type')
        genes_upper = expression_only.genes_upper
        prior_name = "c5.go.bp.v7.4.symbols.gmt+c2.cp.v7.4.symbols.gmt+TF-DNA"
        gene_sets_path = "./gene_sets/"
        if '+' in prior_name:
            prior_names_list = prior_name.split('+')

            _matrix_list = []
            _keys_list = []
            for _name in prior_names_list:
                _matrix, _keys = getGeneSetMatrix(_name, genes_upper, gene_sets_path)
                _matrix_list.append(_matrix)
                _keys_list.append(_keys)

            gene_set_matrix = np.concatenate(_matrix_list, axis=0)
            keys_all = list(itertools.chain(*_keys_list))

            del _matrix_list
            del _keys_list
            gc.collect()

        else:
            gene_set_matrix, keys_all = getGeneSetMatrix(prior_name, genes_upper, gene_sets_path)

        gene_Set_m = (gene.X).dot(gene_set_matrix.T)
        temp = AnnData(gene_Set_m)

        scanpy.tl.pca(temp, n_comps=5)
        adata.obsm[use_rep] = np.concatenate((adata.obsm[use_rep], temp.obsm['X_pca']), axis=1)
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]


@logged
class SCCROSSModel(Model):



    NET_TYPE = SCCROSS
    TRAINER_TYPE = SCCROSSTrainer

    GRAPH_BATCHES: int = 32  # Number of graph batches in each graph epoch
    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 48.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self, adatas: Mapping[str, AnnData], latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            random_seed: int = 0
    ) -> None:

        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.domains, x2u,u2z,z2u, u2x,du_gen ,all_ct = {}, {},{},{},{},  {}, set()
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )

            x2u[k] = select_encoder(data_config["prob_model"])(
                data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )
            u2z[k] = sc.ZEncoder(50, 50)
            z2u[k] = sc.ZDecoder(50,50)
            du_gen[k] = sc.Discriminator_gen(
            len(data_config["features"]), 2, n_batches=0,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            u2x[k] = select_decoder(data_config["prob_model"])(
                len(data_config["features"]),
                n_batches=max(data_config["batches"].size, 1)
            )
            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.domains[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for domain in self.domains.values():
            domain["cell_types"] = all_ct
        if shared_batches:
            all_batches = [domain["batches"] for domain in self.domains.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        du = sc.Discriminator(
            latent_dim, len(self.domains), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )


        prior = sc.Prior()
        super().__init__(
        x2u,u2z,z2u,u2x, du,du_gen, prior

        )

    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "SCCROSSModel", submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(  # pylint: disable=arguments-differ
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            normalize_u: bool = False,
            domain_weight: Optional[Mapping[str, float]] = None,
            lr: float = 1e-3, **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        domain_weight
            Relative domain weight (indexed by domain name)
        lr
            Learning rate
        **kwargs
            Additional keyword arguments passed to trainer
        """
        if domain_weight is None:
            domain_weight = {k: 1.0 for k in self.net.keys}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            normalize_u=normalize_u, domain_weight=domain_weight,
            optim="RMSprop", lr=lr, **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData],
            edge_weight: str = "weight", edge_sign: str = "sign",
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None
    ) -> None:
        r"""
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by domain name)
        graph
            Prior graph
        edge_weight
            Key of edge attribute for edge weight
        edge_sign
            Key of edge attribute for edge sign
        neg_samples
            Number of negative samples for each edge
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        directory
            Directory to store checkpoints and tensorboard logs
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.domains[key] for key in self.net.keys],
            mode="train"
        )


        batch_per_epoch = data.size * (1 - val_split) / data_batch_size

        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory
        )

    @torch.no_grad()
    def get_losses(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData],
            edge_weight: str = "weight", edge_sign: str = "sign",
            neg_samples: int = 10, data_batch_size: int = 128,
            graph_batch_size: int = AUTO
    ) -> Mapping[str, np.ndarray]:
        r"""
        Compute loss function values

        Parameters
        ----------
        adatas
            Datasets (indexed by domain name)
        graph
            Prior graph
        edge_weight
            Key of edge attribute for edge weight
        edge_sign
            Key of edge attribute for edge sign
        neg_samples
            Number of negative samples for each edge
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch

        Returns
        -------
        losses
            Loss function values
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.domains[key] for key in self.net.keys],
            mode="train"
        )


        return super().get_losses(
            data, data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Domain key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        u2z = self.net.u2z[key]
        data = AnnDataset(
            [adata], [self.domains[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for x, xalt, *_ in data_loader:
            xalt = xalt[:,:-50]
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )[0]
            # us = u.sample()
            z = u2z(u.mean)
            if n_sample:
                result.append(z.sample((n_sample,)).cpu().permute(1, 0, 2))
            else:
                result.append(z.mean.detach().cpu())

        return torch.cat(result).numpy()





    @torch.no_grad()
    def generate_cross(
            self, key1: str, key2: str, adata: AnnData, adata_other: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Domain key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key1]
        encoder_other = self.net.x2u[key2]

        u2z = self.net.u2z[key1]
        z2u = self.net.z2u[key2]
        u2x = self.net.u2x[key2]
        data = AnnDataset(
            [adata], [self.domains[key1]],
            mode="eval", getitem_size=batch_size
        )
        data_other = AnnDataset(
            [adata_other], [self.domains[key2]],
            mode="eval", getitem_size=batch_size
        )

        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        data_loader_other = DataLoader(
            data_other, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        result = []
        result_other = []
        l_other = torch.Tensor().cuda()

        for x, xalt, *_ in data_loader_other:
            xalt = xalt[:,:-50]
            u_other,l_other_1 = encoder_other(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )
            l_other = torch.cat((l_other, l_other_1))

            result_other.append(x.cpu())

        l_other = torch.mean(l_other)


        for x, xalt, *_ in data_loader:
            xalt = xalt[:,:-50]
            u,l = encoder(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )

            z = u2z(u.mean)
            u1 = z2u(z.mean)
            b = np.zeros(len(l), dtype=int)
            l = l/torch.mean(l)*l_other


            u1samp = u1.rsample()
            x_out = u2x(u1samp, b, l)

            result.append(x_out.sample().cpu())

        return torch.cat(result).numpy(),torch.cat(result_other).numpy()

    @torch.no_grad()
    def generate_batch(
            self, adatas: Mapping[str, AnnData],obs_from:str,name:str,num:int
    ):
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Domain key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()

        l_s = []
        z_s = torch.Tensor()

        for key,adata in adatas.items():
            x2u = self.net.x2u[key]
            u2z = self.net.u2z[key]
            adata_sub = adata[adata.obs[obs_from].isin([name])]
            data = AnnDataset(
                [adata_sub], [self.domains[key]],
                 mode="eval", getitem_size=len(adata_sub.obs)
            )
            data_loader = DataLoader(
                data, batch_size=1, shuffle=False,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
                persistent_workers=False
            )

            l_s_t = []


            for x, xalt, *_ in data_loader:
                xalt = xalt[:, :-50]

                u, l = x2u(
                    x.to(self.net.device, non_blocking=True),
                    xalt.to(self.net.device, non_blocking=True),
                    lazy_normalizer=True
                )
                z = u2z(u.mean)

                l = torch.mean(l.cpu())


                z_t = torch.mean(z.mean,dim=0,keepdim=True)

                l_s_t.append(l)
                z_s = torch.cat((z_s, z_t))

            l_s.append(np.mean(l_s_t))

        z_s_m = torch.mean(z_s,dim=0,keepdim=True)

        g = 0
        result_s = []

        for key,adata in adatas.items():
            z2u = self.net.z2u[key]
            u2x = self.net.u2x[key]

            u = z2u(z_s_m)
            l = l_s[g]
            b = 0
            g = g+1
            result = []

            for i in range(num):
                u1samp = u.rsample()
                x_out = u2x(u1samp, b, l)
                result.append(x_out.sample().cpu())

            result = torch.cat(result).numpy()
            adata_s = adata[:,adata.var.query("highly_variable").index.to_numpy().tolist()]
            result_a = scanpy.AnnData(result,var=adata_s.var)
            #result_a.obs[obs_from] = name
            result_s.append(result_a)

        return result_s














    @torch.no_grad()
    def generate_align(
            self, key1: str, adata: AnnData,  batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Domain key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key1]


        u2z = self.net.u2z[key1]
        z2u = self.net.z2u[key1]
        u2x = self.net.u2x[key1]

        data = AnnDataset(
            [adata], [self.domains[key1]],
            mode="eval", getitem_size=batch_size
        )


        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )



        result = []
        result_other = []





        for x, xalt, *_ in data_loader:

            u,l = encoder(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )

            z = u2z(u.mean)
            u1 = z2u(z.mean)
            b = np.zeros(len(l), dtype=int)
            l = l/torch.mean(l)*l
            u1samp = u1.rsample()
            x_out = u2x(u1samp, b, l)
            result.append(x_out.sample().cpu())
            result_other.append(x.cpu())


        return torch.cat(result).numpy(),torch.cat(result_other).numpy()






    def __repr__(self) -> str:
        return (
            f"SCCROSS model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )


