
import itertools
import os
from abc import abstractmethod
from typing import Any, List, Mapping, NoReturn, Optional, Tuple

import ignite
import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import normalize_edges
from ..utils import config, logged
from .base import Trainer, TrainingPlugin
from .data import ArrayDataset, DataLoader, GraphDataset, ParallelDataLoader
from .nn import autodevice
from .plugins import EarlyStopping, LRScheduler, Tensorboard


#----------------------- Component interface definitions -----------------------

class GraphEncoder(torch.nn.Module):

    r"""
    Abstract graph encoder
    """

    @abstractmethod
    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Distribution:
        r"""
        Encode graph to vertex latent distribution

        Parameters
        ----------
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        v
            Vertex latent distribution
            (:math:`n_{vertices} \times n_{features}`)
        """
        raise NotImplementedError  # pragma: no cover


class GraphDecoder(torch.nn.Module):

    r"""
    Abstract graph decoder
    """

    @abstractmethod
    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor,
            esgn: torch.Tensor
    ) -> D.Distribution:
        r"""
        Decode graph from vertex latent

        Parameters
        ----------
        v
            Vertex latent (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        recon
            Edge reconstruction distribution
            (:math:`n_{edges}`)
        """
        raise NotImplementedError  # pragma: no cover


class DataEncoder(torch.nn.Module):

    r"""
    Abstract data encoder
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> D.Distribution:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        u
            Sample latent distribution
        """
        raise NotImplementedError  # pragma: no cover


class DataDecoder(torch.nn.Module):

    r"""
    Abstract data decoder
    """

    @abstractmethod
    def forward(
            self, u: torch.Tensor, v: torch.Tensor
    ) -> D.Distribution:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class Discriminator(torch.nn.Module):

    r"""
    Abstract domain discriminator
    """

    @abstractmethod
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        Discriminate among domains

        Parameters
        ----------
        u
            Sample latent

        Returns
        -------
        logits
            Domain logits
        """
        raise NotImplementedError  # pragma: no cover


class Prior(torch.nn.Module):

    r"""
    Abstract prior distribution
    """

    @abstractmethod
    def forward(self) -> D.Distribution:
        r"""
        Get prior distribution

        Returns
        -------
        prior
            Prior distribution
        """
        raise NotImplementedError  # pragma: no cover


#------------------------ Network interface definition -------------------------

class CROSS(torch.nn.Module):



    def __init__(
            self,
            x2u: Mapping[str, DataEncoder],
            u2z: Mapping[str, DataEncoder],
            z2u: Mapping[str, DataDecoder],
            u2x: Mapping[str, DataDecoder],
            du: Discriminator,
            du_gen: Mapping[str, Discriminator], prior: Prior
    ) -> None:
        super().__init__()
        if not set(x2u.keys()) == set(u2x.keys())  != set():
            raise ValueError(
                "`x2u`, `u2x`, `idx` should share the same keys "
                "and non-empty!"
            )
        self.keys = list(x2u.keys())  # Keeps a specific order


        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2z = torch.nn.ModuleDict(u2z)
        self.z2u = torch.nn.ModuleDict(z2u)
        self.u2x = torch.nn.ModuleDict(u2x)

        self.du_gen = torch.nn.ModuleDict(du_gen)
        self.du = du
        self.prior = prior

        self.device = autodevice()

    @property
    def device(self) -> torch.device:
        r"""
        Device of the module
        """
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)

    def forward(self) -> NoReturn:
        r"""
        Invalidated forward operation
        """
        raise RuntimeError("scCross does not support forward operation!")



DataTensors = Tuple[
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]


@logged
class CROSSTrainer(Trainer):


    def __init__(
            self, net: CROSS, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            domain_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        required_kwargs = (
            "lam_data", "lam_kl", "lam_graph", "lam_align",
            "domain_weight", "optim", "lr"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        super().__init__(net)
        self.required_losses = []
        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.required_losses += ["dsc_loss", "vae_loss", "gen_loss"]
        self.earlystop_loss = "vae_loss"

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        if min(domain_weight.values()) < 0:
            raise ValueError("Domain weight must be non-negative!")
        normalizer = sum(domain_weight.values()) / len(domain_weight)
        self.domain_weight = {k: v / normalizer for k, v in domain_weight.items()}

        self.lr = lr
        self.vae_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.x2u.parameters(),
                self.net.u2x.parameters(),
                self.net.u2z.parameters(),
                self.net.z2u.parameters()
            ), lr=self.lr, **kwargs
        )
        self.dsc_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.du.parameters(),
                self.net.du_gen.parameters()),
            lr=self.lr, **kwargs
        )

        self.align_burnin: Optional[int] = None

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:  # pragma: no cover
        r"""
        Compute loss functions

        Parameters
        ----------
        data
            Data tensors
        epoch
            Current epoch number
        dsc_only
            Whether to compute the discriminator loss only

        Returns
        -------
        loss_dict
            Dict containing loss values
        """
        net = self.net
        x, xflag, eidx, ewt, esgn = data

        u = {k: net.x2u[k](x[k]) for k in net.keys}
        usamp = {k: u[k].rsample() for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = prior.sample(u_cat.shape)
            u_cat = u_cat + anneal * noise
        dsc_loss = F.cross_entropy(net.du(u_cat), xflag_cat)
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0).to(torch.int64) + (n_neg > 0).to(torch.int64)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        x_nll = {
            k: -net.u2x[k](
                usamp[k], vsamp[getattr(net, f"{k}_idx")]
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

        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo
        gen_loss = vae_loss - self.lam_align * dsc_loss

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        return losses

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:  # pragma: no cover
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each domain
        in the same order as domain keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, (eidx, ewt, esgn) = data[:K], data[K:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xflag, eidx, ewt, esgn

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:  # pragma: no cover
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        # Discriminator step
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

    @torch.no_grad()
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.eval()
        data = self.format_data(data)
        return self.compute_losses(data, engine.state.epoch)

    def fit(  # pylint: disable=arguments-differ
            self, data: ArrayDataset, val_split: float = None,
            data_batch_size: int = None, graph_batch_size: int = None,
            align_burnin: int = None, safe_burnin: bool = True,
            max_epochs: int = None, patience: Optional[int] = None,
            reduce_lr_patience: Optional[int] = None,
            wait_n_lrs: Optional[int] = None,
            random_seed: int = None, directory: Optional[os.PathLike] = None,
            plugins: Optional[List[TrainingPlugin]] = None
    ) -> None:
        r"""
        Fit network

        Parameters
        ----------
        data
            Data dataset
        graph
            Graph dataset
        val_split
            Validation split
        data_batch_size
            Number of samples in each data minibatch
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
        random_seed
            Random seed
        directory
            Directory to store checkpoints and tensorboard logs
        plugins
            Optional list of training plugins
        """
        required_kwargs = (
            "val_split", "data_batch_size", "graph_batch_size",
            "align_burnin", "max_epochs", "random_seed"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        if patience and reduce_lr_patience and reduce_lr_patience >= patience:
            self.logger.warning(
                "Parameter `reduce_lr_patience` should be smaller than `patience`, "
                "otherwise learning rate scheduling would be ineffective."
            )



        data.getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH))

        data_train, data_val = data.random_split([1 - val_split, val_split], random_state=random_seed)
        data_train.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        data_val.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)


        train_loader = ParallelDataLoader(
            DataLoader(
                data_train, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),

            cycle_flags=[False]
        )
        val_loader = ParallelDataLoader(
            DataLoader(
                data_val, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),

            cycle_flags=[False]
        )

        self.align_burnin = align_burnin
        self.safe_burnin = safe_burnin

        default_plugins = [Tensorboard()]
        if reduce_lr_patience:
            default_plugins.append(LRScheduler(
                self.vae_optim, self.dsc_optim,
                monitor=self.earlystop_loss, patience=reduce_lr_patience,
                burnin=self.align_burnin if safe_burnin else 0
            ))
        if patience:
            default_plugins.append(EarlyStopping(
                monitor=self.earlystop_loss, patience=patience,
                burnin=self.align_burnin if safe_burnin else 0,
                wait_n_lrs=wait_n_lrs or 0
            ))
        plugins = default_plugins + (plugins or [])
        try:
            super().fit(
                train_loader, val_loader=val_loader,
                max_epochs=max_epochs, random_seed=random_seed,
                directory=directory, plugins=plugins
            )
        finally:
            data.clean()
            data_train.clean()
            data_val.clean()
            self.align_burnin = None
            self.safe_burnin = None


    def get_losses(  # pylint: disable=arguments-differ
            self, data: ArrayDataset, graph: GraphDataset,
            data_batch_size: int = None, graph_batch_size: int = None,
            random_seed: int = None
    ) -> Mapping[str, float]:
        required_kwargs = ("data_batch_size", "graph_batch_size", "random_seed")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        self.enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        self.esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        self.eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        data.getitem_size = data_batch_size
        graph.getitem_size = graph_batch_size
        data.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        graph.prepare_shuffle(num_workers=config.GRAPH_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        loader = ParallelDataLoader(
            DataLoader(
                data, batch_size=1, shuffle=True, drop_last=False,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),
            DataLoader(
                graph, batch_size=1, shuffle=True, drop_last=False,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),
            cycle_flags=[False, True]
        )

        try:
            losses = super().get_losses(loader)
        finally:
            data.clean()
            graph.clean()
            self.eidx = None
            self.enorm = None
            self.esgn = None

        return losses

    def state_dict(self) -> Mapping[str, Any]:
        return {
            **super().state_dict(),
            "vae_optim": self.vae_optim.state_dict(),
            "dsc_optim": self.dsc_optim.state_dict()
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.vae_optim.load_state_dict(state_dict.pop("vae_optim"))
        self.dsc_optim.load_state_dict(state_dict.pop("dsc_optim"))
        super().load_state_dict(state_dict)
