r"""
Probability distributions
"""


import torch.distributions as D
import torch.nn.functional as F


import functools
import os

import numpy as np
import pynvml
from torch.nn.modules.batchnorm import _NormBase

import pathlib
import shutil

import ignite.contrib.handlers.tensorboard_logger as tb
import parse

from torch.optim.lr_scheduler import ReduceLROnPlateau
import tempfile

from typing import Any, Iterable, List, Mapping, Optional
import dill
import ignite
import torch

from ..utils import DelayedKeyboardInterrupt, config, logged, EPS
from abc import abstractmethod

EPOCH_STARTED = ignite.engine.Events.EPOCH_STARTED
EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
ITERATION_COMPLETED = ignite.engine.Events.ITERATION_COMPLETED
EXCEPTION_RAISED = ignite.engine.Events.EXCEPTION_RAISED
COMPLETED = ignite.engine.Events.COMPLETED


EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
TERMINATE = ignite.engine.Events.TERMINATE
COMPLETED = ignite.engine.Events.COMPLETED


@logged
class Trainer:

    r"""
    Abstract trainer class

    Parameters
    ----------
    net
        Network module to be trained

    Note
    ----
    Subclasses should populate ``required_losses``, and additionally
    define optimizers here.
    """

    def __init__(self, net: torch.nn.Module) -> None:
        self.net = net
        self.required_losses: List[str] = []

    @abstractmethod
    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        r"""
        A single training step

        Parameters
        ----------
        engine
            Training engine
        data
            Data of the training step

        Returns
        -------
        loss_dict
            Dict containing training loss values
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        r"""
        A single validation step

        Parameters
        ----------
        engine
            Validation engine
        data
            Data of the validation step

        Returns
        -------
        loss_dict
            Dict containing validation loss values
        """
        raise NotImplementedError  # pragma: no cover

    def report_metrics(
            self, train_state: ignite.engine.State,
            val_state: Optional[ignite.engine.State]
    ) -> None:
        r"""
        Report loss values during training

        Parameters
        ----------
        train_state
            Training engine state
        val_state
            Validation engine state
        """
        if train_state.epoch % config.PRINT_LOSS_INTERVAL:
            return
        train_metrics = {
            key: float(f"{val:.3f}")
            for key, val in train_state.metrics.items()
        }
        val_metrics = {
            key: float(f"{val:.3f}")
            for key, val in val_state.metrics.items()
        } if val_state else None
        self.logger.info(
            "[Epoch %d] train=%s, val=%s, %.1fs elapsed",
            train_state.epoch, train_metrics, val_metrics,
            train_state.times["EPOCH_COMPLETED"]  # Also includes validator time
        )

    def fit(
            self, train_loader: Iterable, val_loader: Optional[Iterable] = None,
            max_epochs: int = 100, random_seed: int = 0,
            directory: Optional[os.PathLike] = None,
            plugins: Optional[List["TrainingPlugin"]] = None
    ) -> None:
        r"""
        Fit network

        Parameters
        ----------
        train_loader
            Training data loader
        val_loader
            Validation data loader
        max_epochs
            Maximal number of epochs
        random_seed
            Random seed
        directory
            Training directory
        plugins
            Optional list of training plugins
        """
        interrupt_delayer = DelayedKeyboardInterrupt()
        directory = pathlib.Path(directory or tempfile.mkdtemp(prefix=config.TMP_PREFIX))
        self.logger.info("Using training directory: \"%s\"", directory)

        # Construct engines
        train_engine = ignite.engine.Engine(self.train_step)
        val_engine = ignite.engine.Engine(self.val_step) if val_loader else None

        delay_interrupt = interrupt_delayer.__enter__
        train_engine.add_event_handler(EPOCH_STARTED, delay_interrupt)
        train_engine.add_event_handler(COMPLETED, delay_interrupt)

        # Exception handling
        train_engine.add_event_handler(ITERATION_COMPLETED, ignite.handlers.TerminateOnNan())

        @train_engine.on(EXCEPTION_RAISED)
        def _handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and config.ALLOW_TRAINING_INTERRUPTION:
                self.logger.info("Stopping training due to user interrupt...")
                engine.terminate()
            else:
                raise e

        # Compute metrics
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(train_engine, item)
            if val_engine:
                ignite.metrics.Average(
                    output_transform=lambda output, item=item: output[item]
                ).attach(val_engine, item)

        if val_engine:
            @train_engine.on(EPOCH_COMPLETED)
            def _validate(engine):
                val_engine.run(
                    val_loader, max_epochs=engine.state.epoch
                )  # Bumps max_epochs by 1 per training epoch, so validator resumes for 1 epoch

        @train_engine.on(EPOCH_COMPLETED)
        def _report_metrics(engine):
            self.report_metrics(engine.state, val_engine.state if val_engine else None)

        for plugin in plugins or []:
            plugin.attach(
                net=self.net, trainer=self,
                train_engine=train_engine, val_engine=val_engine,
                train_loader=train_loader, val_loader=val_loader,
                directory=directory
            )

        restore_interrupt = lambda: interrupt_delayer.__exit__(None, None, None)
        train_engine.add_event_handler(EPOCH_COMPLETED, restore_interrupt)
        train_engine.add_event_handler(COMPLETED, restore_interrupt)

        # Start engines
        torch.manual_seed(random_seed)
        train_engine.run(train_loader, max_epochs=max_epochs)

        torch.cuda.empty_cache()  # Works even if GPU is unavailable

    def get_losses(self, loader: Iterable) -> Mapping[str, float]:
        r"""
        Get loss values for given data

        Parameters
        ----------
        loader
            Data loader

        Returns
        -------
        loss_dict
            Dict containing loss values
        """
        engine = ignite.engine.Engine(self.val_step)
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(engine, item)
        engine.run(loader, max_epochs=1)
        torch.cuda.empty_cache()  # Works even if GPU is unavailable
        return engine.state.metrics

    def state_dict(self) -> Mapping[str, Any]:
        r"""
        State dict

        Returns
        -------
        state_dict
            State dict
        """
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        r"""
        Load state from a state dict

        Parameters
        ----------
        state_dict
            State dict
        """


@logged
class Model:

    r"""
    Abstract model class

    Parameters
    ----------
    net
        Network type
    *args
        Positional arguments are passed to the network constructor
    **kwargs
        Keyword arguments are passed to the network constructor

    Note
    ----
    Subclasses may override arguments for API definition.
    """

    NET_TYPE = torch.nn.Module
    TRAINER_TYPE = Trainer

    def __init__(self, *args, **kwargs) -> None:
        self._net = self.NET_TYPE(*args, **kwargs)
        self._trainer: Optional[Trainer] = None  # Constructed upon compile

    @property
    def net(self) -> torch.nn.Module:
        r"""
        Neural network module in the model (read-only)
        """
        return self._net

    @property
    def trainer(self) -> Trainer:
        r"""
        Trainer of the neural network module (read-only)
        """
        if self._trainer is None:
            raise RuntimeError(
                "No trainer has been registered! "
                "Please call `.compile()` first."
            )
        return self._trainer

    def compile(self, *args, **kwargs) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        trainer
            Trainer type
        *args
            Positional arguments are passed to the trainer constructor
        **kwargs
            Keyword arguments are passed to the trainer constructor

        Note
        ----
        Subclasses may override arguments for API definition.
        """
        if self._trainer:
            self.logger.warning(
                "`compile` has already been called. "
                "Previous trainer will be overwritten!"
            )
        self._trainer = self.TRAINER_TYPE(self.net, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:
        r"""
        Alias of ``.trainer.fit``.

        Parameters
        ----------
        *args
            Positional arguments are passed to the ``.trainer.fit`` method
        **kwargs
            Keyword arguments are passed to the ``.trainer.fit`` method

        Note
        ----
        Subclasses may override arguments for API definition.
        """
        self.trainer.fit(*args, **kwargs)

    def get_losses(self, *args, **kwargs) -> Mapping[str, float]:
        r"""
        Alias of ``.trainer.get_losses``.

        Parameters
        ----------
        *args
            Positional arguments are passed to the ``.trainer.get_losses`` method
        **kwargs
            Keyword arguments are passed to the ``.trainer.get_losses`` method

        Returns
        -------
        loss_dict
            Dict containing loss values
        """
        return self.trainer.get_losses(*args, **kwargs)

    def save(self, fname: os.PathLike) -> None:
        r"""
        Save model to file

        Parameters
        ----------
        file
            Specifies path to the file

        Note
        ----
        Only the network is saved but not the trainer
        """
        fname = pathlib.Path(fname)
        trainer_backup, self._trainer = self._trainer, None
        device_backup, self.net.device = self.net.device, torch.device("cpu")
        with fname.open("wb") as f:
            dill.dump(self, f, protocol=4, byref=False, recurse=True)
        self.net.device = device_backup
        self._trainer = trainer_backup

    @staticmethod
    def load(fname: os.PathLike) -> "Model":
        r"""
        Load model from file

        Parameters
        ----------
        fname
            Specifies path to the file

        Returns
        -------
        model
            Loaded model
        """
        fname = pathlib.Path(fname)
        with fname.open("rb") as f:
            model = dill.load(f)
        model.net.device = autodevice()
        return model





@logged
class TrainingPlugin:

    r"""
    Plugin used to extend the training process with certain functions
    """

    @abstractmethod
    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        r"""
        Attach custom handlers to training or validation engine

        Parameters
        ----------
        net
            Network module
        trainer
            Trainer object
        train_engine
            Training engine
        val_engine
            Validation engine
        train_loader
            Training data loader
        val_loader
            Validation data loader
        directory
            Training directory
        """
        raise NotImplementedError  # pragma: no cover




#----------------------------- Utility functions -------------------------------

def freeze_running_stats(m: torch.nn.Module) -> None:
    r"""
    Selectively stops normalization layers from updating running stats

    Parameters
    ----------
    m
        Network module
    """
    if isinstance(m, _NormBase):
        m.eval()


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    r"""
    Get torch computation device automatically
    based on GPU availability and memory usage

    Returns
    -------
    device
        Computation device
    """
    used_device = -1
    if not config.CPU_ONLY:
        try:
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free for i in range(pynvml.nvmlDeviceGetCount())
            ])
            for item in config.MASKED_GPUS:
                free_mems[item] = -1
            best_devices = np.where(free_mems == free_mems.max())[0]
            used_device = np.random.choice(best_devices, 1)[0]
            if free_mems[used_device] < 0:
                used_device = -1
        except pynvml.NVMLError:
            pass
    if used_device == -1:
        autodevice.logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    autodevice.logger.info("Using GPU %d as computation device.", used_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda")



class Tensorboard(TrainingPlugin):

    r"""
    Training logging via tensorboard
    """

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        tb_directory = directory / "tensorboard"
        if tb_directory.exists():
            shutil.rmtree(tb_directory)

        tb_logger = tb.TensorboardLogger(
            log_dir=tb_directory,
            flush_secs=config.TENSORBOARD_FLUSH_SECS
        )
        tb_logger.attach(
            train_engine,
            log_handler=tb.OutputHandler(
                tag="train", metric_names=trainer.required_losses
            ), event_name=EPOCH_COMPLETED
        )
        if val_engine:
            tb_logger.attach(
                val_engine,
                log_handler=tb.OutputHandler(
                    tag="val", metric_names=trainer.required_losses
                ), event_name=EPOCH_COMPLETED
            )
        train_engine.add_event_handler(COMPLETED, tb_logger.close)


@logged
class EarlyStopping(TrainingPlugin):

    r"""
    Early stop model training when loss no longer decreases

    Parameters
    ----------
    monitor
        Loss to monitor
    patience
        Patience to stop early
    burnin
        Burn-in epochs to skip before initializing early stopping
    wait_n_lrs
        Wait n learning rate scheduling events before starting early stopping
    """

    def __init__(
            self, monitor: str, patience: int,
            burnin: int = 0, wait_n_lrs: int = 0
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.burnin = burnin
        self.wait_n_lrs = wait_n_lrs

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        for item in directory.glob("checkpoint_*.pt"):
            item.unlink()

        score_engine = val_engine if val_engine else train_engine
        score_function = lambda engine: -score_engine.state.metrics[self.monitor]
        event_filter = (
            lambda engine, event: event > self.burnin and engine.state.n_lrs >= self.wait_n_lrs
        ) if self.wait_n_lrs else (
            lambda engine, event: event > self.burnin
        )
        event = EPOCH_COMPLETED(event_filter=event_filter)  # pylint: disable=not-callable
        train_engine.add_event_handler(
            event, ignite.handlers.Checkpoint(
                {"net": net, "trainer": trainer},
                ignite.handlers.DiskSaver(
                    directory, atomic=True, create_dir=True, require_empty=False
                ), score_function=score_function,
                filename_pattern="checkpoint_{global_step}.pt",
                n_saved=config.CHECKPOINT_SAVE_NUMBERS,
                global_step_transform=ignite.handlers.global_step_from_engine(train_engine)
            )
        )
        train_engine.add_event_handler(
            event, ignite.handlers.EarlyStopping(
                patience=self.patience,
                score_function=score_function,
                trainer=train_engine
            )
        )

        @train_engine.on(COMPLETED | TERMINATE)
        def _(engine):
            nan_flag = any(
                not bool(torch.isfinite(item).all())
                for item in (engine.state.output or {}).values()
            )
            ckpts = sorted([
                parse.parse("checkpoint_{epoch:d}.pt", item.name).named["epoch"]
                for item in directory.glob("checkpoint_*.pt")
            ], reverse=True)
            if ckpts and nan_flag and train_engine.state.epoch == ckpts[0]:
                self.logger.warning(
                    "The most recent checkpoint \"%d\" can be corrupted by NaNs, "
                    "will thus be discarded.", ckpts[0]
                )
                ckpts = ckpts[1:]
            if ckpts:
                self.logger.info("Restoring checkpoint \"%d\"...", ckpts[0])
                loaded = torch.load(directory / f"checkpoint_{ckpts[0]}.pt")
                net.load_state_dict(loaded["net"])
                trainer.load_state_dict(loaded["trainer"])
            else:
                self.logger.info(
                    "No usable checkpoint found. "
                    "Skipping checkpoint restoration."
                )


@logged
class LRScheduler(TrainingPlugin):

    r"""
    Reduce learning rate on loss plateau

    Parameters
    ----------
    *optims
        Optimizers
    monitor
        Loss to monitor
    patience
        Patience to reduce learning rate
    burnin
        Burn-in epochs to skip before initializing learning rate scheduling
    """

    def __init__(
            self, *optims: torch.optim.Optimizer, monitor: str = None,
            patience: int = None, burnin: int = 0
    ) -> None:
        super().__init__()
        if monitor is None:
            raise ValueError("`monitor` must be specified!")
        self.monitor = monitor
        if patience is None:
            raise ValueError("`patience` must be specified!")
        self.schedulers = [
            ReduceLROnPlateau(optim, patience=patience, verbose=True)
            for optim in optims
        ]
        self.burnin = burnin

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        score_engine = val_engine if val_engine else train_engine
        event_filter = lambda engine, event: event > self.burnin
        for scheduler in self.schedulers:
            scheduler.last_epoch = self.burnin
        train_engine.state.n_lrs = 0

        @train_engine.on(EPOCH_COMPLETED(event_filter=event_filter))  # pylint: disable=not-callable
        def _():
            update_flags = set()
            for scheduler in self.schedulers:
                old_lr = scheduler.optimizer.param_groups[0]["lr"]
                scheduler.step(score_engine.state.metrics[self.monitor])
                new_lr = scheduler.optimizer.param_groups[0]["lr"]
                update_flags.add(new_lr != old_lr)
            if len(update_flags) != 1:
                raise RuntimeError("Learning rates are out of sync!")
            if update_flags.pop():
                train_engine.state.n_lrs += 1
                self.logger.info("Learning rate reduction: step %d", train_engine.state.n_lrs)



#-------------------------------- Distributions --------------------------------

class MSE(D.Distribution):

    r"""
    A "sham" distribution that outputs negative MSE on ``log_prob``

    Parameters
    ----------
    loc
        Mean of the distribution
    """

    def __init__(self, loc: torch.Tensor) -> None:
        super().__init__(validate_args=False)
        self.loc = loc

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc


class RMSE(MSE):

    r"""
    A "sham" distribution that outputs negative RMSE on ``log_prob``

    Parameters
    ----------
    loc
        Mean of the distribution
    """

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value).sqrt()


class ZIN(D.Normal):

    r"""
    Zero-inflated normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the normal distribution
    scale
        Scale of the normal distribution
    """

    def __init__(
            self, zi_logits: torch.Tensor,
            loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob


class ZILN(D.LogNormal):

    r"""
    Zero-inflated log-normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the log-normal distribution
    scale
        Scale of the log-normal distribution
    """

    def __init__(
            self, zi_logits: torch.Tensor,
            loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        zi_log_prob = torch.empty_like(value)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = z_zi_logits - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = D.LogNormal(
            self.loc[~z_mask], self.scale[~z_mask]
        ).log_prob(value[~z_mask]) - F.softplus(nz_zi_logits)
        return zi_log_prob


class ZINB(D.NegativeBinomial):

    r"""
    Zero-inflated negative binomial distribution

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    total_count
        Total count of the negative binomial distribution
    logits
        Logits of the negative binomial distribution
    """

    def __init__(
            self, zi_logits: torch.Tensor,
            total_count: torch.Tensor, logits: torch.Tensor = None
    ) -> None:
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob
