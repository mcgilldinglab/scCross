

import collections
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from ..num import EPS
from . import cross
from .nn import GraphConv
from .prob import ZILN, ZIN, ZINB


#-------------------------- Network modules for CROSS ---------------------------

class GraphEncoder(cross.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class GraphDecoder(cross.GraphDecoder):

    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(cross.DataEncoder):

    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Compute normalizer

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        l
            Normalizer
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Normalize data

        Parameters
        ----------
        x
            Input data
        l
            Normalizer

        Returns
        -------
        xnorm
            Normalized data
        """
        raise NotImplementedError  # pragma: no cover

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xalt: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xalt
            Alternative input data
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xalt` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xalt is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xalt is used instead.
        """
        if xalt.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xalt
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):

    r"""
    Vanilla data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x

def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

from typing import Iterable


class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.
    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    :param use_relu: Whether to have `ReLU` layers or not
    :param bias: Whether to learn bias in linear layers or not
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        #use_batch_norm: bool = False,
        use_relu: bool = True,
        #use_relu: bool = False,
        bias: bool = True,
        RNA_mode = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.LayerNorm(n_out, eps=0.0001),
                            nn.LeakyReLU() if RNA_mode else None,
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.0001)
                            if use_batch_norm
                            else None,
                            nn.LeakyReLU() if use_relu else nn.ReLU(),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                    zip(layers_dim[:-1], layers_dim[1:])
                )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.
        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x






class NBDataEncoder(cross.DataEncoder):

    r"""
    Data encoder for negative binomial data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 1e4
    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        ptr_dim1 =ptr_dim-50
        ptr_dim2 = 50
        for layer in range(self.h_depth):
            setattr(self, f"linear1_{layer}", torch.nn.Linear(ptr_dim1, h_dim))
            setattr(self, f"act1_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn1_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout1_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim1 = h_dim
        self.loc1 = torch.nn.Linear(ptr_dim1, out_features-3)
        self.std_lin1 = torch.nn.Linear(ptr_dim1, out_features-3)

        h_dim = 25
        for layer in range(self.h_depth):
            setattr(self, f"linear2_{layer}", torch.nn.Linear(ptr_dim2, h_dim))
            setattr(self, f"act2_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn2_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout2_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim2 = h_dim
        self.loc2 = torch.nn.Linear(ptr_dim2, 3)
        self.std_lin2 = torch.nn.Linear(ptr_dim2, 3)


    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xalt: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xalt
            Alternative input data
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xalt` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xalt is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xalt is used instead.
        """

        ptr = xalt
        l = self.compute_l(x)
        ptr1 = ptr[:,:-50]
        ptr2 = ptr[:,-50:]

        for layer in range(self.h_depth):
            ptr1 = getattr(self, f"linear1_{layer}")(ptr1)
            ptr1 = getattr(self, f"act1_{layer}")(ptr1)
            ptr1 = getattr(self, f"bn1_{layer}")(ptr1)
            ptr1 = getattr(self, f"dropout1_{layer}")(ptr1)
        loc1 = self.loc1(ptr1)
        std1 = F.softplus(self.std_lin1(ptr1)) + EPS

        for layer in range(self.h_depth):
            ptr2 = getattr(self, f"linear2_{layer}")(ptr2)
            ptr2 = getattr(self, f"act2_{layer}")(ptr2)
            ptr2 = getattr(self, f"bn2_{layer}")(ptr2)
            ptr2 = getattr(self, f"dropout2_{layer}")(ptr2)
        loc2 = self.loc2(ptr2)
        std2 = F.softplus(self.std_lin2(ptr2)) + EPS
        loc = torch.cat((loc1,loc2),1)
        std = torch.cat((std1, std2), 1)


        return D.Normal(loc, std), l



    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class ZEncoder(cross.DataEncoder):

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 1, h_dim: int = 16,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        ptr = x
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class ZDecoder(cross.DataDecoder):

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 1, h_dim: int = 50,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        ptr = x
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)

class DataDecoder(cross.DataDecoder):

    r"""
    Abstract data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent
        b
            Batch index
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class NormalDataDecoder(DataDecoder):

    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    r"""
    Zero-inflated normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZIN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZIN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder1(DataDecoder):

    r"""
    Zero-inflated log-normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    r"""
    Zero-inflated log-normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.h_depth = 1
        ptr_dim = 50
        h_dim = out_features
        #print(out_features)

        setattr(self, f"linear_", torch.nn.Linear(ptr_dim, h_dim))
        setattr(self, f"act_", torch.nn.LeakyReLU(negative_slope=0.2))
        setattr(self, f"bn_", torch.nn.BatchNorm1d(h_dim))
        setattr(self, f"dropout_", torch.nn.Dropout(p=0.2))

    def forward(
            self, u: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        ptr = u
        # print(u.shape)
        # print(v.shape)

        ptr = getattr(self, f"linear_")(ptr)
        ptr = getattr(self, f"act_")(ptr)
        ptr = getattr(self, f"bn_")(ptr)
        ptr = getattr(self, f"dropout_")(ptr)
        scale = F.softplus(self.scale_lin[b])
        loc = scale * ptr + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):

    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.h_depth = 1
        ptr_dim = 50
        h_dim = out_features
        #print(out_features)

        setattr(self, f"linear_", torch.nn.Linear(ptr_dim, h_dim))
        setattr(self, f"act_", torch.nn.LeakyReLU(negative_slope=0.2))
        setattr(self, f"bn_", torch.nn.BatchNorm1d(h_dim))
        setattr(self, f"dropout_", torch.nn.Dropout(p=0.2))

    def forward(
            self, u: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        ptr = u
        #print(u.shape)
        #print(v.shape)

        ptr = getattr(self, f"linear_")(ptr)
        ptr = getattr(self, f"act_")(ptr)
        ptr = getattr(self, f"bn_")(ptr)
        ptr = getattr(self, f"dropout_")(ptr)

        logit_mu = ptr
        mu = F.softmax(logit_mu, dim=1)*l
        #mu = F.softmax(logit_mu, dim=1)
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log()
        )



class NBDataDecoder1(DataDecoder):

    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class ZINBDataDecoder(NBDataDecoder):

    r"""
    Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZINB:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class Discriminator(torch.nn.Sequential, cross.Discriminator):

    r"""
    Domain discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Discriminator_gen(torch.nn.Sequential, cross.Discriminator):

    r"""
    Domain discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return super().forward(x)

class Classifier(torch.nn.Linear):

    r"""
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


class Prior(cross.Prior):

    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


#-------------------- Network modules for independent CROSS ---------------------

class IndDataDecoder(DataDecoder):

    r"""
    Data decoder mixin that makes decoding independent of feature latent

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(  # pylint: disable=unused-argument
            self, in_features: int, out_features: int, n_batches: int = 1
    ) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.v = torch.nn.Parameter(torch.zeros(out_features, in_features))

    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, b: torch.Tensor,
            l: Optional[torch.Tensor]
    ) -> D.Distribution:
        r"""
        Decode data from sample latent

        Parameters
        ----------
        u
            Sample latent
        b
            Batch index
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        return super().forward(u, self.v, b, l)


class IndNormalDataDocoder(IndDataDecoder, NormalDataDecoder):
    r"""
    Normal data decoder independent of feature latent
    """


class IndZINDataDecoder(IndDataDecoder, ZINDataDecoder):
    r"""
    Zero-inflated normal data decoder independent of feature latent
    """


class IndZILNDataDecoder(IndDataDecoder, ZILNDataDecoder):
    r"""
    Zero-inflated log-normal data decoder independent of feature latent
    """


class IndNBDataDecoder(IndDataDecoder, NBDataDecoder):
    r"""
    Negative binomial data decoder independent of feature latent
    """


class IndZINBDataDecoder(IndDataDecoder, ZINBDataDecoder):
    r"""
    Zero-inflated negative binomial data decoder independent of feature latent
    """
