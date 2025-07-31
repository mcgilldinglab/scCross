r"""
Auxiliary functions for :class:`anndata.AnnData` objects that are not covered in :mod:`scanpy`.
This project includes code adapted from Copyright (c) 2025, Gao Lab's project under the MIT License and released under the MIT License.

"""

import os
from collections import defaultdict
from itertools import chain
from typing import Callable, List, Mapping, Optional, Union, Any, Iterable, Set
import collections
import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import episcanpy as epi
import scipy.sparse
import scipy.stats
from scipy import sparse
import sklearn.cluster
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.neighbors
import sklearn.utils.extmath
from anndata import AnnData
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn.preprocessing import normalize
import pybedtools
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from functools import reduce
from operator import add
from tqdm.auto import tqdm
from . import utils
from .utils import logged, smart_tqdm, Kws
import re





def lsi(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = utils.tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi




class ConstrainedDataFrame(pd.DataFrame):



    def __init__(self, *args, **kwargs) -> None:
        df = pd.DataFrame(*args, **kwargs)
        df = self.rectify(df)
        self.verify(df)
        super().__init__(df)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.verify(self)

    @property
    def _constructor(self) -> type:
        return type(self)

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:

        return df

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        r"""
            verify
        """

    @property
    def df(self) -> pd.DataFrame:

        return pd.DataFrame(self)

    def __repr__(self) -> str:

        return repr(self.df)

class Bed(ConstrainedDataFrame):



    COLUMNS = pd.Index(
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "blockCount",
            "blockSizes",
            "blockStarts",
        ]
    )

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Bed, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Bed, cls).verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(df.columns != cls.COLUMNS):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: os.PathLike) -> "Bed":

        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[: loaded.shape[1]]
        return cls(loaded)

    def write_bed(self, fname: os.PathLike, ncols: Optional[int] = None) -> None:

        if ncols and ncols < 3:
            raise ValueError("`ncols` must be larger than 3!")
        df = self.df.iloc[:, :ncols] if ncols else self
        df.to_csv(fname, sep="\t", header=False, index=False)

    def to_bedtool(self) -> pybedtools.BedTool:

        return BedTool(
            Interval(
                row["chrom"],
                row["chromStart"],
                row["chromEnd"],
                name=row["name"],
                score=row["score"],
                strand=row["strand"],
            )
            for _, row in self.iterrows()
        )

    def nucleotide_content(self, fasta: os.PathLike) -> pd.DataFrame:

        result = self.to_bedtool().nucleotide_content(
            fi=os.fspath(fasta), s=True
        )
        result = pd.DataFrame(
            np.stack([interval.fields[6:15] for interval in result]),
            columns=[
                r"%AT",
                r"%GC",
                r"#A",
                r"#C",
                r"#G",
                r"#T",
                r"#N",
                r"#other",
                r"length",
            ],
        ).astype(
            {
                r"%AT": float,
                r"%GC": float,
                r"#A": int,
                r"#C": int,
                r"#G": int,
                r"#T": int,
                r"#N": int,
                r"#other": int,
                r"length": int,
            }
        )
        pybedtools.cleanup()
        return result

    def strand_specific_start_site(self) -> "Bed":

        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def strand_specific_end_site(self) -> "Bed":

        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromStart"] = df.loc[pos_strand, "chromEnd"] - 1
        df.loc[neg_strand, "chromEnd"] = df.loc[neg_strand, "chromStart"] + 1
        return type(self)(df)

    def expand(
        self,
        upstream: int,
        downstream: int,
        chr_len: Optional[Mapping[str, int]] = None,
    ) -> "Bed":

        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)



class Gtf(ConstrainedDataFrame):


    COLUMNS = pd.Index(
        [
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ]
    )

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Gtf, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("start", "end"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("seqname", "start", "end"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.sort_index(axis=1, key=cls._column_key)

    @classmethod
    def _column_key(cls, x: pd.Index) -> np.ndarray:
        x = cls.COLUMNS.get_indexer(x)
        x[x < 0] = x.max() + 1
        return x

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Gtf, cls).verify(df)
        if len(df.columns) < len(cls.COLUMNS) or np.any(
            df.columns[: len(cls.COLUMNS)] != cls.COLUMNS
        ):
            raise ValueError("Invalid GTF format!")

    @classmethod
    def read_gtf(cls, fname: os.PathLike) -> "Gtf":

        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[: loaded.shape[1]]
        return cls(loaded)

    def split_attribute(self) -> "Gtf":

        pattern = re.compile(r'([^\s]+) "([^"]+)";')
        splitted = pd.DataFrame.from_records(
            np.vectorize(lambda x: {key: val for key, val in pattern.findall(x)})(
                self["attribute"]
            ),
            index=self.index,
        )
        if set(self.COLUMNS).intersection(splitted.columns):
            self.logger.warning(
                "Splitted attribute names overlap standard GTF fields! "
                "The standard fields are overwritten!"
            )
        return self.assign(**splitted)

    def to_bed(self, name: Optional[str] = None) -> Bed:

        bed_df = pd.DataFrame(self, copy=True).loc[
            :, ("seqname", "start", "end", "score", "strand")
        ]
        bed_df.insert(
            3, "name", np.repeat(".", len(bed_df)) if name is None else self[name]
        )
        bed_df["start"] -= 1  # Convert to zero-based
        bed_df.columns = ("chrom", "chromStart", "chromEnd", "name", "score", "strand")
        return Bed(bed_df)



def interval_dist(x: Interval, y: Interval) -> int:

    if x.chrom != y.chrom:
        return np.inf * (-1 if x.chrom < y.chrom else 1)
    if x.start < y.stop and y.start < x.stop:
        return 0
    if x.stop <= y.start:
        return x.stop - y.start - 1
    if y.stop <= x.start:
        return x.start - y.stop + 1







def window_matrix(
    left_o: Union[Bed, str],
    right_o: Union[Bed, str],
    window_size: int,
    left_sorted: bool = False,
    right_sorted: bool = False,
    attr_fn: Optional[Callable[[Interval, Interval, float], Mapping[str, Any]]] = None,
) -> sparse.coo_matrix:




    if isinstance(left_o, Bed):
        pbar_total = len(left_o)
        left = left_o.to_bedtool()
    else:
        pbar_total = None
        left = pybedtools.BedTool(left_o)

    left_names = [iv.name for iv in left]
    left_idx = {name: i for i, name in enumerate(left_names)}

    if isinstance(left_o, Bed):
        pbar_total = len(left_o)
        left = left_o.to_bedtool()
    else:
        pbar_total = None
        left = pybedtools.BedTool(left_o)

    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)  # Resumable iterator

    if isinstance(right_o, Bed):
        right = right_o.to_bedtool()
    else:
        right = pybedtools.BedTool(right_o)

    right_names = [iv.name for iv in right]
    right_idx = {name: j for j, name in enumerate(right_names)}

    if isinstance(right_o, Bed):
        right = right_o.to_bedtool()
    else:
        right = pybedtools.BedTool(right_o)
    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)  # Resumable iterator

    attr_fn = attr_fn or (lambda l, r, d: {})
    if pbar_total is not None:
        left = tqdm(left, total=pbar_total, desc="Feature process", disable=True)



    rows, cols, data = [], [], []
    window = collections.OrderedDict()


    for l in left:

        for r in list(window.keys()):
            d = interval_dist(l, r)
            if -window_size <= d <= window_size:

                wgt = attr_fn(l, r, d).get("weight", 1.0)

                rows.append(left_idx[l.name])
                cols.append(right_idx[r.name])
                data.append(wgt)
            elif d > window_size:
                del window[r]
            else:
                break
        else:

            for r in right:
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    wgt = attr_fn(l, r, d).get("weight", 1.0)
                    rows.append(left_idx[l.name])
                    cols.append(right_idx[r.name])
                    data.append(wgt)
                    window[r] = None
                elif d > window_size:
                    continue

                if d < -window_size:
                    break

    pybedtools.cleanup()


    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(left_names), len(right_names)))
    return matrix, left_names, right_names







def compose_matrix(*matrices: sparse.spmatrix) -> sparse.spmatrix:
    """
    Sum multiple adjacency matrices into one.
    """
    return sum(matrices)



def peak_annote_matrix(
    rna: AnnData,
    *others: AnnData,
    gene_region: str = "combined",
    promoter_len: int = 2000,
    extend_range: int = 0,
    extend_fn: Callable[[int], float] = lambda x: ((x + 1000) / 1000) ** (-0.75),
    signs: Optional[List[int]] = None,
    propagate_highly_variable: bool = True,
) -> sparse.coo_matrix:
    signs = signs or [1] * len(others)
    # prepare BEDs
    rna_bed = Bed(rna.var.assign(name=rna.var_names))
    other_beds = [Bed(other.var.assign(name=other.var_names)) for other in others]
    if gene_region == "promoter":
        rna_bed = rna_bed.strand_specific_start_site().expand(promoter_len, 0)
    elif gene_region == "combined":
        rna_bed = rna_bed.expand(promoter_len, 0)
    # compute matrices
    matrices = []
    for other_bed, sign in zip(other_beds, signs):
        mat, lnames, rnames = window_matrix(
            rna_bed, other_bed, extend_range,
            attr_fn=lambda l, r, d, s=sign: {"weight": extend_fn(abs(d))}
        )
        matrices.append(mat)
    combined = compose_matrix(*matrices)

    # optional propagation of HVG using matrix powers (not shown)
    return combined


def get_gene_annotation(
    adata: AnnData,
    var_by: str = None,
    gtf: os.PathLike = None,
    gtf_by: str = None,
    by_func: Optional[Callable] = None,
) -> None:

    if gtf is None:
        raise ValueError("Missing required argument `gtf`!")
    if gtf_by is None:
        raise ValueError("Missing required argument `gtf_by`!")
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = Gtf.read_gtf(gtf).query("feature == 'gene'").split_attribute()
    if by_func:
        by_func = np.vectorize(by_func)
        var_by = by_func(var_by)
        gtf[gtf_by] = by_func(gtf[gtf_by])
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_by], keep="last"
    )
    merge_df = (
        pd.concat(
            [
                pd.DataFrame(gtf.to_bed(name=gtf_by)),
                pd.DataFrame(gtf).drop(
                    columns=Gtf.COLUMNS
                ),
            ],
            axis=1,
        )
        .set_index(gtf_by)
        .reindex(var_by)
        .set_index(adata.var.index)
    )
    adata.var = adata.var.assign(**merge_df)



def mnn_prior(
        adatas: [AnnData]

) -> None:
    r"""
        MNN prior generation

        Parameters
        ----------
        adatas
            Input dataset

    """
    common_genes = set()
    adatas_modify = adatas.copy()
    rna = adatas_modify[0]
    atac = adatas_modify[1]
    gtf = None
    for i in range(len(adatas_modify)):
        if adatas_modify[i].obs['domain'][0] == 'scATAC-seq':
            gtf = adatas_modify[i].obs['path'][0]


    for i in range(len(adatas_modify)):
        if adatas_modify[i].obs['domain'][0] == 'scATAC-seq':
            atac = adatas_modify[i]
            split = adatas_modify[i].var_names.str.split(r"[:-]")
            adatas_modify[i].var["chrom"] = split.map(lambda x: x[0])
            adatas_modify[i].var["chromStart"] = split.map(lambda x: x[1]).astype(int)
            adatas_modify[i].var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
        else:

            if len(common_genes) == 0:
                common_genes = set(adatas_modify[i].var.index.values)
            else:
                common_genes &= set(adatas_modify[i].var.index.values)

            if 'chrom' not in adatas_modify[i].var.columns:
                get_gene_annotation(
                    adatas_modify[i], gtf=gtf,
                    gtf_by="gene_name"
                )

            if adatas_modify[i].obs['domain'][0] == 'scRNA-seq':
                rna = adatas_modify[i]
                if 'highly_variable' in rna.var.columns:
                    hv_genes = rna.var.query("highly_variable").index.to_numpy().tolist()
                    common_genes &= set(hv_genes)
                continue



    peak_mat = peak_annote_matrix(rna, atac)

    atac2rna = AnnData(
        X=atac.X.dot(peak_mat.T),
        obs=atac.obs,
        var=rna.var
    )

    sc.pp.normalize_total(atac2rna)
    sc.pp.log1p(atac2rna)
    sc.pp.scale(atac2rna)

    for i in range(len(adatas_modify)):
        if adatas_modify[i].obs['domain'][0] == 'scATAC-seq':
            adatas_modify[i] = atac2rna

    if len(adatas_modify) > 2:
        for i in range(len(adatas)):
            adatas_modify[i] = adatas_modify[i][:, list(common_genes)]
    elif 'highly_variable' in rna.var.columns:
        for i in range(len(adatas)):
            adatas_modify[i] = adatas_modify[i][:, rna.var.query("highly_variable").index.to_numpy().tolist()]


    adatas_mnn = sc.external.pp.mnn_correct(*adatas_modify, k=20)
    adatas_mnn = adatas_mnn[0]
    sc.tl.pca(adatas_mnn, n_comps=50)
    for i in range(len(adatas)):
        adata_temp = adatas_mnn[adatas_mnn.obs['batch'].isin([str(i)])]
        if adatas[i].obs['domain'][0] == 'scATAC-seq':
            adatas[i].obsm['X_lsi'] = np.concatenate((adatas[i].obsm['X_lsi'], adata_temp.obsm['X_pca']), axis=1)
        else:
            adatas[i].obsm['X_pca'] = np.concatenate((adatas[i].obsm['X_pca'], adata_temp.obsm['X_pca']), axis=1)

def geneActivity(
        adata: AnnData,gtf_file = './reference/gencode.vM30.annotation.gtf',
                                upstream=2000,
                                feature_type='transcript',
                                annotation='HAVANA',
                                raw=False

) -> AnnData:
    r"""
        Gene activity score calculation


        Parameters
        ----------
        adata
            Input dataset
        gtf_file
            GTF reference path
        upstream
            upstream finding
        feature_type
            feature type
        annotation
            annotation type
        raw
            raw data or not

        Returns
        -------
        AnnData
            Gene activity score data
            with shape :math:`n_{cell} \times n_{gene}`
    """


    adata.obs['path'] = gtf_file


    return adata










def aggregate_obs(
        adata: AnnData, by: str, X_agg: Optional[str] = "sum",
        obs_agg: Optional[Mapping[str, str]] = None,
        obsm_agg: Optional[Mapping[str, str]] = None,
        layers_agg: Optional[Mapping[str, str]] = None
) -> AnnData:
    r"""
    Aggregate obs in a given dataset by certain categories

    Parameters
    ----------
    adata
        Dataset to be aggregated
    by
        Specify a column in ``adata.obs`` used for aggregation,
        must be discrete.
    X_agg
        Aggregation function for ``adata.X``, must be one of
        ``{"sum", "mean", ``None``}``. Setting to ``None`` discards
        the ``adata.X`` matrix.
    obs_agg
        Aggregation methods for ``adata.obs``, indexed by obs columns,
        must be one of ``{"sum", "mean", "majority"}``, where ``"sum"``
        and ``"mean"`` are for continuous data, and ``"majority"`` is for
        discrete data. Fields not specified will be discarded.
    obsm_agg
        Aggregation methods for ``adata.obsm``, indexed by obsm keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    layers_agg
        Aggregation methods for ``adata.layers``, indexed by layer keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.

    Returns
    -------
    aggregated
        Aggregated dataset
    """
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) \
        if pd.api.types.is_categorical_dtype(by) \
        else pd.Index(np.unique(by))
    agg_sum = scipy.sparse.coo_matrix((
        np.ones(adata.shape[0]), (
            agg_idx.get_indexer(by),
            np.arange(adata.shape[0])
        )
    )).tocsr()
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: agg_sum @ x,
        "mean": lambda x: agg_mean @ x,
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy()
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame({
        k: agg_method[v](adata.obs[k])
        for k, v in obs_agg.items()
    }, index=agg_idx.astype(str))
    obsm = {
        k: agg_method[v](adata.obsm[k])
        for k, v in obsm_agg.items()
    }
    layers = {
        k: agg_method[v](adata.layers[k])
        for k, v in layers_agg.items()
    }
    for c in obs:
        if pd.api.types.is_categorical_dtype(adata.obs[c]):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return AnnData(
        X=X, obs=obs, var=adata.var,
        obsm=obsm, varm=adata.varm, layers=layers
    )




def extract_rank_genes_groups(
        adata: AnnData, groups: Optional[List[str]] = None,
        filter_by: str = "pvals_adj < 0.01", sort_by: str = "scores",
        ascending: str = False
) -> pd.DataFrame:
    r"""
    Extract result of :func:`scanpy.tl.rank_genes_groups` in the form of
    marker gene data frame for specific cell groups

    Parameters
    ----------
    adata
        Input dataset
    groups
        Target groups for which markers should be extracted,
        by default extract all groups.
    filter_by
        Marker filtering criteria (passed to :meth:`pandas.DataFrame.query`)
    sort_by
        Column used for sorting markers
    ascending
        Whether to sort in ascending order

    Returns
    -------
    marker_df
        Extracted marker data frame

    Note
    ----
    Markers shared by multiple groups will be assign to the group
    with highest score.
    """
    if "rank_genes_groups" not in adata.uns:
        raise ValueError("Please call `sc.tl.rank_genes_groups` first!")
    if groups is None:
        groups = adata.uns["rank_genes_groups"][sort_by].dtype.names
    df = pd.concat([
        pd.DataFrame({
            k: np.asarray(v[g])
            for k, v in adata.uns["rank_genes_groups"].items()
            if k != "params"
        }).assign(group=g)
        for g in groups
    ])
    df["group"] = pd.Categorical(df["group"], categories=groups)
    df = df.sort_values(
        sort_by, ascending=ascending
    ).drop_duplicates(
        subset=["names"], keep="first"
    ).sort_values(
        ["group", sort_by], ascending=[True, ascending]
    ).query(filter_by)
    df = df.reset_index(drop=True)
    return df





@logged
def get_metacells(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        common: bool = True, seed: int = 0,
        agg_kwargs: Optional[List[Kws]] = None
) -> List[AnnData]:
    r"""
    Aggregate datasets into metacells

    Parameters
    ----------
    *adatas
        Datasets to be correlated
    use_rep
        Data representation based on which to cluster meta-cells
    n_meta
        Number of metacells to use
    common
        Whether to return only metacells common to all datasets
    seed
        Random seed for k-Means clustering
    agg_kwargs
        Keyword arguments per dataset passed to :func:`aggregate_obs`

    Returns
    -------
    adatas
        A list of AnnData objects containing the metacells

    Note
    ----
    When a single dataset is provided, the metacells are clustered
    with the dataset itself.
    When multiple datasets are provided, the metacells are clustered
    jointly with all datasets.
    """
    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    if n_meta is None:
        raise ValueError("Missing required argument `n_meta`!")
    adatas = [
        AnnData(
            X=adata.X,
            obs=adata.obs.set_index(adata.obs_names + f"-{i}"), var=adata.var,
            obsm=adata.obsm, varm=adata.varm, layers=adata.layers
        ) for i, adata in enumerate(adatas)
    ]  # Avoid unwanted updates to the input objects

    get_metacells.logger.info("Clustering metacells...")
    combined = anndata.concat(adatas)
    try:
        import faiss
        kmeans = faiss.Kmeans(
            combined.obsm[use_rep].shape[1], n_meta,
            gpu=False, seed=seed
        )
        kmeans.train(combined.obsm[use_rep])
        _, combined.obs["metacell"] = kmeans.index.search(combined.obsm[use_rep], 1)
    except ImportError:
        get_metacells.logger.warning(
            "`faiss` is not installed, using `sklearn` instead... "
            "This might be slow with a large number of cells. "
            "Consider installing `faiss` following the guide from "
            "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
        )
        kmeans = sklearn.cluster.KMeans(n_clusters=n_meta, random_state=seed)
        combined.obs["metacell"] = kmeans.fit_predict(combined.obsm[use_rep])
    for adata in adatas:
        adata.obs["metacell"] = combined[adata.obs_names].obs["metacell"]

    get_metacells.logger.info("Aggregating metacells...")
    agg_kwargs = agg_kwargs or [{}] * len(adatas)
    if not len(agg_kwargs) == len(adatas):
        raise ValueError("Length of `agg_kwargs` must match the number of datasets!")
    adatas = [
        aggregate_obs(adata, "metacell", **kwargs)
        for adata, kwargs in zip(adatas, agg_kwargs)
    ]
    if common:
        common_metacells = list(set.intersection(*(
            set(adata.obs_names) for adata in adatas
        )))
        if len(common_metacells) == 0:
            raise RuntimeError("No common metacells found!")
        return [adata[common_metacells].copy() for adata in adatas]
    return adatas


def _metacell_corr(
        *adatas: AnnData, skeleton: nx.Graph = None, method: str = "spr"
) -> nx.Graph:
    if skeleton is None:
        raise ValueError("Missing required argument `skeleton`!")
    for adata in adatas:
        sc.pp.normalize_total(adata)
    if set.intersection(*(set(adata.var_names) for adata in adatas)):
        raise ValueError("Overlapping features are currently not supported!")
    adata = anndata.concat(adatas, axis=1)
    edgelist = nx.to_pandas_edgelist(skeleton)
    source = adata.var_names.get_indexer(edgelist["source"])
    target = adata.var_names.get_indexer(edgelist["target"])
    if method == "pcc":
        sc.pp.log1p(adata)
        X = utils.densify(adata.X.T)
    elif method == "spr":
        X = utils.densify(adata.X.T)
        X = np.array([scipy.stats.rankdata(x) for x in X])
    else:
        raise ValueError(f"Unrecognized method: {method}!")
    mean = X.mean(axis=1)
    meansq = np.square(X).mean(axis=1)
    std = np.sqrt(meansq - np.square(mean))
    edgelist["corr"] = np.array([
        ((X[s] * X[t]).mean() - mean[s] * mean[t]) / (std[s] * std[t])
        for s, t in zip(source, target)
    ])
    return nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=type(skeleton))


@logged
def metacell_corr(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        skeleton: nx.Graph = None, method: str = "spr"
) -> nx.Graph:
    r"""
    Metacell based correlation

    Parameters
    ----------
    *adatas
        Datasets to be correlated, where ``.X`` are raw counts
        (indexed by domain name)
    use_rep
        Data representation based on which to cluster meta-cells
    n_meta
        Number of metacells to use
    skeleton
        Skeleton graph determining which pair of features to correlate
    method
        Correlation method, must be one of {"pcc", "spr"}

    Returns
    -------
    corr
        A skeleton-based graph containing correlation
        as edge attribute "corr"
    """
    for adata in adatas:
        if not utils.all_counts(adata.X):
            raise ValueError("``.X`` must contain raw counts!")
    adatas = get_metacells(*adatas, use_rep=use_rep, n_meta=n_meta, common=True)
    metacell_corr.logger.info(
        "Computing correlation on %d common metacells...",
        adatas[0].shape[0]
    )
    return _metacell_corr(*adatas, skeleton=skeleton, method=method)


def _metacell_regr(
        *adatas: AnnData, skeleton: nx.DiGraph = None,
        model: str = "Lasso", **kwargs
) -> nx.DiGraph:
    if skeleton is None:
        raise ValueError("Missing required argument `skeleton`!")
    for adata in adatas:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    if set.intersection(*(set(adata.var_names) for adata in adatas)):
        raise ValueError("Overlapping features are currently not supported!")
    adata = anndata.concat(adatas, axis=1)

    targets = [node for node, in_degree in skeleton.in_degree() if in_degree]
    biadj = biadjacency_matrix(
        skeleton, adata.var_names, targets, weight=None
    ).astype(bool).T.tocsr()
    X = utils.densify(adata.X)
    Y = utils.densify(adata[:, targets].X.T)
    coef = []
    model = getattr(sklearn.linear_model, model)
    for target, y, mask in smart_tqdm(zip(targets, Y, biadj), total=len(targets)):
        X_ = X[:, mask.indices]
        lm = model(**kwargs).fit(X_, y)
        coef.append(pd.DataFrame({
            "source": adata.var_names[mask.indices],
            "target": target,
            "regr": lm.coef_
        }))
    coef = pd.concat(coef)
    return nx.from_pandas_edgelist(coef, edge_attr=True, create_using=type(skeleton))


@logged
def metacell_regr(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        skeleton: nx.DiGraph = None, model: str = "Lasso", **kwargs
) -> nx.DiGraph:
    r"""
    Metacell-based regression

    Parameters
    ----------
    *adatas
        Datasets to be correlated, where ``.X`` are raw counts
        (indexed by domain name)
    use_rep
        Data representation based on which to cluster meta-cells
    n_meta
        Number of metacells to use
    skeleton
        Skeleton graph determining which pair of features to correlate
    model
        Regression model (should be a class name under
        :mod:`sklearn.linear_model`)
    **kwargs
        Additional keyword arguments are passed to the regression model

    Returns
    -------
    regr
        A skeleton-based graph containing regression weights
        as edge attribute "regr"
    """
    for adata in adatas:
        if not utils.all_counts(adata.X):
            raise ValueError("``.X`` must contain raw counts!")
    adatas = get_metacells(*adatas, use_rep=use_rep, n_meta=n_meta, common=True)
    metacell_regr.logger.info(
        "Computing regression on %d common metacells...",
        adatas[0].shape[0]
    )
    return _metacell_regr(*adatas, skeleton=skeleton, model=model, **kwargs)
