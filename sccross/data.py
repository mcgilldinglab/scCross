r"""
Auxiliary functions for :class:`anndata.AnnData` objects
that are not covered in :mod:`scanpy`.
"""

import os
from collections import defaultdict
from itertools import chain
from typing import Callable, List, Mapping, Optional

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import episcanpy as epi
import scipy.sparse
import scipy.stats
import sklearn.cluster
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.neighbors
import sklearn.utils.extmath
from anndata import AnnData
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn.preprocessing import normalize


from . import utils
from .utils import logged, smart_tqdm, Kws


def lsi(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
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







def mnn_prior(
        adatas: [AnnData]

) -> None:

    adatas_modify = adatas.copy()

    for i in range(len(adatas_modify)):
        if adatas_modify[i].obs['domain'][0] == 'scATAC-seq':
            adatas_modify[i] = adatas_modify[i].uns['gene']



    common_genes = set()
    for i in range(len(adatas_modify)):
        adata_i = adatas_modify[i].var.index.values
        atac_i_1 = []
        for g in adata_i:
            g = g.split('-')[0]
            atac_i_1.append(g)

        adatas_modify[i].var.index = atac_i_1
        adatas_modify[i].var_names_make_unique()
        if len(common_genes) == 0:
            common_genes = set(adatas_modify[i].var.index.values)
        else:
            common_genes &= set(adatas_modify[i].var.index.values)

    for i in range(len(adatas)):
        adatas_modify[i] = adatas_modify[i][:, list(common_genes)]

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
        adata: AnnData,gtf_file = './reference/gencode.vM30.annotation.gtf',key_added='gene',
                                upstream=2000,
                                feature_type='transcript',
                                annotation='HAVANA',
                                raw=False

) -> AnnData:
    geneAct = epi.tl.geneactivity(adata,
                                gtf_file=gtf_file,
                                key_added=key_added,
                                upstream=upstream,
                                feature_type=feature_type,
                                annotation=annotation,
                                raw=raw)

    sc.pp.normalize_total(geneAct)
    sc.pp.log1p(geneAct)
    sc.pp.scale(geneAct)
    return geneAct










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
