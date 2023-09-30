
import os

import anndata

import scanpy as sc

from matplotlib import rcParams

import sccross


rcParams["figure.figsize"] = (4, 4)

PATH = "s01_preprocessing"
os.makedirs(PATH, exist_ok=True)


rna = anndata.read_h5ad("Saunders-2018.h5ad")

met = anndata.read_h5ad("Luo-2017.h5ad")

atac = anndata.read_h5ad("10x-ATAC-Brain5k.h5ad")


rna.layers["raw_count"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, n_comps=100, use_highly_variable=True, svd_solver="auto")

rna.X = rna.layers["raw_count"]
del rna.layers["raw_count"]


sc.pp.neighbors(rna, n_pcs=100, metric="cosine")
sc.tl.umap(rna)


rna.obs["cell_type"].cat.set_categories([
    "Layer2/3", "Layer5a", "Layer5", "Layer5b", "Layer6",
    "Claustrum", "CGE", "MGE"
], inplace=True)



met.X = met.layers["norm"].copy()
sc.pp.log1p(met)
sc.pp.scale(met, max_value=10)
sc.tl.pca(met, n_comps=100, use_highly_variable=True, svd_solver="auto")


met.X = met.layers["norm"]
del met.layers["norm"]


sc.pp.neighbors(met, n_pcs=100, metric="cosine")
sc.tl.umap(met)


met.obs["cell_type"].cat.set_categories([
    "mL2/3", "mL4", "mL5-1", "mDL-1", "mDL-2", "mL5-2",
    "mL6-1", "mL6-2", "mDL-3", "mIn-1", "mVip",
    "mNdnf-1", "mNdnf-2", "mPv", "mSst-1", "mSst-2"
], inplace=True)


sccross.data.lsi(atac, n_components=100, use_highly_variable=False, n_iter=15)


sc.pp.neighbors(atac, n_pcs=100, use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)


atac.obs["cell_type"].cat.set_categories([
    "L2/3 IT", "L4", "L5 IT", "L6 IT", "L5 PT",
    "NP", "L6 CT", "Vip", "Pvalb", "Sst"
], inplace=True)


fig = sc.pl.umap(atac, color="cell_type", title="scATAC-seq cell type", return_fig=True)
fig.savefig(f"{PATH}/atac_ct.pdf")

atac2rna = sccross.data.geneActivity(atac)

rna.write("rna_preprocessed.h5ad", compression="gzip")
met.write("met_preprocessed.h5ad", compression="gzip")
atac.write("atac_preprocessed.h5ad", compression="gzip")
atac2rna.write("atac2rna.h5ad", compression="gzip")

