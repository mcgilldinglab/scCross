
import gc

import anndata
import faiss

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd

import sccross


rna = anndata.read_h5ad("Cao-2020.h5ad", backed="r")
atac = anndata.read_h5ad("Domcke-2020.h5ad", backed="r")


rna_organ_fracs = rna.obs["Organ"].str.lower().value_counts() / rna.shape[0]
atac_organ_fracs = atac.obs["tissue"].str.lower().value_counts() / atac.shape[0]
cmp_organ_fracs = pd.DataFrame({"rna": rna_organ_fracs, "atac": atac_organ_fracs})

organ_min_fracs = cmp_organ_fracs.min(axis=1)


rs = np.random.RandomState(0)
rna_subidx, atac_subidx = [], []
for organ, min_frac in organ_min_fracs.iteritems():
    print(f"Dealing with {organ}...")
    rna_idx = np.where(rna.obs["Organ"].str.lower() == organ)[0]
    rna_subidx.append(rs.choice(rna_idx, round(min_frac * rna.shape[0]), replace=False))
    atac_idx = np.where(atac.obs["tissue"].str.lower() == organ)[0]
    atac_subidx.append(rs.choice(atac_idx, round(min_frac * atac.shape[0]), replace=False))
rna_subidx = np.concatenate(rna_subidx)
rna_mask = np.zeros(rna.shape[0], dtype=bool)
rna_mask[rna_subidx] = True
rna.obs["mask"] = rna_mask
atac_subidx = np.concatenate(atac_subidx)
atac_mask = np.zeros(atac.shape[0], dtype=bool)
atac_mask[atac_subidx] = True
atac.obs["mask"] = atac_mask


rna_organ_balancing = np.sqrt(cmp_organ_fracs["atac"] / cmp_organ_fracs["rna"])
atac_organ_balancing = np.sqrt(cmp_organ_fracs["rna"] / cmp_organ_fracs["atac"])


rna.obs["organ_balancing"] = rna_organ_balancing.loc[rna.obs["Organ"].str.lower()].to_numpy()
atac.obs["organ_balancing"] = atac_organ_balancing.loc[atac.obs["tissue"].str.lower()].to_numpy()


rna = rna.to_memory()


hvg_df = sc.pp.highly_variable_genes(rna[rna.obs["mask"], :], n_top_genes=4000, flavor="seurat_v3", inplace=False)
rna.var = rna.var.assign(**hvg_df.to_dict(orient="series"))


rna.layers["raw_counts"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
rna = rna[:, rna.var.highly_variable]
gc.collect()



X = rna.X
X_masked = X[rna.obs["mask"]]
mean = X_masked.mean(axis=0).A1
std = np.sqrt(X_masked.power(2).mean(axis=0).A1 - mean ** 2)
X = X.toarray()
X -= mean
X /= std
X = X.clip(-10, 10)
X_masked = X[rna.obs["mask"]]


u, s, vh = randomized_svd(X_masked.T @ X_masked, 100, n_iter=15, random_state=0)
rna.obsm["X_pca"] = X @ vh.T


rna.X = rna.layers["raw_counts"]
del rna.layers["raw_counts"], X, X_masked, mean, std, u, s, vh
gc.collect()

sc.pp.neighbors(rna, n_pcs=rna.obsm["X_pca"].shape[1], metric="cosine")
sc.tl.umap(rna)
del rna.obsp["connectivities"], rna.obsp["distances"]
gc.collect()



kmeans = faiss.Kmeans(rna.obsm["X_pca"].shape[1], 100000, gpu=True, seed=0)
kmeans.train(rna.obsm["X_pca"][rna.obs["mask"]])
_, rna.obs["metacell"] = kmeans.index.search(rna.obsm["X_pca"], 1)

# %%
rna.obs["metacell"] = pd.Categorical(rna.obs["metacell"])
rna.obs["metacell"].cat.rename_categories(lambda x: f"rna-metacell-{x}", inplace=True)
rna.obs["n_cells"] = 1



rna.write("rna_preprocessed.h5ad", compression="gzip")



atac = atac.to_memory()



X = sccross.utils.tfidf(atac.X)
X = Normalizer(norm="l1").fit_transform(X)
X = np.log1p(X * 1e4)


X_masked = X[atac.obs["mask"]]
u, s, vh = randomized_svd(X_masked, 100, n_iter=15, random_state=0)
X_lsi = X @ vh.T / s
X_lsi -= X_lsi.mean(axis=1, keepdims=True)
X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
atac.obsm["X_lsi"] = X_lsi.astype(np.float32)


atac = atac[:, atac.var["highly_variable"]]
del X, X_masked, X_lsi, u, s, vh
gc.collect()


sc.pp.neighbors(atac, n_pcs=atac.obsm["X_lsi"].shape[1], use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)
del atac.obsp["connectivities"], atac.obsp["distances"]
gc.collect()


kmeans = faiss.Kmeans(atac.obsm["X_lsi"].shape[1], 40000, gpu=True, seed=0)
kmeans.train(atac.obsm["X_lsi"][atac.obs["mask"]])
_, atac.obs["metacell"] = kmeans.index.search(atac.obsm["X_lsi"], 1)

# %%
atac.obs["metacell"] = pd.Categorical(atac.obs["metacell"])
atac.obs["metacell"].cat.rename_categories(lambda x: f"atac-metacell-{x}", inplace=True)
atac.obs["n_cells"] = 1

atac2rna = sccross.data.geneActivity(atac,gtf_file='/reference/gencode.v38.annotation.gtf.gz')

atac.write("atac_preprocessed.h5ad", compression="gzip")
atac2rna.write("atac2rna.h5ad", compression="gzip")

