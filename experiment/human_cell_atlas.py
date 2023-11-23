import anndata
import scanpy as sc
import sccross
import pandas as pd
import numpy as np
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

# Read data
rcParams["figure.figsize"] = (4, 4)
rna = anndata.read_h5ad("../data/human_cell_atlas/rna_preprocessed.h5ad")
atac = anndata.read_h5ad("../data/human_cell_atlas/atac_preprocessed.h5ad")

# meta cell
sc.pp.neighbors(rna, n_pcs=rna.obsm["X_pca"].shape[1], use_rep="X_pca", metric="cosine")
sc.tl.leiden(rna)
rna.obs['metacell'] = rna.obs['leiden']

rna_agg = sccross.data.aggregate_obs(
    rna, by="metacell", X_agg="sum",
    obs_agg={
        "cell_type": "majority", "Organ": "majority", "domain": "majority",
        "n_cells": "sum", "organ_balancing": "sum"
    },
    obsm_agg={"X_pca": "mean", "X_umap": "mean"}
)


atac_agg = sccross.data.aggregate_obs(
    atac, by="metacell", X_agg="sum",
    obs_agg={
        "cell_type": "majority", "tissue": "majority", "domain": "majority",
        "n_cells": "sum", "organ_balancing": "sum"
    },
    obsm_agg={"X_lsi": "mean", "X_umap": "mean"}
)




# Configure data
sccross.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer = 'counts',
     use_rep="X_pca"
)

sccross.models.configure_dataset(
    atac, "NB", use_highly_variable=False,
    use_rep="X_lsi"
)

# MNN prior

sccross.data.mnn_prior([rna_agg,atac_agg])

for i in range(len(rna.obs)):
    rna[i].obsm['X_pca'] = np.concatenate((rna[i].obsm['X_pca'], rna_agg[rna_agg.obs['metacell']==rna[i].obs['metacell']].obsm['X_pca'][-50:]), axis=1)

for i in range(len(atac.obs)):
    atac[i].obsm['X_lsi'] = np.concatenate((atac[i].obsm['X_lsi'], atac_agg[atac_agg.obs['metacell']==atac[i].obs['metacell']].obsm['X_lsi'][-50:]), axis=1)


# Training
cross = sccross.models.fit_SCCROSS(
    {"rna": rna, "atac": atac},
    fit_kws={"directory": "sccross"}
)


# Save model
cross.save("cross.dill")
#cross = sccross.models.load_model("cross.dill")


# Integration benchmark
rna.obsm["X_cross"] = cross.encode_data("rna", rna)
atac.obsm["X_cross"] = cross.encode_data("atac", atac)


combined = anndata.concat([rna, atac])

sc.pp.neighbors(combined, use_rep="X_cross", metric="cosine")
sc.tl.umap(combined)
sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65, save='integration.pdf')

sc.tl.leiden(rna)
sc.tl.leiden(atac)

ARI = adjusted_rand_score(rna.obs['cell_type'], rna.obs['leiden'])
NMI = normalized_mutual_info_score(rna.obs['cell_type'],rna.obs['leiden'])
print("RNA:ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))


ARI = adjusted_rand_score(atac.obs['cell_type'], atac.obs['leiden'])
NMI = normalized_mutual_info_score(atac.obs['cell_type'],atac.obs['leiden'])
print("ATAC:ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))

ASW = sccross.metrics.avg_silhouette_width(combined.obsm['X_cross'],combined.obs['cell_type'])
ASWb = sccross.metrics.avg_silhouette_width_batch(combined.obsm['X_cross'],combined.obs['domain'],combined.obs['cell_type'])
GCT = sccross.metrics.graph_connectivity(combined.obsm['X_cross'],combined.obs['cell_type'])
print("ASW: "+str(ASW)+"ASWb: "+str(ASWb)+"GCT: "+str(GCT))










