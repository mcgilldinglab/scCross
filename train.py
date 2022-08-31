import anndata
import itertools
import networkx as nx
import pandas as pd
import scanpy as sc
import sys
import scanpy as sc

import sccross
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

sccross.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

#Data preparing
rna = anndata.read_h5ad("../rna_preprocessed.h5ad")
atac = anndata.read_h5ad("../atac_preprocessed.h5ad")


sccross.models.configure_dataset(
    rna, "NB", use_highly_variable=False,
    use_layer="counts", use_rep="X_pca"
)
sccross.models.configure_dataset(
    atac, "NB", use_highly_variable=False,
    use_rep="X_lsi"
)


#Model training
cross = sccross.models.fit_SCCROSS(
    {"rna": rna, "atac": atac},
    fit_kws={"directory": "cross"}
)

cross.save("cross.dill")

#Data integration
rna.obsm["X_cross"] = cross.encode_data("rna", rna)
atac.obsm["X_cross"] = cross.encode_data("atac", atac)


sc.pp.neighbors(rna,use_rep='X_cross', metric="cosine")
sc.tl.leiden(rna)
ARI = adjusted_rand_score(rna.obs['cell_type'], rna.obs['leiden'])
NMI = normalized_mutual_info_score(rna.obs['cell_type'],rna.obs['leiden'])
print("ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))


sc.pp.neighbors(atac,use_rep='X_cross', metric="cosine")
sc.tl.leiden(atac)
ARI = adjusted_rand_score(atac.obs['cell_type'], atac.obs['leiden'])
NMI = normalized_mutual_info_score(atac.obs['cell_type'],atac.obs['leiden'])
print("ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))

combined = anndata.concat([rna, atac])


ASW = sccross.metrics.avg_silhouette_width(combined.obsm['X_cross'],combined.obs['cell_type'])
GCT = sccross.metrics.graph_connectivity(combined.obsm['X_cross'],combined.obs['cell_type'])
print("ASW: "+str(ASW)+" GCT: "+str(GCT))


#Data generation
rna.obsm['X_generation'] = sccross.generate("rna","atac", rna)
atac.obsm['X_generation'] = atac.X
combined = anndata.concat([rna, atac])
sc.pp.neighbors(combined,use_rep='X_generation', metric="cosine")
sc.tl.umap(combined)
sc.pl.umap(combined, color=['cell_type','domain'],add_outline=True,size=40,frameon=False,save='rna2atac.pdf')
