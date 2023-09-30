import anndata
import scanpy as sc
import sccross
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

# Read data
rcParams["figure.figsize"] = (4, 4)
rna = anndata.read_h5ad("../data/matched_mouse_brain/rna_preprocessed.h5ad")
atac = anndata.read_h5ad("../data/matched_mouse_brain/atac_preprocessed.h5ad")


# Configure data
sccross.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer = 'counts',
     use_rep="X_pca"
)

sccross.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)

# MNN prior

sccross.data.mnn_prior([rna,atac])


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

a1,b1 = sccross.metrics.foscttm(rna.obsm['X_cross'],rna.obsm['X_cross_atac'])

for i in [250,500,1000,2000,4000]:
    if len(a1)>i:
        foscttm = (a1[0:i-1].mean()+b1[0:i-1].mean())/2
        print('FOSCTTM'+ str(i)+': '+ str(foscttm))








