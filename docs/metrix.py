import anndata
import itertools
import networkx as nx
import pandas as pd
import scanpy as sc
import scglue
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import seaborn as sns
from matplotlib import rcParams
rna = anndata.read_h5ad("rna_preprocessed.h5ad")
atac = anndata.read_h5ad("atac_preprocessed.h5ad")
graph = nx.read_graphml("prior.graphml.gz")
scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)
graph = graph.subgraph(itertools.chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
))
glue = scglue.models.load_model("./glue/pretrain/pretrain.dill")
rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
sc.pp.neighbors(rna,use_rep='X_glue')
sc.tl.leiden(rna)
ARI = adjusted_rand_score(rna.obs['cell_type'], rna.obs['leiden'])
NMI = normalized_mutual_info_score(rna.obs['cell_type'],rna.obs['leiden'])
print("ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))

sc.pp.neighbors(atac,use_rep='X_glue')
sc.tl.leiden(atac)
ARI = adjusted_rand_score(atac.obs['cell_type'], atac.obs['leiden'])
NMI = normalized_mutual_info_score(atac.obs['cell_type'],atac.obs['leiden'])
print("ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))