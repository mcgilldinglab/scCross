
import scanpy as sc
import sccross


combined = sc.read_10x_mtx('./raw_feature_bc_matrix')
rna = combined[:,combined.var.iloc[:,2].isin('Gene Expression')]
rna_obs = sc.read_h5ad('merged.h5ad')
rna.obs = rna_obs.obs
atac = combined[:,combined.var.iloc[:,2].isin('Peak')]

rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")

sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.pp.neighbors(rna, metric="cosine")
sc.tl.umap(rna)

sccross.data.lsi(atac, n_components=100, n_iter=15)
sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)

atac2rna = sccross.data.geneActivity(atac)


rna.write("rna_preprocessed.h5ad", compression="gzip")
atac.write("atac_preprocessed.h5ad", compression="gzip")
atac2rna.write("atac2rna.h5ad", compression="gzip")
