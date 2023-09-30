import anndata
import scanpy as sc
import sccross


rna = anndata.read_mtx('./lymph_RNA')
atac = anndata.read_mtx('./lymph_ATAC')
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

rna.obs['cell_type'] = rna.obs['labels']
atac.obs['cell_type'] = rna.obs['labels']
rna.write("rna_preprocessed.h5ad", compression="gzip")
atac.write("atac_preprocessed.h5ad", compression="gzip")
atac2rna.write("atac2rna.h5ad", compression="gzip")