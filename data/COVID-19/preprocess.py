
import scanpy as sc
import sccross


combined = sc.read_h5ad('./haniffa21.processed.h5ad')
rna = combined[:,combined.var['feature_type'].isin('Gene Expression')]
adt = combined[:,combined.var['feature_type'].isin('Antibody Capture')]

rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")

sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.pp.neighbors(rna, metric="cosine")
sc.tl.umap(rna)

sc.pp.normalize_total(adt)
sc.pp.log1p(adt)
sc.pp.scale(adt)
sc.tl.pca(adt, n_comps=100, svd_solver="auto")
sc.pp.neighbors(adt, metric="cosine")
sc.tl.umap(adt)



rna.write("rna_preprocessed.h5ad", compression="gzip")
adt.write("atac_preprocessed.h5ad", compression="gzip")

