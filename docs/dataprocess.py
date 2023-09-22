import anndata
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams

rna = anndata.read_h5ad("../Chen-2019-RNA.h5ad")
atac = anndata.read_h5ad("../Chen-2019-ATAC.h5ad")
rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")

scglue.data.lsi(atac, n_components=100, n_iter=15)

scglue.data.get_gene_annotation(
    rna, gtf="gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    gtf_by="gene_name"
)
rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1])
atac.var["chromEnd"] = split.map(lambda x: x[2])
atac.var.head()
graph = scglue.genomics.rna_anchored_prior_graph(rna, atac)
rna.write("rna_preprocessed.h5ad", compression="gzip")
atac.write("atac_preprocessed.h5ad", compression="gzip")
nx.write_graphml(graph, "prior.graphml.gz")