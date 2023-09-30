import anndata
import networkx as nx
import scanpy as sc
import sccross
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

rcParams["figure.figsize"] = (4, 4)
rna = anndata.read_h5ad("../data/matched_mouse_brain/rna_preprocessed.h5ad")
atac = anndata.read_h5ad("../data/matched_mouse_brain/atac_preprocessed.h5ad")

sccross.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer = 'counts',
     use_rep="X_pca"
)

sccross.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)

sccross.data.mnn_prior([rna,atac])

cross = sccross.models.fit_SCCROSS(
    {"rna": rna, "atac": atac},
    fit_kws={"directory": "sccross"}
)

cross.save("cross.dill")
#cross = sccross.models.load_model("cross.dill")


rna.obsm["X_cross"] = cross.encode_data("rna", rna)
atac.obsm["X_cross"] = cross.encode_data("atac", atac)


combined = anndata.concat([rna, atac])

sc.pp.neighbors(combined, use_rep="X_cross", metric="cosine")
sc.tl.umap(combined)
sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65)

atacCrorna = cross.generate_cross( 'atac', 'rna', atac, rna)
atacCrorna = sc.AnnData(atacCrorna,obs=atac.obs,var= rna.var.query("highly_variable"))

sc.pp.normalize_total(atacCrorna)
sc.pp.log1p(atacCrorna)
sc.pp.scale(atacCrorna)
sc.tl.pca(atacCrorna, n_comps=100, svd_solver="auto")
sc.pp.neighbors(atacCrorna,  metric="cosine")
sc.tl.umap(atacCrorna)
sc.pl.umap(atacCrorna, color=["cell_type"])

rna.obsm['enhanced'] = cross.generate_enhance(  'rna', rna)

rna_enhanced = sc.AnnData(rna.obsm['enhanced'],obs=rna.obs,var = rna.var.query("highly_variable"))
sc.pp.normalize_total(rna_enhanced)
sc.pp.log1p(rna_enhanced)
sc.pp.scale(rna_enhanced)
sc.tl.rank_genes_groups(rna_enhanced,'cell_type')
df = pd.DataFrame(rna_enhanced.uns['rank_genes_groups']['names'])
df.to_csv('rna_enhanced_rankGenes_cellType.csv')


multi_simu = cross.generate_multiSim({'rna':rna,'atac':atac},'cell_type','Ast',len(rna[rna.obs['cell_type'].isin(['Ast'])]))
for adata in multi_simu:
    adata.obs['cell_type'] = 'Ast_s'

rna_temp = rna.copy()
rna_temp.X = rna_temp.layers['counts']
rna_temp = rna_temp[:,rna_temp.var.query("highly_variable").index]
rna_temp = sc.concat([rna_temp,multi_simu[0]])

atac_temp = atac.copy()
atac_temp = atac_temp[:,atac_temp.var.query("highly_variable").index]
atac_temp = sc.concat([atac_temp,multi_simu[1]])

sc.pp.normalize_total(rna_temp)
sc.pp.log1p(rna_temp)
sc.pp.scale(rna_temp)
sc.tl.pca(rna_temp, n_comps=100, svd_solver="auto")
sc.pp.neighbors(rna_temp,  metric="cosine")
sc.tl.umap(rna_temp)
sc.pl.umap(rna_temp, color=["cell_type"])

sccross.data.lsi(atac_temp, n_components=100, n_iter=15)
sc.pp.neighbors(atac_temp, use_rep = 'X_lsi',  metric="cosine")
sc.tl.umap(atac_temp)
sc.pl.umap(atac_temp, color=["cell_type"])

rna.X = rna.layers['counts']
genes = ['Slc1a2','Gpc5','Lsamp','Csmd1','Snhg11','Meg3']
difGenes = cross.perturbation_difGenes('rna',rna,'cell_type','In','Ast',genes)

gene_up = ['Slc1a2','Gpc5','Lsamp']
rna[rna.obs['cell_type'].isin(['In']),genes] += 0.5*rna[rna.obs['cell_type'].isin(['In']),genes]
rnaCroatac = cross.generate_cross( 'rna', 'atac', rna, atac)
rnaCroatac = sc.AnnData(rnaCroatac,obs=rna.obs,var= atac.var.query("highly_variable"))

sccross.data.lsi(rnaCroatac, n_components=100, n_iter=15)
sc.pp.neighbors(rnaCroatac, use_rep = 'X_lsi',  metric="cosine")
sc.tl.umap(rnaCroatac)
sc.pl.umap(rnaCroatac, color=["cell_type"])

sc.pl.umap(atac, color=["cell_type"])





