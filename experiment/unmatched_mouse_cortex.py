import anndata
import scanpy as sc
import sccross
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

# Read data
rcParams["figure.figsize"] = (4, 4)
rna = anndata.read_h5ad("../data/unmatched_mouse_cortex/rna_preprocessed.h5ad")
atac = anndata.read_h5ad("../data/unmatched_mouse_cortex/atac_preprocessed.h5ad")
met = anndata.read_h5ad("../data/unmatched_mouse_cortex/snm_preprocessed.h5ad")

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

sccross.models.configure_dataset(
    met, "NB", use_highly_variable=False,
    use_rep="X_pca"
)

# MNN prior
sccross.data.mnn_prior([rna,atac,met])


# Training
cross = sccross.models.fit_SCCROSS(
    {"rna": rna, "atac": atac,'snm':met},
    fit_kws={"directory": "sccross"}
)


# Save model
cross.save("cross.dill")
#cross = sccross.models.load_model("cross.dill")


# Integration
rna.obsm["X_cross"] = cross.encode_data("rna", rna)
atac.obsm["X_cross"] = cross.encode_data("atac", atac)
met.obsm["X_cross"] = cross.encode_data("snm", met)

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

ARI = adjusted_rand_score(met.obs['cell_type'], met.obs['leiden'])
NMI = normalized_mutual_info_score(met.obs['cell_type'],met.obs['leiden'])
print("snmC:ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))

ASW = sccross.metrics.avg_silhouette_width(combined.obsm['X_cross'],combined.obs['cell_type'])
ASWb = sccross.metrics.avg_silhouette_width_batch(combined.obsm['X_cross'],combined.obs['domain'],combined.obs['cell_type'])
GCT = sccross.metrics.graph_connectivity(combined.obsm['X_cross'],combined.obs['cell_type'])
print("ASW: "+str(ASW)+"ASWb: "+str(ASWb)+"GCT: "+str(GCT))


datalist = {'rna':rna,'atac':atac,'met':met}

# Cross generation
for key1,data1 in datalist.items():
    for key2, data2 in datalist.items():
        if key1 != key2:
            cross_ge = cross.generate_cross( key1, key2, data1, data2)
            cross_ge = sc.AnnData(cross_ge, obs=data1.obs, var=data2.var.query("highly_variable"))
            if key2 == 'atac':
                sccross.data.lsi(cross_ge, n_components=100, n_iter=15)
                sc.pp.neighbors(cross_ge, use_rep='X_lsi', metric="cosine")
                sc.tl.umap(cross_ge)
                sc.pl.umap(cross_ge, color=["cell_type"], save=key1 + '_to_' + key2 + '.pdf')
            else:
                sc.pp.normalize_total(cross_ge)
                sc.pp.log1p(cross_ge)
                sc.pp.scale(cross_ge)
                sc.tl.pca(cross_ge, n_comps=100, svd_solver="auto")
                sc.pp.neighbors(cross_ge, metric="cosine")
                sc.tl.umap(cross_ge)
                sc.pl.umap(cross_ge, color=["cell_type"], save=key1 + '_to_' + key2 + '.pdf')


# Data augmentation
for key, data in datalist.items():
    data.obsm['augmented'] = cross.generate_augment(key, data)

    data_augmented = sc.AnnData(data.obsm['augmented'],obs=data.obs,var = data.var.query("highly_variable"))
    if key == 'atac':
        sccross.data.lsi(data_augmented, n_components=100, n_iter=15)
        sc.pp.neighbors(data_augmented, use_rep='X_lsi', metric="cosine")
        sc.tl.umap(data_augmented)
        sc.pl.umap(data_augmented, color=["cell_type"], save=key + '_augment' + '.pdf')

    else:
        sc.pp.normalize_total(data_augmented)
        sc.pp.log1p(data_augmented)
        sc.pp.scale(data_augmented)
        sc.tl.pca(data_augmented, n_comps=100, svd_solver="auto")
        sc.pp.neighbors(data_augmented, metric="cosine")
        sc.tl.umap(data_augmented)
        sc.pl.umap(data_augmented, color=["cell_type"], save=key + '_augment' + '.pdf')
        sc.tl.rank_genes_groups(data_augmented, 'cell_type')
        df = pd.DataFrame(data_augmented.uns['rank_genes_groups']['names'])
        df.to_csv(key + '_augmented_rankGenes_cellType.csv')


# Multi-omics data simulation

fold = [0.5,1,5,10]
cell_type = list(set(rna.obs['cell_type']) & set(atac.obs['cell_type']) & set(met.obs['cell_type']))
for i in fold:
    for j in cell_type:
        multi_simu = cross.generate_multiSim(datalist,'cell_type',j, int(i*len(rna[rna.obs['cell_type'].isin([j])])))
        for adata in multi_simu:
            adata.obs['cell_type'] = j + '_s'

        rna_temp = rna.copy()
        rna_temp.X = rna_temp.layers['counts']
        rna_temp = rna_temp[:, rna_temp.var.query("highly_variable").index]
        rna_temp = sc.concat([rna_temp, multi_simu[0]])

        atac_temp = atac.copy()
        atac_temp = atac_temp[:, atac_temp.var.query("highly_variable").index]
        atac_temp = sc.concat([atac_temp, multi_simu[1]])

        met_temp = met.copy()
        met_temp = met_temp[:, met_temp.var.query("highly_variable").index]
        met_temp = sc.concat([met_temp, multi_simu[2]])

        sc.pp.normalize_total(rna_temp)
        sc.pp.log1p(rna_temp)
        sc.pp.scale(rna_temp)
        sc.tl.pca(rna_temp, n_comps=100, svd_solver="auto")
        sc.pp.neighbors(rna_temp, metric="cosine")
        sc.tl.umap(rna_temp)
        sc.pl.umap(rna_temp, color=["cell_type"], save='RNA' + j + '_' + str(i) + '.pdf')

        sccross.data.lsi(atac_temp, n_components=100, n_iter=15)
        sc.pp.neighbors(atac_temp, use_rep='X_lsi', metric="cosine")
        sc.tl.umap(atac_temp)
        sc.pl.umap(atac_temp, color=["cell_type"], save='ATAC' + j + '_' + str(i) + '.pdf')

        sc.pp.normalize_total(met_temp)
        sc.pp.log1p(met_temp)
        sc.pp.scale(met_temp)
        sc.tl.pca(met_temp, n_comps=100, svd_solver="auto")
        sc.pp.neighbors(met_temp, metric="cosine")
        sc.tl.umap(met_temp)
        sc.pl.umap(met_temp, color=["cell_type"], save='met' + j + '_' + str(i) + '.pdf')








