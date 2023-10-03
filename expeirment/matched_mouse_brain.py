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
    atac, "NB", use_highly_variable=False,
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



datalist = {'rna':rna,'atac':atac}

# Cross generation
for key1,data1 in datalist.items():
    for key2, data2 in datalist.items():
        if key1 != key2:
            cross_ge = cross.generate_cross( key1, key2, data1, data2)
            cross_ge = sc.AnnData(cross_ge,obs=data1.obs,var= data2.var.query("highly_variable"))

            sc.pp.normalize_total(cross_ge)
            sc.pp.log1p(cross_ge)
            sc.pp.scale(cross_ge)
            sc.tl.pca(cross_ge, n_comps=100, svd_solver="auto")
            sc.pp.neighbors(cross_ge,  metric="cosine")
            sc.tl.umap(cross_ge)
            sc.pl.umap(cross_ge, color=["cell_type"],save=key1+'_to_'+key2+'.pdf')


# Data enhancing
for key, data in datalist.items():
    data.obsm['enhanced'] = cross.generate_enhance(key, data)

    data_enhanced = sc.AnnData(data.obsm['enhanced'],obs=data.obs,var = data.var.query("highly_variable"))
    sc.pp.normalize_total(data_enhanced)
    sc.pp.log1p(data_enhanced)
    sc.pp.scale(data_enhanced)
    sc.tl.pca(data_enhanced, n_comps=100, svd_solver="auto")
    sc.pp.neighbors(data_enhanced, metric="cosine")
    sc.tl.umap(data_enhanced)
    sc.pl.umap(data_enhanced, color=["cell_type"], save=key + '_enhance' + '.pdf')
    sc.tl.rank_genes_groups(data_enhanced,'cell_type')
    df = pd.DataFrame(data_enhanced.uns['rank_genes_groups']['names'])
    df.to_csv(key+'_enhanced_rankGenes_cellType.csv')


# Multi-omics data simulation

fold = [0.5,1,5,10]
cell_type = list(set(rna.obs['cell_type']) & set(atac.obs['cell_type']))
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








