import anndata
import scanpy as sc
import sccross
import pandas as pd
import numpy as np
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics.pairwise import cosine_distances
import gc

# Read data
rcParams["figure.figsize"] = (4, 4)
rna = anndata.read_h5ad("../data/COVID-19/rna_preprocessed.h5ad")
adt = anndata.read_h5ad("../data/COVID-19/adt_preprocessed.h5ad")


# Configure data
sccross.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer = 'counts',
     use_rep="X_pca"
)

sccross.models.configure_dataset(
    adt, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)

# MNN prior
sccross.data.mnn_prior([rna,adt])
rna_mnn = rna.obsm['X_pca'].iloc[:,-50:]

# Training
cross = sccross.models.fit_SCCROSS(
    {"rna": rna, "adt": adt},
    fit_kws={"directory": "sccross"}
)


# Save model
cross.save("cross.dill")
#cross = sccross.models.load_model("cross.dill")



rna.obsm["X_cross"] = cross.encode_data("rna", rna)
adt.obsm["X_cross"] = cross.encode_data("adt", adt)


# Perturbation
rna.X = rna.layers['counts']
genes = rna.var.query("highly_variable").index.to_numpy().tolist()
difGenes = cross.perturbation_difGenes('rna',rna,'Status','Covid','Healthy',genes)

gene_up = difGenes['up'][difGenes['up']>0]
gene_down = difGenes['down'][difGenes['down']>0]
rna[rna.obs['Status'].isin(['Healthy']),gene_down].X += 0.5*rna[rna.obs['Status'].isin(['Healthy']),gene_down].X
rna[rna.obs['Status'].isin(['Healthy']),gene_up].X -= 0.5*rna[rna.obs['Status'].isin(['Healthy']),gene_up].X
rnaCroadt = cross.generate_cross( 'rna', 'adt', rna, adt)
rnaCroadt = sc.AnnData(rnaCroadt,obs=rna.obs,var= adt.var.query("highly_variable"))
print(rnaCroadt.X)


rna = anndata.read_h5ad("../data/COVID-19/rna_preprocessed.h5ad")
rna_temp = rna_mnn
rna_i = rna.copy()
rna_i.X = rna_i.layers['raw']
sc.pp.highly_variable_genes(rna_i, n_top_genes=100, flavor="seurat_v3")
hl = rna_i.var.index[rna_i.var['highly_variable']]
del rna_i
gc.collect()

rna.obsm['X_pca'] = np.concatenate((rna.obsm['X_pca'],rna_temp.obsm['X_pca']),axis=1)

rna.obs['domain'] = 'scRNA-seq'
rna.obs['cell_type'] = rna.obs['initial_clustering']
rna_t = rna_temp.copy()
rna_k = rna.copy()
rna = []

for j in list(rna_k.obs['initial_clustering'].cat.categories):
    rna = rna_k[rna_k.obs['initial_clustering'].isin([j])]
    rna_temp = rna_t[rna_t.obs['initial_clustering'].isin([j])]

    rna.obsm["X_cross"] = cross.encode_data("rna", rna)

    if len(rna[rna.obs['Status'] == "Covid"].obsm["X_cross"]) == 0 or len(rna[rna.obs['Status'] == "Healthy"].obsm["X_cross"]) == 0 or len(rna.obsm["X_cross"])<100:
        del rna
        gc.collect()
        continue

    cos_o = cosine_distances(rna[rna.obs['Status'] == "Covid"].obsm["X_cross"],rna[rna.obs['Status'] == "Healthy"].obsm["X_cross"])

    cos_o = cos_o.mean()

    data = []


    for i in range(5):
      for gene in hl[20*i:20*(i+1)]:
        temp = []
        temp.append(gene)
        rna_u = rna.copy()
        rna_u.X = rna.layers["raw"]
        rna_u.X = np.array(rna_u.X.todense())
        rna_u[:, gene].X += 1
        sc.pp.normalize_total(rna_u)
        sc.pp.log1p(rna_u)
        sc.pp.scale(rna_u)
        sc.tl.pca(rna_u, n_comps=100)
        rna_u.obsm['X_pca'] = np.concatenate((rna_u.obsm['X_pca'], rna_temp.obsm['X_pca']), axis=1)


        rna_u.obsm["X_cross"] = cross.encode_data("rna", rna_u)

        cos_u = cosine_distances(
            rna_u[rna_u.obs['Status'] == "Covid"].obsm["X_cross"],
            rna[rna_u.obs['Status'] == "Healthy"].obsm["X_cross"],
        )

        temp.append(cos_o-cos_u.mean())
        del rna_u
        gc.collect()

        rna_d = rna.copy()
        rna_d.X = rna.layers["raw"]
        rna_d.X = np.array(rna_d.X.todense())
        rna_d[:, gene].X -= 1
        rna_d.X[np.where(rna_d.X<0.0)] = 0
        sc.pp.normalize_total(rna_d)
        sc.pp.log1p(rna_d)
        sc.pp.scale(rna_d)
        sc.tl.pca(rna_d, n_comps=100)
        rna_d.obsm['X_pca'] = np.concatenate((rna_d.obsm['X_pca'], rna_temp.obsm['X_pca']), axis=1)


        rna_d.obsm["X_cross"] = cross.encode_data("rna", rna_d)
        cos_d = cosine_distances(
            rna_d[rna_d.obs['Status'] == "Covid"].obsm["X_cross"],
            rna[rna_d.obs['Status'] == "Healthy"].obsm["X_cross"],
        )
        temp.append(cos_o-cos_d.mean())
        data.append(temp)
        del rna_d
        gc.collect()

    df = pd.DataFrame(data,columns=['gene','up','down'])
    df.to_csv('up_down_'+j+'.csv')


