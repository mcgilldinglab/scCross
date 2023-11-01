import anndata
import scanpy as sc
import sccross
import pandas as pd
from matplotlib import rcParams


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

