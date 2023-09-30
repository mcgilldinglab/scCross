#!/bin/bash

set -e

gunzip GSE156793_S3_gene_count.loom.gz  # Produces: GSE156793_S3_gene_count.loom

tar xf GSE149683_RAW.tar  # Produces: *.seurat.RDS.gz
gunzip *.seurat.RDS.gz  # Produces: *.seurat.RDS

for item in *.seurat.RDS; do
    echo "Dealing with ${item}..."
    Rscript rds2pkl.R -i "${item}" -o "${item%\.seurat\.RDS}.pkl.gz"
done  # Produces: *.pkl.gz

