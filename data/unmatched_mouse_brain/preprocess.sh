#!/bin/bash

set -e

Rscript export_data.R  # Produces: F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx, F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.rownames, F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.colnames, F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.cell_cluster_outcomes.csv
gzip F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx  # Produces: gzip F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx.gz


Rscript signac.r  # Produces: signac_idents.csv, signac_meta_data.csv


tar xf snmcSeq_processed_data.tar.gz

