# scCross
A Deep Learning-Based Model for the integration, cross-dataset cross-modality generation, self enhancing and matched multi-omics simulation of single-cell multi-omics data. Our model excels at maintaining in-silico perturbations during cross-modality generation and harnessing these perturbations to identify key genes.

For detailed instructions, comprehensive documentation, and helpful tutorials, please visit:
  
* [https://sccross.readthedocs.io](https://sccross.readthedocs.io/en/latest/)


## Overview
<img title="Model Overview" alt="Alt text" src="/figures/main.png">
Single-cell multi-omics provides deep biological insights, but data scarcity and modality integration remain significant challenges. We introduce scCross, harnessing variational autoencoder and generative adversarial network (VAE-GAN) principles, meticulously designed to integrate diverse single-cell multi-omics data. Incorporating biological priors, scCross adeptly aligns modalities with enhanced relevance. Its standout feature is generating cross-modality single-cell data and in-silico perturbations, enabling deeper cellular state examinations and drug explorations. Applied to dual and triple-omics datasets, scCross maps data into a unified latent space, surpassing existing methods. By addressing data limitations and offering novel biological insights, scCross promises to advance single-cell research and therapeutic discovery.

## Key Capabilities

1. Combine more than three single-cell multi-omics datasets, whether they are matched or unmatched, into a unified latent space. This space can be used for downstream analysis, even when dealing with over 4 million cells of varying types.

2. Generate cross-compatible single-cell data between two or more different omics even cross dataset.

3. Augment single-cell omics data through self-improvement techniques.

4. Simulate single-cell multi-omics data that match a specific cellular state, irrespective of the type and quantity of omics data involved.

5. Accurately identify key genes by comparing two different cell clusters using in-silico perturbation methods.

6. Maintain genomic integrity during omics perturbations and cross-generations effectively.







## Installation


You may install scCross by the following command:

```
pip install sccross
```

