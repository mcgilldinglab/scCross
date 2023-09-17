# scCross
A Deep Learning-Based Model for integration, cross-dataset cross-modality generation and matched muti-omics simulation of single-cell multi-omics data. Our model can also maintain in-silico perturbations in cross-modality generation and can use in-silico perturbations to find key genes.

## Overview
<img title="Model Overview" alt="Alt text" src="/figures/main.png">
Single-cell multi-omics provides deep biological insights, but data scarcity and modality integration remain significant challenges. We introduce scCross, harnessing variational autoencoder and generative adversarial network (VAE-GAN) principles, meticulously designed to integrate diverse single-cell multi-omics data. Incorporating biological priors, scCross adeptly aligns modalities with enhanced relevance. Its standout feature is generating cross-modality single-cell data and in-silico perturbations, enabling deeper cellular state examinations and drug explorations. Applied to dual and triple-omics datasets, scCross maps data into a unified latent space, surpassing existing methods. By addressing data limitations and offering novel biological insights, scCross promises to advance single-cell research and therapeutic discovery.

## Key Capabilities

1. Integrate more than three matched or unmatchd sinlge cell multi-omics datasets of totally different or partly same kinds of omics into one latent space which can be used in following downsteam analysis. The cell amount can be over 4 million.
2. Cross generate abitary two kinds of single cell data in training set to each other.
3. Simulate matched single cell muti-omics data of a special cellular status in arbitary kind of omics and arbitary amount.
4. Find key genes precisely in comparing two kinds of cell clusters via in-silico pertubation.
5. Effectively maintain genome changing when perturbating one omics and cross generating to to other omics.


## Installation


### Installing CellAgentChat

You may install scCross and its dependencies by the following command:

```
pip3 install git+https://github.com/mcgilldinglab/scCross
```

## Tutorial:
### Data preprocessing
* [Data preprocessing](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/data_preprocessing/preprocessing.ipynb)

### Training scCross on matched mouse brain dataset
* [Matched mouse brain dataset](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/benchmark/training_matched_mouse_brain.ipynb)
### Training scCross on unmatched mouse brain dataset
* [Unmatched mouse brain dataset](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/benchmark/training_unmatched_mouse_brain.ipynb)
### Training scCross on matched mouse blood dataset
* [Matched mouse blood dataset](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/benchmark/training_matched_mouse_blood.ipynb)
### Training scCross on matched mouse lymph node dataset
* [Matched mouse lymph node dataset](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/benchmark/training_matched_mouse_lymnode.ipynb)
### Training scCross on human cell atlas dataset
* [Human cell atlas dataset](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/training_human_cell_atlas.ipynb)
### Training scCross on COVID-19 dataset
* [COVID-19 dataset](https://github.com/mcgilldinglab/scCross/blob/main/tutorial/training_COVID-19.ipynb)