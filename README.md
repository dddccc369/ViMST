# SpaMask: Dual Masking Graph Autoencoder with Contrastive Learning for Spatial Transcriptomics 
## Introduction
Investigating spatial transcriptomics microenvironments is crucial for unraveling cellular heterogeneity. Existing methods struggle to extract non-redundant information from histopathological images simultaneously and spatially resolved gene expression profiles. We propose a vision transformer-based dual-modality multi-task graph contrastive network for exploring the spatial transcriptomics domain (ViMST), which integrates gene expression, image features, and spatial coordinates for tissue microenvironments investigation. It employs ViT for feature extraction and dual masked GCNs to model modalities separately. A novel joint topology decoder learns morphology-expression spatial covariation, enhancing relationship modeling across multiple tasks. The evaluation results across nine spatial transcriptomics datasets reveal that ViMST consistently performs better in spatial domain identification and data denoising than eight state-of-the-art methods. It demonstrates robust performance in multiple tissue microenvironments research tasks, including data visualization, trajectory inference, spatially variable genes (SVGs) identification, horizontal integration analysis, cellular heterogeneity analysis, and epithelial-mesenchymal transition (EMT) studies. These results confirm ViMST's strong generalization capability and practical utility for spatial transcriptomics research. 

![Alt text](%EF%BF%BD%EF%BF%BD%CD%BC111.jpg)

## Data
•	10x Visium human dorsolateral prefrontal cortex dataset: http://spatial.libd.org/spatialLIBD/;
•	10x Visium human breast cancer dataset, 10x Visium human lymph node dataset, 10x Visium anterior and posterior mouse brain dataset, 10x Visium mouse kidney dataset, and 10x Visium mouse brain dataset: https://www.10xgenomics.com/datasets/; 
•	10x Visium human liver cancer dataset:  https://db.cngb.org/stomics/datasets/STDS0000219/summary/
•	10x Visium human pancreatic cancer dataset: https://data.humantumoratlas.org/;
•	Stereo-seq mouse olfactory bulb dataset: https://github.com/JinmiaoChenLab/SEDR_analyses;

## Setup
•   `pip install -r requirement.txt`

## Get Started
•    Please see `Tutorial.ipynb`.


## Citing


## Article link


