# Zero-shot in-silico perturbation
 This repo is my implementation of the ZS in-silico perturbation pipeline for predicting drug-disease interaction. It is built for the open source **EVA-RNA (60M)** model and transcriptomics data.

## Overview
This project explores whether a pre-trained transcriptomics foundation model can predict drug efficacy zero-shot, without any perturbation-labeled training data. Given a disease bulk RNA-seq cohort and a drug's target gene(s), we use EVA-RNA to simulate, in silico, the transcriptomic effect of knocking down those targets. A drug is predicted to be efficacious if its simulated perturbation moves the disease expression profile toward a healthy state. We evaluate this on 86 drug-disease pairs spanning 5 autoimmune and inflammatory diseases, achieving a **global AUROC of 0.6525**.

# Setup env
Follow these instructions to create the environment for running the code from this repo.
```
conda create -n eva-rna python==3.10
conda activate eva-rna
pip install transformers huggingface_hub torch==2.6.0 scanpy anndata tqdm scipy scikit-misc
```

## Install eva-rna 
This step is necessary using utils functions and monkey patching `EvaRnaModel` for layer selective perturbation.
Be sure to login to HuggingFace via `hf auth login`.
```git clone https://huggingface.co/ScientaLab/eva-rna eva_rna
touch eva_rna/__init__.py
```

## Download benchmark 
This can be done either via the Notion page's link or via command line from drive. 
```mkdir data 
wget -O data/benchmark_drug_target_disease_matrix.csv https://drive.usercontent.google.com/u/0/uc?id=1rPDrinSIDbpyK65WC_kWFGuR2DAjINSJ&export=download
```

# Run perturbation pipeline
To run the zero-shot in-silico perturbation pipeline, run the command below:
``` 
MODE='layer_selective'
N_TOP_GENES=100
python perturbation --mode $MODE$ --n_top_genes $N_TOP_GENES --all_axis True
```
All data and scores from this pipeline is saved at `./data/perturbation_scores/${N_TOP_GENES}_top_genes/${MODE}`

# Evaluate pipeline scores
To evaluate the scores obtained from the perturbation pipeline, run the command below:
```
python evaluate.py --results ./data/perturbation_scores/${N_TOP_GENES}_top_genes/${MODE}/perturbation_results.csv
```
Discounting logs, it should output results such as this:
```
====================================================
  EVA-RNA Perturbation Benchmark — Evaluation
====================================================
  Pairs evaluated : 84  (38 effective / 46 ineffective)
  NaN handling    : dropped
  Global AUROC    : 0.6525

  Per-disease breakdown:
    AD        AUROC = 0.5333  (n=22)
    CD        AUROC = 0.6667  (n=16)
    PSO       AUROC = 0.6825  (n=16)
    T1D       AUROC = 0.8854  (n=14)
    UC        AUROC = 0.6508  (n=16)
====================================================
```

# Pipeline Architecture

- **Cohort loading & HVG filtering**: For each disease, bulk RNA-seq data is loaded from HuggingFace and filtered to the top N highly variable genes, with *drug target genes force-included* to avoid dropping therapeutically relevant signal.
- **Tokenisation**: Gene Entrez IDs are mapped to EVA-RNA's vocabulary, producing a shared token ID tensor for the cohort (mirroring ``utils.encode_from_anndata``)
- **Encoding**: EVA-RNA encodes all healthy and disease samples. Depending on the mode, this yields either final-layer gene embeddings (``latent_space``) or intermediate hidden states at layer n (``layer_selective``).
- **Disease-to-healthy axis**: A displacement axis is computed as ``mean(z_healthy) − mean(z_disease)`` in the chosen latent space, following Bjerregaard et al. eq. 3.
- **Per-sample perturbation**: For each disease sample, a knockdown loss is backpropagated through the decoder to obtain ``∇_z L``. The gradient is L2-normalised and added to the latent representation (EVA report, eq. 19).
- **Scoring**: The cosine similarity between each sample's perturbation gradient and the disease-to-healthy axis (Bjerregaard et al. eq. 5) gives a per-patient score. The mean across patients is the drug-disease scalar used for AUROC evaluation.

# Key design choices 

- **Latent-space scoring instead of decoded-space.** An earlier formulation scored perturbations by Pearson correlation improvement in decoded expression space. This was abandoned because EVA-RNA's decoder compresses diverse inputs toward a similar manifold, making deltas in expression space noisy and sign-unreliable. Scoring entirely in latent space, by comparing the gradient direction to the disease-to-healthy displacement axis, avoids decoding and keeps the signal in a space where the geometry is meaningful.
- **Layer-selective perturbation**. Rather than perturbing only the final gene embeddings, layer_selective mode attaches the gradient leaf at an intermediate transformer layer (default: layer 23 of 24). This allows gradient signal to propagate through the upper layers' self-attention, capturing how perturbation of a target gene cascades across the transcriptome. It mirrors how the model itself contextualises expression.
- **Full-gradient vs. target-gene cosine scoring.** Two scoring modes are supported: full-gradient (cosine over all S×H dimensions) and target-gene-only (cosine restricted to the hidden states at the target gene sequence positions). The full-gradient mode is the default, since attention propagation means all gene positions carry informative gradient signal even for a single-target drug.
- **Gene alias resolution.** The benchmark mixes HGNC symbols, CD antigen names, and protein names. A custom gene_alias_map.py resolves these to the HGNC symbols present in EVA-RNA's vocabulary. Target genes absent from the HVG-filtered set are force-included during cohort loading to avoid silently dropping therapeutically relevant targets.