
# Setup env
```
conda create -n eva-rna python==3.10
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
To run the zero-shot in-sillico perturbation pipeline, run the command below:
``` 
MODE='layer_selective'
N_TOP_GENES=100
python perturbation --mode $MODE$ --n_top_genes $N_TOP_GENES --all_axis True
```
All data and scores from this pipeline is saved at `./data/perturbation_scores/${N_TOP_GENES}_top_genes/${MODE}`

# Evaluate pipeline scores
To evaluate the scores obtained from the perturbation pipeline, run the command below:

```python evaluate.py --results ./data/perturbation_scores/${N_TOP_GENES}_top_genes/${MODE}/perturbation_results.csv
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