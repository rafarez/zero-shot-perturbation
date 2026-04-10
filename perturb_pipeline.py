"""Zero-shot in-silico perturbation pipeline for EVA-RNA.

Three perturbation modes are supported, selected via ``--mode``:

    latent_space (default)
        Perturbs the final gene embeddings z (shape B, S, H) directly.
        Implements EVA report equations (19)–(20).
        Gradient ∇_z L is computed with grad enabled only for model.decode(z).

    layer_selective
        Perturbs the intermediate hidden states h_n (shape B, S+1, H) after
        layer n of the transformer stack, with grad enabled from layer n onward.
        Update rule mirrors latent_space: h'_n = h_n + ∇_{h_n} L (eq. 19/25).
        Scoring uses the disease-to-healthy axis computed at the same layer n,
        keeping the gradient and the reference axis in the same space.

Scoring (all modes)
-------------------
Each disease sample's perturbation gradient is compared to the latent
disease-to-healthy displacement axis via cosine similarity
(Bjerregaard et al. eq. 5).  For latent_space and layer_selective modes,
this axis lives in the corresponding hidden-state space.  
CLS token positions are excluded from the axis computation in all modes.

Usage
-----
    python perturb.py [--mode {latent_space,layer_selective}]
                      [--n_top_genes N_TOP_GENES]
                      [--grad_from_layer GRAD_FROM_LAYER]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
from scipy.sparse import issparse
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


#from eva_rna.utils import _normalize_and_log

from gene_alias_map import MissingTargetGenesList, GENE_ALIAS_MAP
from encode_and_save import load_cohort_data
from gradient_flow_pert_loss import perturbation_loss
from scoring_cosine import (
    compute_healthy_disease_axis,
    compute_shift_score,
)

# ---------------------------------------------------------------------------
# CLI / configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot in-silico perturbation pipeline for EVA-RNA."
    )
    parser.add_argument(
        "--mode",
        choices=["latent_space", "layer_selective"],
        default="latent_space",
        help=(
            "Perturbation mode. "
            "'latent_space': perturb final gene embeddings (default). "
            "'layer_selective': perturb intermediate hidden states at layer n."
        ),
    )
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=100,
        help="Number of highly variable genes to retain (default: 100).",
    )
    parser.add_argument(
        "--all_axis",
        action="store_true",
        help=(
            "Wether to use the full gradient vector for cosine scoring (all gene"
            "positions), or to restrict to target-gene positions only."
            ),
    )
    parser.add_argument(
        "--grad_from_layer",
        type=int,
        default=23,
        help=(
            "Transformer layer index (0-based) at which to attach the gradient "
            "leaf for layer_selective mode. Layers 0..n-1 run under no_grad; "
            "layers n..N-1 run with grad enabled. Ignored for other modes. "
            "Default: 22 (i.e. last 2 layers of the 24-layer 60M model)."
        ),
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="data/benchmark_drug_target_disease_matrix.csv",
        help="Benchmark path"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Hardcoded constants
# ---------------------------------------------------------------------------
BATCH_SIZE       = 16      # samples per forward pass (backward is always per-sample)
EPS              = 1e-8    # numerical stability for gradient L2 normalisation
PERTURBATION_DIR = -1      # δ = -1: knockdown for all targets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation helpers (mirroring utils.encode_from_anndata)
# ---------------------------------------------------------------------------

def prepare_tokenisation(
    adata: ad.AnnData,
    tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, list[int]]:
    """Filter genes to the model vocabulary and build the shared token ID tensor.

    Mirrors the vocab-filtering and tokenisation logic in
    ``utils.encode_from_anndata``:
      1. Extract gene IDs from ``adata.var`` (Entrez IDs as strings in index).
      2. Filter to genes present in the tokenizer vocabulary.
      3. Convert to integer token IDs via ``tokenizer.convert_tokens_to_ids``.
      4. Preprocess and filter the expression matrix to the same gene subset.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData cohort (HVG-filtered).  Gene Entrez IDs are in ``adata.var_names``.
    tokenizer : EvaRnaTokenizer
        EVA-RNA tokenizer.
    device : torch.device
        Target device for the token ID tensor.

    Returns
    -------
    X_filtered : np.ndarray
        Log-normalised expression matrix filtered to vocab genes,
        shape ``(n_samples, n_vocab_genes)``, dtype float32.
    token_ids_tensor : torch.Tensor
        Shared integer token IDs, shape ``(n_vocab_genes,)``.
        Expand to ``(batch, n_vocab_genes)`` before passing to the model.
    gene_indices : list[int]
        Column indices into the original ``adata.X`` that were kept.
    """
    gene_ids = adata.var_names.astype(str).tolist()

    gene_mask    = [tokenizer.gene_in_vocab(g) for g in gene_ids]
    gene_indices = [i for i, m in enumerate(gene_mask) if m]
    n_matched    = len(gene_indices)

    if n_matched == 0:
        raise ValueError(
            "No genes in adata match the model vocabulary.  Check that "
            "adata.var_names contains NCBI Entrez IDs (e.g. '7157' for TP53)."
        )

    filtered_gene_ids = [gene_ids[i] for i in gene_indices]
    token_ids = tokenizer.convert_tokens_to_ids(filtered_gene_ids)
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X[:, gene_indices]
    X = _normalize_and_log(X).astype(np.float32)

    return X, token_ids_tensor, gene_indices


def make_batch_tensors(
    X_filtered: np.ndarray,
    token_ids_tensor: torch.Tensor,
    sample_indices: list[int] | np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build gene_ids and expression_values tensors for a batch of samples.

    Parameters
    ----------
    X_filtered : np.ndarray
        Full log-normalised expression matrix, shape ``(n_samples, n_genes)``.
    token_ids_tensor : torch.Tensor
        Shared token IDs, shape ``(n_genes,)``.
    sample_indices : list[int] | np.ndarray
        Row indices into ``X_filtered`` for this batch.
    device : torch.device
        Target device.

    Returns
    -------
    gene_ids : torch.Tensor
        Shape ``(batch, n_genes)``.
    expression_values : torch.Tensor
        Shape ``(batch, n_genes)``.
    """
    B = len(sample_indices)
    batch_X = X_filtered[sample_indices]
    gene_ids = token_ids_tensor.unsqueeze(0).expand(B, -1)
    expression_values = torch.from_numpy(batch_X).to(device)
    return gene_ids, expression_values


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_samples(
    model,
    X_filtered: np.ndarray,
    token_ids_tensor: torch.Tensor,
    device: torch.device,
    desc: str = "Encoding",
    up_to_layer: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all samples and return gene embeddings.

    Runs in no-grad / inference mode (frozen weights).

    Parameters
    ----------
    model : EvaRnaModel
        Frozen EVA-RNA model.
    X_filtered : np.ndarray
        Log-normalised expression, shape ``(n_samples, n_genes)``, float32.
    token_ids_tensor : torch.Tensor
        Shared token IDs, shape ``(n_genes,)``.
    device : torch.device
        Computation device.
    desc : str
        Progress bar label.
    up_to_layer : int | None
        If given, runs only the first ``up_to_layer`` transformer layers and
        returns the intermediate hidden states (including CLS at position 0)
        as the first return value.  Used for layer_selective mode to cache h_n.
        The second return value (decoded expression) is set to an empty tensor
        in this case, as decoding from an intermediate layer is not meaningful.
        If None (default), runs the full encoder and decoder.

    Returns
    -------
    all_embeddings : torch.Tensor
        - Full mode (up_to_layer=None): gene-level hidden states after the
          final encoder layer, shape ``(n_samples, S, H)``, CLS stripped.
        - Layer-selective mode (up_to_layer=n): intermediate hidden states
          *including CLS at position 0*, shape ``(n_samples, S+1, H)``.
          Keeping CLS here allows ``encode_from_layer`` to resume the forward
          pass without any bookkeeping change; callers should strip it
          (``[:, 1:, :]``) before computing the disease-to-healthy axis or
          using as a grad leaf.
    """
    n_samples = len(X_filtered)
    all_emb     = []

    for start in tqdm(range(0, n_samples, BATCH_SIZE), desc=desc, leave=False):
        idx = list(range(start, min(start + BATCH_SIZE, n_samples)))
        gene_ids, expr_vals = make_batch_tensors(
            X_filtered, token_ids_tensor, idx, device
        )

        if up_to_layer is None:
            out     = model.encode(gene_ids, expr_vals)  # (B, S)
            all_emb.append(out.gene_embeddings.cpu())    # (B, S, H)
        else:
            # encode_up_to_layer returns (EvaRnaOutput, hidden_with_cls, pmask)
            # We cache hidden_with_cls (CLS still present) so the perturbation
            # function can use it directly as the detach-and-reattach leaf.
            _, hidden_with_cls, _ = model.encode_up_to_layer(
                gene_ids, expr_vals, n=up_to_layer
            )
            all_emb.append(hidden_with_cls.cpu())        # (B, S+1, H)

    embeddings = torch.cat(all_emb, dim=0)
    return embeddings


# ---------------------------------------------------------------------------
# Per-sample perturbation — latent space (equations 19-20)
# ---------------------------------------------------------------------------

def perturb_one_sample(
    model,
    gene_embeddings_1: torch.Tensor,
    gene_ids_1: torch.Tensor,
    target_gene_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply latent-space perturbation to a single expression profile.

    Implements EVA report equations (19)–(20):
        z'  = z + ∇_z L_pert    (∇_z L normalised to unit L2 norm, eq. 25)
        x'  = f_dec(z')

    The encoder forward pass is omitted: the caller passes in precomputed
    gene embeddings from ``encode_samples``, avoiding a redundant
    encoder pass inside the per-sample loop.

    Parameters
    ----------
    model : EvaRnaModel
        EVA-RNA model (weights frozen).
    gene_embeddings_1 : torch.Tensor
        Precomputed final-layer gene embeddings, shape ``(1, S, H)``, on CPU.
    gene_ids_1 : torch.Tensor
        Token IDs, shape ``(1, S)``, required by ``perturbation_loss``.
    target_gene_ids : list[int]
        Token IDs of target genes.
    device : torch.device

    Returns
    -------
    grad_z : torch.Tensor
        Raw (un-normalised) perturbation gradient, shape ``(1, S, H)``, CPU.
        Returned un-normalised for cosine scoring (cosine is scale-invariant).
    z_prime : torch.Tensor
        Perturbed embeddings, shape ``(1, S, H)``, CPU.
    x_prime : torch.Tensor
        Decoded expression from z', shape ``(1, S)``, CPU.
    """
    model.eval()

    z = gene_embeddings_1.to(device).clone().detach().requires_grad_(True)  # (1, S, H)

    with torch.enable_grad():
        pred_expr = model.decode(z)   # (1, S)

    directions = [PERTURBATION_DIR] * len(target_gene_ids)

    try:
        with torch.enable_grad():
            loss = perturbation_loss(
                predicted_expression=pred_expr,
                gene_ids=gene_ids_1,
                target_gene_ids=target_gene_ids,
                perturbation_directions=directions,
            )
            loss.mean().backward()
    except KeyError as exc:
        log.warning("Skipping perturbation (target gene absent): %s", exc)
        with torch.no_grad():
            x_orig = model.decode(z.detach())
        return torch.zeros_like(z.detach()).cpu(), z.detach().cpu(), x_orig.detach().cpu()

    grad_z = z.grad  # (1, S, H)

    if grad_z is None:
        log.warning("∇_z is None — returning original embedding.")
        with torch.no_grad():
            x_orig = model.decode(z.detach())
        return torch.zeros_like(z.detach()).cpu(), z.detach().cpu(), x_orig.detach().cpu()

    grad_norm         = grad_z.norm(dim=(-2, -1), keepdim=True)      # (1, 1, 1)
    grad_z_normalised = grad_z / (grad_norm + EPS)                   # (1, S, H)

    z_prime = z.detach() + grad_z_normalised.detach()                # (1, S, H)

    with torch.no_grad():
        x_prime = model.decode(z_prime)                              # (1, S)

    return grad_z.detach().cpu(), z_prime.cpu(), x_prime.cpu()



# ---------------------------------------------------------------------------
# Per-sample perturbation — layer selective
# ---------------------------------------------------------------------------

def perturb_one_layer_selective_sample(
    model,
    hidden_with_cls_1: torch.Tensor,
    gene_ids_1: torch.Tensor,
    target_gene_ids: list[int],
    grad_from_layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply layer-selective perturbation to a single expression profile.

    Mirrors the structure of ``perturb_one_sample`` with the leaf tensor
    set at the intermediate hidden states h_n (after layer n-1) rather than
    the final gene embeddings.

    The forward chain is:
        h_n  (leaf, requires_grad=True)
          → encode_from_layer(h_n, n)   [layers n..N-1, grad enabled]
          → .gene_embeddings             [final-layer z, on graph]
          → model.decode(z)              [predicted expression, on graph]
          → perturbation_loss            [scalar loss]
          → backward()                   [populates h_n.grad]

    The returned gradient ∇_{h_n} L has the same shape ``(1, S, H)`` as the
    gradient returned by ``perturb_one_sample``, with the CLS position stripped
    so that ``compute_shift_score`` receives tensors of matching dimensionality
    regardless of mode.

    Parameters
    ----------
    model : EvaRnaModel
        EVA-RNA model (weights frozen, layer-selective patch applied).
    hidden_with_cls_1 : torch.Tensor
        Precomputed intermediate hidden states including CLS at position 0,
        shape ``(1, S+1, H)``, on CPU.  As cached by ``encode_samples``
        with ``up_to_layer=grad_from_layer``.
    gene_ids_1 : torch.Tensor
        Token IDs, shape ``(1, S)``, required by ``perturbation_loss``.
    target_gene_ids : list[int]
        Token IDs of target genes.
    grad_from_layer : int
        Layer index (0-based) at which the leaf tensor lives.  Must match the
        value used when caching ``hidden_with_cls_1``.
    device : torch.device

    Returns
    -------
    grad_h : torch.Tensor
        Raw (un-normalised) gradient w.r.t. h_n, CLS stripped,
        shape ``(1, S, H)``, CPU.  Used for cosine scoring against the
        disease-to-healthy axis computed at the same layer.
    h_prime : torch.Tensor
        Perturbed hidden states (CLS stripped), shape ``(1, S, H)``, CPU.
    x_prime : torch.Tensor
        Decoded expression from h', shape ``(1, S)``, CPU.
    """
    model.eval()

    # Re-attach the cached intermediate hidden states as a fresh leaf tensor.
    # CLS token is kept at position 0 so that encode_from_layer can resume
    # the forward pass in the same state as forward() would at layer n.
    h_n = (
        hidden_with_cls_1
        .to(device)
        .clone()
        .detach()
        .requires_grad_(True)
    )                                                                 # (1, S+1, H)

    directions = [PERTURBATION_DIR] * len(target_gene_ids)

    try:
        with torch.enable_grad():
            # Complete the encoder from layer n onward; returns EvaRnaOutput
            # with .gene_embeddings of shape (1, S, H) — CLS already stripped
            # inside encode_from_layer (mirrors forward() line 228).
            out = model.encode_from_layer(h_n, n=grad_from_layer)    # EvaRnaOutput

            # Decode final gene embeddings to predicted expression
            pred_expr = model.decode(out.gene_embeddings)            # (1, S)

            loss = perturbation_loss(
                predicted_expression=pred_expr,
                gene_ids=gene_ids_1,
                target_gene_ids=target_gene_ids,
                perturbation_directions=directions,
            )
            loss.mean().backward()
    except KeyError as exc:
        log.warning("Skipping layer-selective perturbation (target gene absent): %s", exc)
        with torch.no_grad():
            out_orig  = model.encode_from_layer(h_n.detach(), n=grad_from_layer)
            x_orig    = model.decode(out_orig.gene_embeddings)
        zero_grad = torch.zeros_like(h_n[:, 1:, :].detach())
        return zero_grad.cpu(), h_n[:, 1:, :].detach().cpu(), x_orig.detach().cpu()

    grad_h = h_n.grad  # (1, S+1, H) — gradient w.r.t. full hidden states incl. CLS

    if grad_h is None:
        log.warning("∇_{h_n} is None — returning original hidden states.")
        with torch.no_grad():
            out_orig = model.encode_from_layer(h_n.detach(), n=grad_from_layer)
            x_orig   = model.decode(out_orig.gene_embeddings)
        zero_grad = torch.zeros_like(h_n[:, 1:, :].detach())
        return zero_grad.cpu(), h_n[:, 1:, :].detach().cpu(), x_orig.detach().cpu()

    # Strip CLS from gradient and hidden states before returning.
    # The CLS position is not part of gene_embeddings in encode() output, and
    # is excluded from the disease-to-healthy axis computation in all modes.
    grad_h_genes = grad_h[:, 1:, :]                                  # (1, S, H)
    h_n_genes    = h_n[:, 1:, :].detach()                            # (1, S, H)

    # L2-normalise over (S, H) per sample — mirrors eq. 25 and perturb_one_sample
    grad_norm          = grad_h_genes.norm(dim=(-2, -1), keepdim=True)  # (1, 1, 1)
    grad_h_normalised  = grad_h_genes / (grad_norm + EPS)               # (1, S, H)

    # h'_n = h_n + ∇_{h_n} L  (gradient ascent, eq. 19 applied at layer n)
    h_prime = h_n_genes + grad_h_normalised.detach()                 # (1, S, H)

    # Decode h' by completing the encoder from layer n (re-insert CLS for
    # encode_from_layer, which expects hidden_states_with_cls).
    cls_h   = h_n[:, :1, :].detach()                                 # (1, 1, H)
    h_prime_with_cls = torch.cat([cls_h, h_prime], dim=1)            # (1, S+1, H)

    with torch.no_grad():
        out_prime = model.encode_from_layer(
            h_prime_with_cls, n=grad_from_layer
        )
        x_prime = model.decode(out_prime.gene_embeddings)            # (1, S)

    return grad_h_genes.detach().cpu(), h_prime.cpu(), x_prime.cpu()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_perturbation_pipeline(
    model,
    tokenizer,
    benchmark: pd.DataFrame,
    device: torch.device,
    mode: str,
    n_top_genes: int,
    grad_from_layer: int,
    all_axis: bool = True,
) -> pd.DataFrame:
    """Run the full zero-shot perturbation pipeline over the benchmark.

    Parameters
    ----------
    model : EvaRnaModel
        Frozen EVA-RNA model, on ``device``.
    tokenizer : EvaRnaTokenizer
    benchmark : pd.DataFrame
    device : torch.device
    mode : str
        One of ``"latent_space"``, ``"layer_selective"``.
    n_top_genes : int
        Number of highly variable genes retained by ``load_cohort_data``.
    grad_from_layer : int
        Transformer layer index for layer_selective mode (ignored otherwise).
    all_axis : bool
        If True, use the full gradient vector for cosine scoring (all gene
        positions).  If False, restrict to target-gene positions only.

    Returns
    -------
    pd.DataFrame
        Results with columns:
        ``drug_name``, ``disease_abbrev``, ``median_score``, ``expected_efficacy``.
    """
    output_dir = Path(f"data/perturbation_scores/{n_top_genes}_top_genes/{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    with MissingTargetGenesList(
        out=f"missing_target_genes_top{n_top_genes}.json"
    ) as missing:
        for disease_abbrev, disease_group in benchmark.groupby("disease_abbrev"):
            log.info("=== Disease: %s ===", disease_abbrev)

            # ---- Load cohort -----------------------------------------------
            target_genes_list = []
            for tg in disease_group.target_genes:
                for gene in tg.split(";"):
                    gene = gene.strip()
                    if gene in GENE_ALIAS_MAP:
                        target_genes_list += GENE_ALIAS_MAP[gene]
                    else:
                        target_genes_list.append(gene)

            adata = load_cohort_data(disease_abbrev, n_top_genes, target_genes_list)

            X_filtered, token_ids_tensor, _ = prepare_tokenisation(
                adata, tokenizer, device
            )

            symbol_to_entrez: dict[str, str] = {
                symbol: entrez
                for entrez, symbol in zip(
                    adata.var_names, adata.var["gene_symbols"]
                )
            }

            disease_mask = (adata.obs["disease"] != "Control").values
            healthy_mask = (adata.obs["disease"] == "Control").values

            disease_expr = X_filtered[disease_mask]
            healthy_expr = X_filtered[healthy_mask]

            n_disease = int(disease_mask.sum())
            n_healthy = int(healthy_mask.sum())
            log.info("  %d disease samples, %d healthy controls", n_disease, n_healthy)

            # ---- Encode all samples (mode-dependent) -----------------------
            log.info("  Encoding healthy samples...")
            log.info("  Encoding disease samples...")

            if mode == "latent_space":
                # Cache final-layer gene embeddings (CLS stripped) for both
                # cohorts.  These are the leaf tensors in perturb_one_sample.
                healthy_gene_embs = encode_samples(
                    model, healthy_expr, token_ids_tensor, device, desc="Healthy"
                )                                          # (n_healthy, S, H), (n_healthy, S)
                disease_gene_embs = encode_samples(
                    model, disease_expr, token_ids_tensor, device, desc="Disease"
                )                                          # (n_disease, S, H), (n_disease, S)

                # Disease-to-healthy axis in final latent space.
                # CLS is already stripped by encode_samples in full mode
                # (encode() returns gene_embeddings = hidden_states[:, 1:, :]).
                latent_axis = compute_healthy_disease_axis(
                    healthy_gene_embs, disease_gene_embs
                )                                          # (S*H,)

            elif mode == "layer_selective":
                # Cache intermediate hidden states at layer n (CLS included)
                # for both cohorts, to avoid re-running the first n layers per
                # drug per sample.
                healthy_h_with_cls = encode_samples(
                    model, healthy_expr, token_ids_tensor, device,
                    desc="Healthy", up_to_layer=grad_from_layer,
                )                                          # (n_healthy, S+1, H)
                disease_h_with_cls = encode_samples(
                    model, disease_expr, token_ids_tensor, device,
                    desc="Disease", up_to_layer=grad_from_layer,
                )                                          # (n_disease, S+1, H)

                # Axis at layer n: strip CLS (position 0) before computing.
                # This ensures the axis lives in the same (S, H) gene-token
                # space as the CLS-stripped gradient ∇_{h_n} L returned by
                # perturb_one_layer_selective_sample.
                latent_axis = compute_healthy_disease_axis(
                    healthy_h_with_cls[:, 1:, :],         # (n_healthy, S, H)
                    disease_h_with_cls[:, 1:, :],         # (n_disease, S, H)
                )                                          # (S*H,)

                disease_gene_embs = disease_h_with_cls    # alias for loop below
                healthy_gene_embs = healthy_h_with_cls


            # ---- Per-drug perturbation loop --------------------------------
            for _, row in disease_group.iterrows():
                drug_name        = row["drug_name"]
                target_genes_raw = row["target_genes"]
                expected         = row["expected_efficacy"]

                target_gene_ids: list[int] = []
                for symbol in str(target_genes_raw).split(";"):
                    symbol = symbol.strip()
                    entrez = symbol_to_entrez.get(symbol)
                    if entrez is None:
                        if symbol in GENE_ALIAS_MAP:
                            for s in GENE_ALIAS_MAP[symbol]:
                                entrez_alias = symbol_to_entrez.get(s)
                                if entrez_alias is not None:
                                    target_gene_ids.append(
                                        tokenizer.convert_tokens_to_ids(entrez_alias)
                                    )
                                else:
                                    missing.update(symbol, disease_abbrev, drug_name)
                        else:
                            log.warning(
                                "  Target gene '%s' not in cohort var — skipping for drug %s.",
                                symbol, drug_name,
                            )
                            missing.update(symbol, disease_abbrev, drug_name)
                    else:
                        target_gene_ids.append(tokenizer.convert_tokens_to_ids(entrez))

                if not target_gene_ids:
                    log.warning(
                        "  No target genes resolved for drug %s in disease %s — skipping.",
                        drug_name, disease_abbrev,
                    )
                    results.append({
                        "drug_name":         drug_name,
                        "disease_abbrev":    disease_abbrev,
                        "median_score":      float("nan"),
                        "expected_efficacy": expected,
                    })
                    continue

                log.info("  Drug: %-30s  targets: %s", drug_name, target_gene_ids)

                # Build target positions for per-gene cosine scoring
                gene_ids_ref, _ = make_batch_tensors(
                    disease_expr, token_ids_tensor, [0], device
                )
                token_row = gene_ids_ref[0]
                target_positions: list[int] = []
                for tid in target_gene_ids:
                    matches = (token_row == tid).nonzero(as_tuple=True)[0]
                    if matches.numel() > 0:
                        target_positions.append(int(matches[0].item()))

                if not target_positions:
                    log.warning(
                        "  No target positions in token sequence for drug %s"
                        " — falling back to full-gradient scoring.", drug_name,
                    )
                    target_positions_arg: list[int] | None = None
                elif all_axis:
                    target_positions_arg = None
                else:
                    target_positions_arg = target_positions

                # ---- Per-sample perturbation loop --------------------------
                grad_list: list[torch.Tensor] = []

                for i in tqdm(range(n_disease), desc=f"{drug_name}", leave=False):
                    gene_ids_1, expr_vals_1 = make_batch_tensors(
                        disease_expr, token_ids_tensor, [i], device
                    )                                      # (1, S) each

                    if mode == "latent_space":
                        gene_embs_1 = disease_gene_embs[i].unsqueeze(0)  # (1, S, H)
                        grad_1, _, _ = perturb_one_sample(
                            model=model,
                            gene_embeddings_1=gene_embs_1,
                            gene_ids_1=gene_ids_1,
                            target_gene_ids=target_gene_ids,
                            device=device,
                        )                                  # (1, S, H)

                    elif mode == "layer_selective":
                        h_with_cls_1 = disease_gene_embs[i].unsqueeze(0)  # (1, S+1, H)
                        grad_1, _, _ = perturb_one_layer_selective_sample(
                            model=model,
                            hidden_with_cls_1=h_with_cls_1,
                            gene_ids_1=gene_ids_1,
                            target_gene_ids=target_gene_ids,
                            grad_from_layer=grad_from_layer,
                            device=device,
                        )                                  # (1, S, H)

                    grad_list.append(grad_1)

                grad_tensor = torch.cat(grad_list, dim=0)  # (n_disease, S[, H])

                out_path = output_dir / f"{disease_abbrev}_{drug_name}_grad.npy"
                np.save(out_path, grad_tensor.numpy())
                log.info("  Saved grad → %s", out_path)

                cosines = compute_shift_score(
                    grad_tensor, latent_axis,
                    target_positions=target_positions_arg,
                )                                          # (n_disease,)
                median_score = float(cosines.mean().item())

                results.append({
                    "drug_name":         drug_name,
                    "disease_abbrev":    disease_abbrev,
                    "median_score":      median_score,
                    "expected_efficacy": expected,
                })

    results_df = pd.DataFrame(results)
    csv_path   = output_dir / "perturbation_results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("Summary saved → %s", csv_path)
    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    log.info(
        "Mode: %s | n_top_genes: %d | grad_from_layer: %d | all_axis: %s | benchmark_path: %s",
        args.mode, args.n_top_genes, args.grad_from_layer, args.all_axis, args.benchmark_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading EVA-RNA model...")
    model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    if args.mode == "layer_selective":
        from patch_eva_rna import apply_layer_selective_patch
        apply_layer_selective_patch(model)
        log.info(
            "Layer-selective patch applied. Gradient leaf at layer %d.",
            args.grad_from_layer,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "ScientaLab/eva-rna", trust_remote_code=True
    )

    benchmark = pd.read_csv(args.benchmark_path)
    log.info("Benchmark: %d drug-disease pairs", len(benchmark))

    results = run_perturbation_pipeline(
        model=model,
        tokenizer=tokenizer,
        benchmark=benchmark,
        device=device,
        mode=args.mode,
        n_top_genes=args.n_top_genes,
        grad_from_layer=args.grad_from_layer,
        all_axis=args.all_axis
    )
    log.info("Done.")