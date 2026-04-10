"""Perturbation scoring utilities for EVA-RNA zero-shot efficacy prediction.

Scoring formulation
-------------------
We follow Bjerregaard et al. (2025) equations 3–5, adapted to EVA-RNA's
gene_embeddings latent space.

In the original paper, ``z`` is a scalar VAE bottleneck (e.g. R^32).  In
EVA-RNA, the equivalent latent is ``gene_embeddings``, shape ``(S, H)`` — the
per-gene contextualised hidden states that are the direct input to the
expression decoder.  Both the perturbation gradient ``∇_z L`` and the
disease–healthy displacement axis live in this same ``(S, H)`` space.

There is intentionally no CLS token involved.  The CLS embedding is a
sample-level summary produced by a separate pathway and is never part of the
perturbation computation graph (the backward pass stops at gene_embeddings).

Two scoring modes are supported, selectable via the ``target_positions``
argument of ``compute_shift_score``:

Full-gradient mode (target_positions=None)
    Flatten the full gradient (S, H) → (S*H,) and compare against the full
    displacement axis (S*H,).  Treats the perturbation as a whole-transcriptome
    event.  Attention propagation means all S gene positions carry non-zero
    gradient even for a single-target drug.

Target-gene mode (target_positions=[pos_g, ...])
    Restrict both the gradient and the axis to the hidden states of the target
    gene(s) only: grad_zs[:, target_positions, :] and axis[target_positions, :],
    each of shape (n_targets, H) before flattening.  This is a closer reading
    of Bjerregaard eq. 5, where s_i is the score of gene i specifically — here
    applied to the drug's target gene(s) rather than all genes.  Scores drug
    efficacy by whether the gradient at the target gene position(s) aligns with
    the healthy direction, without dilution from attention-propagated noise in
    the other S-1 gene positions.

Three public functions are exposed:

    compute_healthy_disease_axis(healthy_gene_embs, disease_gene_embs)
        Displacement axis a = mean(z_healthy) - mean(z_disease), shape (S*H,),
        unit-normalised.  Shared by both scoring modes.

    compute_shift_score(grad_zs, axis, target_positions=None)
        Per-sample cosine similarities, shape (n_disease,).  Caller takes
        .mean() for the drug-disease scalar score.

    compute_healthy_centroid(healthy_predicted_expression)
        Legacy helper retained for pipeline compatibility.
"""

from __future__ import annotations

import torch


def compute_healthy_disease_axis(
    healthy_gene_embs: torch.Tensor,
    disease_gene_embs: torch.Tensor,
) -> torch.Tensor:
    """Compute the latent displacement axis from disease toward healthy.

    Implements Bjerregaard et al. eq. 3, adapted to EVA-RNA gene_embeddings
    space:
        a = mean(z_healthy) - mean(z_disease)

    The axis is L2-normalised so that cosine similarities in
    ``compute_shift_score`` are in [-1, 1] regardless of the magnitude of the
    group means.

    Parameters
    ----------
    healthy_gene_embs : torch.Tensor
        Gene embeddings for healthy control samples, shape
        ``(n_healthy, S, H)``, as returned by ``encode_and_decode_samples``.
    disease_gene_embs : torch.Tensor
        Gene embeddings for disease samples, shape
        ``(n_disease, S, H)``.

    Returns
    -------
    torch.Tensor
        Unit-normalised displacement axis, shape ``(S*H,)``.
    """
    if healthy_gene_embs.ndim != 3 or disease_gene_embs.ndim != 3:
        raise ValueError(
            "Both inputs must be 3D tensors (n_samples, S, H), "
            f"got {healthy_gene_embs.shape} and {disease_gene_embs.shape}."
        )

    mean_healthy = healthy_gene_embs.mean(dim=0)    # (S, H)
    mean_disease = disease_gene_embs.mean(dim=0)    # (S, H)

    axis = (mean_healthy - mean_disease).flatten()  # (S*H,)
    axis_norm = axis.norm()
    if axis_norm < 1e-12:
        raise ValueError(
            "Healthy and disease gene embedding centroids are identical — "
            "cannot define a displacement axis.  Check that healthy and "
            "disease samples are correctly separated in adata.obs."
        )
    return axis / axis_norm                         # (S*H,), unit norm


def compute_shift_score(
    grad_zs: torch.Tensor,
    axis: torch.Tensor,
    target_positions: list[int] | None = None,
) -> torch.Tensor:
    """Score a drug-disease pair by gradient–axis cosine similarity (eq. 5).

    For each disease sample i, compute the cosine similarity between (a slice
    of) its perturbation gradient and the corresponding slice of the
    disease-to-healthy displacement axis:

        s_i = (∇_z L_i · a) / (||∇_z L_i|| · ||a||)

    Because ``axis`` is already unit-normalised, the denominator reduces to
    ``||∇_z L_i||`` (or ``||∇_z L_i[target_positions, :]||`` in target-gene mode).

    Two modes are selected by ``target_positions``:

    Full-gradient mode (target_positions=None)
        The full gradient (S, H) and the full axis (S*H,) are compared.
        grad_zs is reshaped to (n_disease, S*H) for the dot product.

    Target-gene mode (target_positions=[pos_g, ...])
        Only the hidden states at the target gene sequence positions are used.
        ``grad_zs[:, target_positions, :]`` has shape
        ``(n_disease, n_targets, H)``, flattened to ``(n_disease, n_targets*H)``.
        ``axis`` is reshaped to ``(S, H)`` and sliced to
        ``(n_targets, H)`` before flattening to ``(n_targets*H,)``.
        The axis slice is re-normalised to unit norm so cosine values remain
        in [-1, 1].

    Parameters
    ----------
    grad_zs : torch.Tensor
        Stacked per-sample perturbation gradients, shape ``(n_disease, S, H)``.
        Raw (un-normalised) gradients from ``perturb_one_sample``; cosine is
        scale-invariant so pre-normalisation is unnecessary.
    axis : torch.Tensor
        Unit-normalised displacement axis, shape ``(S*H,)``, from
        ``compute_healthy_disease_axis``.
    target_positions : list[int] or None
        Sequence positions (indices into dim 1 of grad_zs) of the drug's target
        gene(s) within the filtered token sequence.  When None, full-gradient
        mode is used.  When provided, target-gene mode is used and the cosine
        is computed only over those positions.

    Returns
    -------
    torch.Tensor
        Per-sample cosine similarities, shape ``(n_disease,)``, each in
        [-1, 1].  Take ``.mean()`` for the drug-disease scalar score.
    """
    if grad_zs.ndim != 3:
        raise ValueError(
            f"grad_zs must be 3D (n_disease, S, H), got {tuple(grad_zs.shape)}."
        )
    if axis.ndim != 1:
        raise ValueError(
            f"axis must be 1D (S*H,), got {tuple(axis.shape)}."
        )

    n, S, H = grad_zs.shape

    if target_positions is None:
        # ── Full-gradient mode ───────────────────────────────────────────────
        grads_flat = grad_zs.reshape(n, S * H)             # (n_disease, S*H)
        axis_vec   = axis                                   # (S*H,)
    else:
        # ── Target-gene mode ─────────────────────────────────────────────────
        if not target_positions:
            raise ValueError("target_positions must be non-empty when provided.")
        pos = torch.tensor(target_positions, dtype=torch.long)  # (n_targets,)
        grads_flat = grad_zs[:, pos, :].reshape(n, -1)     # (n_disease, n_targets*H)

        # Slice the axis at the same positions, then re-normalise.
        # axis has shape (S*H,); reshape to (S, H) to index by gene position.
        axis_2d  = axis.reshape(S, H)                      # (S, H)
        axis_vec = axis_2d[pos, :].flatten()               # (n_targets*H,)
        axis_norm = axis_vec.norm()
        if axis_norm < 1e-12:
            # Axis has no signal at target positions — fall back to zero scores.
            return torch.zeros(n)
        axis_vec = axis_vec / axis_norm                     # (n_targets*H,), unit norm

    grad_norms = grads_flat.norm(dim=1, keepdim=True) + 1e-8   # (n_disease, 1)
    grads_unit = grads_flat / grad_norms                        # (n_disease, ?)
    cosines    = grads_unit @ axis_vec                          # (n_disease,)
    return cosines


# ---------------------------------------------------------------------------
# Legacy helper — retained for pipeline compatibility
# ---------------------------------------------------------------------------

def compute_healthy_centroid(healthy_predicted_expression: torch.Tensor) -> torch.Tensor:
    """Compute the mean decoded expression of healthy controls.

    No longer used for scoring under the latent-space formulation, but
    retained so ``run_perturbation_pipeline`` does not require restructuring.

    Parameters
    ----------
    healthy_predicted_expression : torch.Tensor
        Shape ``(n_healthy, seq_len)``.

    Returns
    -------
    torch.Tensor
        Shape ``(seq_len,)``.
    """
    if healthy_predicted_expression.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor (n_healthy, seq_len), "
            f"got {tuple(healthy_predicted_expression.shape)}."
        )
    return healthy_predicted_expression.mean(dim=0)