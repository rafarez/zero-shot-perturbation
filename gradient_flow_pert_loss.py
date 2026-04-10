"""Zero-shot in-silico perturbation loss for EVA-RNA.

Implements the gradient-flow perturbation objective from:
  EVA technical report (Scienta Lab, 2026) — Section 3.6.2
  

Usage
-----
>>> loss = perturbation_loss(
...     predicted_expression=model.decode(gene_embeddings),  # (B, S)
...     gene_ids=gene_ids,                                   # (B, S)
...     target_gene_ids=[1234, 5678],
...     perturbation_directions=[-1, +1],
... )
>>> loss.backward()  # gradients flow back to the hidden states

>>> # With per-gene intensities:
>>> loss = perturbation_loss(
...     predicted_expression=model.decode(gene_embeddings),
...     gene_ids=gene_ids,
...     target_gene_ids=[1234, 5678],
...     perturbation_directions=[-1, +1],
...     alpha=[2.0, 0.5],
... )
"""

from __future__ import annotations

import torch


def perturbation_loss(
    predicted_expression: torch.Tensor,
    gene_ids: torch.Tensor,
    target_gene_ids: list[int],
    perturbation_directions: list[int],
    alpha: list[float] | None = None,
) -> torch.Tensor:
    """Compute the gradient-flow perturbation loss for EVA-RNA.

    For each target gene g, the per-gene loss is:

        L_g = delta_g * alpha_g * ê_g

    where
        delta_g      ∈ {-1, +1}  — perturbation direction (inhibition / activation)
        alpha_g      > 0          — perturbation intensity, defaults to 1.0 per gene
        ê_g                   — sum of decoder predicted expression at the
                                positions where gene_ids == g, across the batch

    The returned tensor has shape ``(batch,)`` — one loss value per sample —
    computed as the mean of per-gene losses across all targets.  Keeping the
    batch dimension intact ensures that gradient magnitude is independent of
    cohort size, and that per-sample gradients can be L2-normalised individually
    before being applied to z (EVA report eq. 25).

    Backpropagating this loss through the decoder yields gradients ∇L/∇z
    that encode the direction of "more expression" for activation targets and
    "less expression" for inhibition targets.  The caller is responsible for
    applying those gradients to the relevant hidden states (layer-selective
    perturbation, l = n-1) following equation (23) of the EVA report:

        h'(l) = h(l) + ∇_{h(l)} L

    Parameters
    ----------
    predicted_expression : torch.Tensor
        Decoder output of shape ``(batch, seq_len)``.  The hidden states that
        produced it must have ``requires_grad=True`` so that ``.backward()``
        propagates gradients back to them.
    gene_ids : torch.Tensor
        Integer gene token IDs of shape ``(batch, seq_len)``, in the same
        sequence order as ``predicted_expression``.  Used to locate target
        genes by their Entrez ID within the sequence.
    target_gene_ids : list[int]
        Entrez IDs of the drug's target genes.
    perturbation_directions : list[int]
        Per-gene direction: ``+1`` for activation (overexpression),
        ``-1`` for inhibition (knockdown).  Must be the same length as
        ``target_gene_ids``.
    alpha : list[float] | None
        Per-gene perturbation intensities.  If ``None`` (default), all genes
        use alpha = 1.0 following the EVA paper.  If provided, must be a list of
        positive floats of the same length as ``target_gene_ids``.

    Returns
    -------
    torch.Tensor
        Per-sample loss of shape ``(batch,)``, averaged over all target genes.

    Raises
    ------
    ValueError
        If list arguments differ in length, if a direction is not ±1, if
        ``target_gene_ids`` is empty, or if any alpha value is non-positive.
    KeyError
        If a target gene ID is not found anywhere in ``gene_ids``.

    Notes
    -----
    **Sign convention and gradient ascent**
    The loss is used for gradient *ascent* — the caller steps as
    ``h' = h + ∇_h L`` (EVA report eq. 23).

    With δ_g = -1 (inhibition):
        L_g = -ê_g  →  ∇_h L = -∇_h ê_g
        h' = h - ∇_h ê_g   →  moves h to *decrease* ê_g  ✓

    With δ_g = +1 (activation):
        L_g = +ê_g  →  ∇_h L = +∇_h ê_g
        h' = h + ∇_h ê_g   →  moves h to *increase* ê_g  ✓

    **Differentiability**
    Gene positions are located via a soft boolean mask cast to float, keeping
    the full computation graph intact.  Direct integer indexing (e.g. via
    torch.where) would detach the selected values from autograd.
    """
    _validate_inputs(target_gene_ids, perturbation_directions, alpha)

    # Resolve alpha: default to 1.0 per gene if not provided
    alphas: list[float] = alpha if alpha is not None else [1.0] * len(target_gene_ids)

    per_gene_losses: list[torch.Tensor] = []

    for gene_id, direction, alpha_g in zip(target_gene_ids, perturbation_directions, alphas):
        # Locate target gene positions across the batch.
        # match : (B, S) boolean — True where gene_ids == gene_id
        match = gene_ids == gene_id  # (B, S)

        if not match.any():
            raise KeyError(
                f"Target gene ID {gene_id} was not found in any position of "
                f"gene_ids.  Check that the tokenizer vocabulary contains this "
                f"Entrez ID and that the gene was included in the input sequence "
                f"(EVA-RNA subsamples genes; some may be absent)."
            )

        # Extract predicted expression at the target gene's position(s).
        # The mask is cast to float and multiplied element-wise so the
        # operation stays on the computation graph — direct indexing with
        # torch.where indices would detach gradients.
        # Shape after sum: (B,) — one scalar per sample (0 if gene absent).
        target_expr = (predicted_expression * match.float()).sum(dim=-1)  # (B,)

        # L_g : (B,) — per-sample loss for this target gene
        # No reduction over the batch: each sample keeps its own signal so
        # that gradients can be normalised per-sample downstream.
        loss_g = direction * alpha_g * target_expr  # (B,)
        per_gene_losses.append(loss_g)

    # Stack to (num_genes, B), then mean over genes → (B,).
    # Averaging over targets keeps gradient magnitude comparable across
    # drugs with different numbers of molecular targets.
    return torch.stack(per_gene_losses).mean(dim=0)  # (B,)


# ---------------------------------------------------------------------------
# Input validation (kept outside the main function for readability)
# ---------------------------------------------------------------------------

def _validate_inputs(
    target_gene_ids: list[int],
    perturbation_directions: list[int],
    alpha: list[float] | None,
) -> None:
    if not target_gene_ids:
        raise ValueError("target_gene_ids must not be empty.")

    n = len(target_gene_ids)

    if len(perturbation_directions) != n:
        raise ValueError(
            f"target_gene_ids (len={n}) and perturbation_directions "
            f"(len={len(perturbation_directions)}) must have the same length."
        )

    invalid_dirs = [d for d in perturbation_directions if d not in (-1, +1)]
    if invalid_dirs:
        raise ValueError(
            f"All perturbation directions must be +1 or -1, got: {invalid_dirs}"
        )

    if alpha is not None:
        if len(alpha) != n:
            raise ValueError(
                f"alpha (len={len(alpha)}) must have the same length as "
                f"target_gene_ids (len={n})."
            )
        non_positive = [a for a in alpha if a <= 0]
        if non_positive:
            raise ValueError(
                f"All alpha values must be positive, got: {non_positive}"
            )