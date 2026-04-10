"""Monkey-patch helpers for layer-selective perturbation in EVA-RNA.

Adds two methods to an ``EvaRnaModel`` instance that split the transformer
forward pass at a chosen layer boundary:

    model.encode_up_to_layer(gene_ids, expression_values, n, ...)
        Runs the embedding projection + value embedder + layers 0..n-1.
        Returns an ``EvaRnaOutput`` whose ``gene_embeddings`` field holds the
        intermediate hidden states at layer n (CLS token stripped, shape
        ``(batch, seq_len, hidden_size)``).  This is the leaf tensor for the
        layer-selective gradient.

    model.encode_from_layer(hidden_states_with_cls, n, padding_mask, autocast)
        Runs layers n..N-1 on the supplied hidden states (CLS token still
        present), strips CLS, and returns an ``EvaRnaOutput`` identical in
        structure to ``model.encode()``.

Design notes
------------
- Both methods mirror ``forward()`` (lines 101-239 of modeling_eva_rna.py)
  as closely as possible — same autocast logic, same flash-attention branch,
  same CLS handling.
- ``encode_up_to_layer`` keeps the CLS token in ``hidden_states_with_cls``
  so that ``encode_from_layer`` can accept it without additional bookkeeping.
  The CLS position is stripped at the *end* of each method before returning
  ``EvaRnaOutput``, exactly as ``forward()`` does on line 228.
- LayerNorm is pre-LN and baked into every ``_FlashEncoderLayer`` /
  ``nn.TransformerEncoderLayer``, so no separate norm step is needed after
  the final layer — again matching ``forward()``.
- The patch is applied to an **instance** (not the class) so it does not
  affect other code that imports ``EvaRnaModel``.

Usage
-----
    from patch_eva_rna import apply_layer_selective_patch

    model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
    apply_layer_selective_patch(model)

    # Now available:
    out = model.encode_up_to_layer(gene_ids, expr_vals, n=22)
    out2 = model.encode_from_layer(hidden_states_with_cls, n=22)
"""

from __future__ import annotations

import types
import warnings

import torch
from eva_rna.modeling_eva_rna import EvaRnaOutput, NON_GENE_EXPRESSION_VALUE


# ---------------------------------------------------------------------------
# The two new methods (defined as plain functions, bound to instance below)
# ---------------------------------------------------------------------------

def _encode_up_to_layer(
    self,
    gene_ids: torch.Tensor,
    expression_values: torch.Tensor,
    n: int,
    attention_mask: torch.Tensor | None = None,
    autocast: bool = True,
) -> tuple[EvaRnaOutput, torch.Tensor, torch.Tensor | None]:
    """Run the embedding stage + first ``n`` transformer layers.

    Mirrors ``forward()`` up to — but not including — ``self.layers[n]``.
    The CLS token is prepended exactly as in ``forward()``.

    Parameters
    ----------
    gene_ids : torch.Tensor
        Token IDs, shape ``(batch, seq_len)``.
    expression_values : torch.Tensor
        Log1p-normalised expression, shape ``(batch, seq_len)``.
    n : int
        Number of transformer layers to run (0-based stop index, exclusive).
        Valid range: 1 ≤ n ≤ len(self.layers).
    attention_mask : torch.Tensor | None
        Boolean padding mask (True = attend).
    autocast : bool
        Match ``forward()`` autocast behaviour.

    Returns
    -------
    intermediate_out : EvaRnaOutput
        ``gene_embeddings`` = hidden states after layer n-1, CLS stripped,
        shape ``(batch, seq_len, hidden_size)``.  This tensor is the intended
        leaf for ``requires_grad_(True)`` in the perturbation function.
    hidden_states_with_cls : torch.Tensor
        Full hidden states including CLS at position 0, shape
        ``(batch, seq_len+1, hidden_size)``.  Pass this directly to
        ``encode_from_layer`` to complete the forward pass.
    padding_mask : torch.Tensor | None
        Padding mask with CLS column prepended, or None.  Passed through to
        ``encode_from_layer`` unchanged so it does not need to be recomputed.
    """
    n_layers = len(self.layers)
    if not (1 <= n <= n_layers):
        raise ValueError(
            f"n must be in [1, {n_layers}], got {n}."
        )

    batch_size = gene_ids.shape[0]
    device = gene_ids.device

    if self._use_flash and device.type != "cuda":
        raise RuntimeError(
            f"Flash attention requires CUDA, but inputs are on '{device}'."
        )

    # --- autocast setup (identical to forward()) ----------------------------
    _autocast_enabled = False
    _autocast_dtype = torch.float32
    if autocast and device.type == "cuda":
        _autocast_enabled = True
        _autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    if self._use_flash and not _autocast_enabled:
        warnings.warn(
            "autocast=False was requested, but flash attention requires "
            "half-precision. Enabling autocast automatically.",
            stacklevel=2,
        )
        _autocast_enabled = True
        _autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    with torch.autocast(
        device_type=device.type,
        dtype=_autocast_dtype,
        enabled=_autocast_enabled,
    ):
        # --- Prepend CLS token (mirrors forward() lines 169-183) ------------
        cls_tokens = torch.full(
            (batch_size, 1),
            self.config.cls_token_id,
            dtype=gene_ids.dtype,
            device=device,
        )
        cls_values = torch.full(
            (batch_size, 1),
            NON_GENE_EXPRESSION_VALUE,
            dtype=expression_values.dtype,
            device=device,
        )
        gene_ids_full        = torch.cat([cls_tokens, gene_ids], dim=1)
        expression_values_full = torch.cat([cls_values, expression_values], dim=1)

        # --- Padding mask (mirrors forward() lines 185-192) -----------------
        if attention_mask is not None:
            padding_mask = ~attention_mask
            cls_mask = torch.zeros(
                (batch_size, 1), dtype=padding_mask.dtype, device=device
            )
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            padding_mask = None

        # --- Embedding stage (mirrors forward() lines 194-205) --------------
        gene_emb = self.gene_embeddings(gene_ids_full)

        is_gene = ~torch.isin(gene_ids_full, self.special_token_ids)

        safe_values = torch.clamp(expression_values_full, min=0.0)
        value_emb = self.value_embedder(safe_values.unsqueeze(-1))
        value_emb = self.value_embedder_norm(value_emb)
        value_emb = torch.where(
            is_gene.unsqueeze(-1), value_emb, torch.zeros_like(value_emb)
        )

        hidden_states = gene_emb + value_emb   # (batch, seq_len+1, hidden_size)

        # --- First n transformer layers (mirrors forward() lines 207-225) ---
        if self._use_flash and padding_mask is not None:
            valid_mask = ~padding_mask
            from eva_rna.modeling_eva_rna import _pack_sequences, _unpack_sequences
            packed, cu_seqlens, max_seqlen = _pack_sequences(hidden_states, padding_mask)
            for layer in self.layers[:n]:
                packed = layer(src=packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            hidden_states = _unpack_sequences(packed, valid_mask, hidden_states.shape)
        else:
            for layer in self.layers[:n]:
                hidden_states = layer(
                    src=hidden_states, src_key_padding_mask=padding_mask
                )

        # hidden_states_with_cls retains the CLS position so that
        # encode_from_layer can resume from the same state as forward() would.
        hidden_states_with_cls = hidden_states  # (batch, seq_len+1, hidden_size)

        # Strip CLS for the returned EvaRnaOutput (mirrors forward() line 228)
        gene_embeddings_intermediate = hidden_states_with_cls[:, 1:, :]  # (batch, seq_len, H)

        # Apply gene padding mask (mirrors forward() lines 230-234)
        if padding_mask is not None:
            gene_padding_mask = padding_mask[:, 1:]
            gene_embeddings_intermediate = gene_embeddings_intermediate.masked_fill(
                gene_padding_mask.unsqueeze(-1), 0.0
            )

    return (
        EvaRnaOutput(
            gene_embeddings=gene_embeddings_intermediate,
            cls_embedding=None,    # CLS not yet meaningful at intermediate layer
        ),
        hidden_states_with_cls,
        padding_mask,
    )


def _encode_from_layer(
    self,
    hidden_states_with_cls: torch.Tensor,
    n: int,
    padding_mask: torch.Tensor | None = None,
    autocast: bool = True,
) -> EvaRnaOutput:
    """Run transformer layers ``n..N-1`` and return the final ``EvaRnaOutput``.

    Picks up exactly where ``encode_up_to_layer`` stopped, producing output
    identical in structure to ``model.encode()``.

    Parameters
    ----------
    hidden_states_with_cls : torch.Tensor
        Intermediate hidden states including CLS at position 0, shape
        ``(batch, seq_len+1, hidden_size)``.  Typically the leaf tensor
        returned by ``encode_up_to_layer``, re-attached with
        ``requires_grad_(True)`` before being passed here.
    n : int
        Layer index to start from (0-based, inclusive).
        Must match the ``n`` used in ``encode_up_to_layer``.
    padding_mask : torch.Tensor | None
        Padding mask with CLS column already prepended, as returned by
        ``encode_up_to_layer``.
    autocast : bool
        Match ``forward()`` autocast behaviour.

    Returns
    -------
    EvaRnaOutput
        ``cls_embedding`` of shape ``(batch, hidden_size)`` and
        ``gene_embeddings`` of shape ``(batch, seq_len, hidden_size)``,
        identical in structure to the output of ``model.encode()``.
    """
    n_layers = len(self.layers)
    if not (0 <= n < n_layers):
        raise ValueError(
            f"n must be in [0, {n_layers - 1}), got {n}."
        )

    device = hidden_states_with_cls.device

    # --- autocast setup (mirrors forward()) ---------------------------------
    _autocast_enabled = False
    _autocast_dtype = torch.float32
    if autocast and device.type == "cuda":
        _autocast_enabled = True
        _autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    if self._use_flash and not _autocast_enabled:
        warnings.warn(
            "autocast=False was requested, but flash attention requires "
            "half-precision. Enabling autocast automatically.",
            stacklevel=2,
        )
        _autocast_enabled = True
        _autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    with torch.autocast(
        device_type=device.type,
        dtype=_autocast_dtype,
        enabled=_autocast_enabled,
    ):
        hidden_states = hidden_states_with_cls

        # --- Remaining transformer layers (mirrors forward() lines 207-225) -
        if self._use_flash and padding_mask is not None:
            valid_mask = ~padding_mask
            from eva_rna.modeling_eva_rna import _pack_sequences, _unpack_sequences
            packed, cu_seqlens, max_seqlen = _pack_sequences(hidden_states, padding_mask)
            for layer in self.layers[n:]:
                packed = layer(src=packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            hidden_states = _unpack_sequences(packed, valid_mask, hidden_states.shape)
        else:
            for layer in self.layers[n:]:
                hidden_states = layer(
                    src=hidden_states, src_key_padding_mask=padding_mask
                )

        # --- Strip CLS and build output (mirrors forward() lines 227-239) ---
        cls_embedding  = hidden_states[:, 0, :]   # (batch, hidden_size)
        gene_embeddings = hidden_states[:, 1:, :]  # (batch, seq_len, hidden_size)

        if padding_mask is not None:
            gene_padding_mask = padding_mask[:, 1:]
            gene_embeddings = gene_embeddings.masked_fill(
                gene_padding_mask.unsqueeze(-1), 0.0
            )

    return EvaRnaOutput(
        gene_embeddings=gene_embeddings,
        cls_embedding=cls_embedding,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_layer_selective_patch(model) -> None:
    """Bind ``encode_up_to_layer`` and ``encode_from_layer`` to a model instance.

    Patches the specific *instance*, not the class, to avoid side-effects on
    other code that may import ``EvaRnaModel``.

    Parameters
    ----------
    model : EvaRnaModel
        The loaded EVA-RNA model instance (must already be on its target device
        and have weights frozen if desired).

    Example
    -------
    >>> from patch_eva_rna import apply_layer_selective_patch
    >>> apply_layer_selective_patch(model)
    >>> out, h_with_cls, pmask = model.encode_up_to_layer(gene_ids, expr, n=22)
    >>> out2 = model.encode_from_layer(h_with_cls, n=22, padding_mask=pmask)
    """
    model.encode_up_to_layer = types.MethodType(_encode_up_to_layer, model)
    model.encode_from_layer  = types.MethodType(_encode_from_layer,  model)