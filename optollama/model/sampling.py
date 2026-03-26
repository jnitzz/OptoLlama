import torch


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Apply combined top-k and top-p (nucleus) filtering to logits.

    Filters along the last dimension of ``logits`` (shape ``[..., V]``).

    Args
    ----
    logits : torch.Tensor
        Raw logits of shape ``[..., V]``.
    top_k : int
        If > 0, keep only the top-k highest-probability tokens.
    top_p : float
        If > 0, keep the smallest set of tokens whose cumulative probability
        exceeds ``top_p`` (nucleus sampling).
    filter_value : float
        Value assigned to filtered-out positions (default: ``-inf``).

    Returns
    -------
    torch.Tensor
        Filtered logits of the same shape as the input.

    Notes
    -----
    Uses a large finite number for NaNs/Infs to keep kernels stable.
    If both ``top_k`` and ``top_p`` are disabled, returns logits unchanged.
    """
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
    v = logits.size(-1)

    if top_k and top_k > 0:
        k = min(int(top_k), v)
        kth = torch.topk(logits, k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, filter_value), logits)

    if top_p and top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        remove = cumprobs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        mask = torch.zeros_like(remove, dtype=torch.bool).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, filter_value)

    return logits
