import torch

def token_accuracy(stacks, preds, eos, pad, msk):
    """
    Compute both global accuracy (weighted) and per-sample accuracy vector.

    Args:
        stacks: target ids [B, L]
        preds:  logits [B, L, V] or ids [B, L]
        eos:    int id of EOS
        pad:    int id of PAD
        msk:    int id of MSK

    Returns:
        global_acc : scalar float tensor (on CPU)
        per_sample : [B] float tensor (on CPU)
    """
    if preds.dim() == 3:
        preds = preds.argmax(dim=-1)

    L = min(stacks.size(1), preds.size(1))
    stacks = stacks[:, :L]
    preds  = preds[:,  :L]

    # positions strictly before first EOS
    is_eos = (stacks == eos)
    before_first_eos = (is_eos.cumsum(dim=1) == 0)

    # exclude PADs and MSKs
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    correct = (stacks == preds) & valid

    # --- per-sample accuracy ---
    per_correct = correct.sum(dim=1).float()
    per_total   = valid.sum(dim=1).clamp_min(1).float()
    per_sample  = (per_correct / per_total).detach().cpu()

    # --- global weighted accuracy ---
    global_acc = (per_correct.sum() / per_total.sum()).detach().cpu()

    return global_acc, per_sample


def masked_mae(x, y):
    # treat only finite y as valid (predicted_spectra)
    mask = torch.isfinite(y).all(dim=-1, keepdim=True)  # [B,1,W]
    valid = mask.expand_as(y)
    num = (torch.abs(x - torch.nan_to_num(y))).where(valid, torch.zeros_like(x)).sum(dim=1).sum(dim=1)
    den = valid.sum(dim=1).sum(dim=1).clamp_min(1)
    return num / den
