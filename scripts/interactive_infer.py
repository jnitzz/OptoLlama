#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, os, sys, json
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch

import dataset
from utils import init_tokenmaps, load_state_dict_flexible
from model import build_model
from call_tmm_fast import load_materials, TMMSpectrum

# -------------------- small helpers --------------------

def pick_device(user_device: Optional[str] = None) -> torch.device:
    if user_device:
        return torch.device(user_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_tmm(cfg, device: torch.device, idx_to_token: Dict[int,str]) -> Tuple[TMMSpectrum, torch.Tensor, torch.Tensor]:
    import math
    degree = math.pi / 180.0
    theta  = torch.tensor(cfg.INCIDENCE_ANGLE * degree, device=device, dtype=torch.complex128).unsqueeze(0)
    cfg.WAVELENGTHS = np.arange(cfg.WAVELENGTH_MIN, cfg.WAVELENGTH_MAX + 1, cfg.WAVELENGTH_STEPS)
    wl_tensor = torch.tensor(cfg.WAVELENGTHS, dtype=torch.complex128, device=device)
    nk_dict = load_materials(cfg)
    tmm = TMMSpectrum(nk_dict, idx_to_token, device=device).to(device)
    return tmm, wl_tensor, theta

def logits_to_token_ids(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [1, S, V] from model(spectrum)
    returns: [1, S] argmax ids
    """
    return logits.argmax(dim=-1)

def decode_until_eos(ids: torch.Tensor, eos: int, pad: int, msk: int, idx_to_token: Dict[int,str]) -> List[str]:
    ids = ids.view(-1).tolist()
    out = []
    for t in ids:
        if t == eos: break
        if t in (pad, msk): continue
        out.append(idx_to_token[int(t)])
    return out

def pretty_stack(tokens: List[str]) -> str:
    return "  ".join(tokens)

def load_user_spectrum(path: Optional[str], W: int) -> np.ndarray:
    """
    Returns np.ndarray shape [W, 3] with columns [R, A, T] (float).
    Accepted formats:
      - .npy: array of shape (W,3) or (3,W)
      - .json: either [[R,A,T], ...] length W, or flat length 3W (R/A/T blocks)
      - .csv/.txt: 3 columns per line (R,A,T); header allowed/ignored
      - None: read three lines from stdin: 'R: ...', 'A: ...', 'T: ...'
    """
    def as_w3(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (W, 3):
            return arr
        if arr.ndim == 2 and arr.shape == (3, W):
            return arr.T
        if arr.ndim == 1 and arr.size == 3*W:
            return arr.reshape(3, W).T
        raise ValueError(f"Expected spectrum of shape (W,3) or (3,W) or flat 3W, got {arr.shape}.")

    if path is None:
        # REPL entry: three lines
        def parse_line(prefix: str) -> np.ndarray:
            line = input(prefix).strip()
            if ':' in line: line = line.split(':', 1)[1]
            vals = [float(x) for x in line.replace(',', ' ').split()]
            if len(vals) != W:
                raise ValueError(f"Expected {W} values, got {len(vals)}")
            return np.array(vals, dtype=np.float64)
        R = parse_line("R: ")
        A = parse_line("A: ")
        T = parse_line("T: ")
        return np.stack([R, A, T], axis=1)

    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
        return as_w3(arr)
    if ext == '.json':
        with open(path, 'r') as fh:
            obj = json.load(fh)
        return as_w3(np.array(obj, dtype=np.float64))
    # csv / txt
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'): continue
            # skip header if detect non-numeric
            try:
                parts = [float(x) for x in line.replace(',', ' ').split()]
            except ValueError:
                continue
            if len(parts) >= 3:
                rows.append(parts[:3])
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[0] != W:
        raise ValueError(f"Expected {W} rows, got {arr.shape[0]} (make sure wavelengths match the config grid).")
    if arr.shape[1] < 3:
        raise ValueError("Need at least 3 columns: R A T.")
    return arr[:, :3]

def simulate_pred_spectrum(
    tokens_or_probs: torch.Tensor,
    tmm: TMMSpectrum,
    wl_tensor: torch.Tensor,
    theta: torch.Tensor,
    eos: int, pad: int, msk: int,
    tau_for_soft: float = -1.0
) -> torch.Tensor:
    """
    Accepts either token IDs [1,S] (long) or logits/probs [1,S,V] (float).
    Returns predicted spectrum [W,3] on CPU.
    """
    # Reuse the same rules as in analyze_inference.py
    x = tokens_or_probs
    if x.dim() == 2 and x.dtype in (torch.long, torch.int64):
        res = tmm(x, wl_tensor, theta, eos=eos, pad=pad, msk=msk)
    elif x.dim() == 3:
        if tau_for_soft <= 0:
            hard  = torch.nn.functional.one_hot(x.argmax(-1), num_classes=x.shape[-1]).to(x.dtype)
            probs = torch.softmax(x, dim=-1)
            tokens_for_tmm = hard + probs - probs.detach()
        else:
            tokens_for_tmm = torch.softmax(x / tau_for_soft, dim=-1)
        res = tmm(tokens_for_tmm, wl_tensor, theta, eos=eos, pad=pad, msk=msk)
    else:
        raise ValueError("Unexpected shape for tokens/logits")
    if res.dim() == 3 and res.size(1) == 3:   # [1,3,W] -> [W,3]
        return res[0].permute(1, 0).real.detach().cpu()
    if res.dim() == 2:                        # [1,3W] -> [W,3]
        W = res.size(1) // 3
        return res[0].view(3, W).permute(1, 0).real.detach().cpu()
    raise RuntimeError(f"Unexpected TMM output shape {tuple(res.shape)}")

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive spectrum->stack inference for OptoLlama")
    ap.add_argument("--config", required=True, help="Config module name, e.g. config_MD44")
    ap.add_argument("--checkpoint", default="auto", help="'auto'|'none'|path/to/ckpt.pt")
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    ap.add_argument("--echo_pred_spectrum", action="store_true",
                    help="Also simulate the predicted spectrum via TMM and report error vs input.")
    ap.add_argument("--input", default=None,
                    help="Optional file to load a single spectrum (npy|json|csv|txt). If omitted, REPL mode.")
    args = ap.parse_args()

    # Load config + device
    cfg = importlib.import_module(args.config)
    device = pick_device(args.device)

    # Token tables
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos, pad, msk = init_tokenmaps(cfg.PATH_DATA)

    # Use the validation dataset only to infer dims and sample spectrum shape
    val_ds = dataset.SpectraDataset(cfg.PATH_VALID, tokens, device='cpu')
    # spectrum_sample: torch.Tensor [W,3] (on CPU)
    spectrum_sample, _, _ = val_ds[0]
    W = spectrum_sample.shape[0]
    vocab_size = len(tokens)
    max_stack_depth = val_ds.get_maximum_depth()

    # Build model
    model = build_model(
        model_type=getattr(cfg, 'ARCH', getattr(cfg, 'OL_MODEL', 'dit')),
        spectrum_dim=spectrum_sample.shape[-1],
        vocab_size=vocab_size,
        timesteps=cfg.STEPS,
        max_len=W,
        max_stack_depth=max_stack_depth,
        mask_idx=msk,
        d_model=cfg.D_MODEL,
        n_blocks=cfg.N_BLOCKS,
        n_heads=cfg.N_HEADS,
        dropout=cfg.DROPOUT,
        idx_to_token=idx_to_token,
        pad_idx=pad,
        eos_idx=eos,
        device=str(device),
        sample_spectrum=spectrum_sample.unsqueeze(0),
    ).to(torch.float32).to(device)

    # Load checkpoint
    ckpt_path = None
    if args.checkpoint == 'auto':
        ckpt_path = os.path.join(cfg.PATH_SAVED, 'ol3l-checkpoint.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = None
    elif args.checkpoint != 'none':
        ckpt_path = args.checkpoint

    if ckpt_path:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = load_state_dict_flexible(model, state['model_state'])
        print(f"[info] Loaded checkpoint: {ckpt_path}")
    else:
        print("[warn] No checkpoint loaded (random weights).")

    # Optional TMM for echoing predicted spectrum
    if args.echo_pred_spectrum:
        tmm, wl_tensor, theta = build_tmm(cfg, device, idx_to_token)
    else:
        tmm = wl_tensor = theta = None

    model.eval()
    torch.set_grad_enabled(False)

    def infer_one(spec_np: np.ndarray):
        if spec_np.shape != (W, 3):
            raise ValueError(f"Spectrum must have shape ({W}, 3) [R,A,T]. Got {spec_np.shape}.")
        spec = torch.tensor(spec_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,W,3]
        logits, _ = model(spec)  # [1,S,V]
        ids = logits_to_token_ids(logits)          # [1,S]
        toks = decode_until_eos(ids[0], eos, pad, msk, idx_to_token)
        print("\nPredicted stack:")
        print("  " + pretty_stack(toks))
        # Optional echo:
        if args.echo_pred_spectrum:
            pred_spec = simulate_pred_spectrum(logits, tmm, wl_tensor, theta, eos, pad, msk, tau_for_soft=-1.0).numpy()
            # simple errors
            mae = float(np.mean(np.abs(pred_spec - spec_np)))
            mse = float(np.mean((pred_spec - spec_np) ** 2))
            print(f"\nEcho TMM (pred spectrum from tokens):")
            print(f"  MAE={mae:.6f}  MSE={mse:.6e}")
        print("")

    # One-shot file input
    if args.input:
        spec = load_user_spectrum(args.input, W)
        infer_one(spec)
        return

    # REPL Mode
    print("\n--- OptoLlama Interactive Inference ---")
    print(f"Config: {args.config}  |  Device: {device}  |  Wavelength samples: {W}")
    print("Enter a spectrum file path (.npy/.json/.csv/.txt) OR press Enter to paste R/A/T lists.")
    print('Type "quit" to exit.\n')

    while True:
        try:
            line = input("spectrum> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not line:
            try:
                spec = load_user_spectrum(None, W)
            except Exception as e:
                print(f"[error] {e}")
                continue
        else:
            if line.lower() in ("q", "quit", "exit"): break
            if not os.path.exists(line):
                print(f"[error] File not found: {line}")
                continue
            try:
                spec = load_user_spectrum(line, W)
            except Exception as e:
                print(f"[error] {e}")
                continue
        try:
            infer_one(spec)
        except Exception as e:
            print(f"[error] inference failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # handy defaults for local tinkering — adjust as needed
        sys.argv.extend(["--config", "config_MD49", "--checkpoint", "auto", "--echo_pred_spectrum"])
    main()
