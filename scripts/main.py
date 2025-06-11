# main.py
"""
Central entry-point – trains or tests depending on the first CLI argument.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType
from typing import Mapping, Tuple
import os, shutil, json, inspect
from train import OpticalGPTTrainer
from evaluator import OpticalGPTEvaluator
from utils import load_JSONPICKLE, seed_everything
import importlib

def _prepare_token_maps(
        cfg: ModuleType,                        # NEW: the config module
    ) -> Tuple[
        Mapping[int, str],                      #       idx  → token
        Mapping[str, int],                      #       material → id
        int, int, int,                          #       pad_idx, sos_idx, eos_idx
    ]:
    tokens = load_JSONPICKLE(cfg.PATH_DATA, "tokens")
    for s in ("<PAD>", "<SOS>", "<EOS>"):
        if s not in tokens:
            tokens.append(s)

    token_to_idx = {tk: i for i, tk in enumerate(tokens)}
    pad_idx = token_to_idx["<PAD>"]
    sos_idx = token_to_idx["<SOS>"]
    eos_idx = token_to_idx["<EOS>"]

    idx_to_token = {i: tk for tk, i in token_to_idx.items()}
    materials = sorted(
        {tk.split("__")[0] for tk in tokens if tk not in ("<PAD>", "<SOS>", "<EOS>")}
    )
    mat2id = {m: i for i, m in enumerate(materials)}
    return idx_to_token, mat2id, pad_idx, sos_idx, eos_idx

# ------------------------------------------------------------
# main.py  – simplified entry
# ------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Return the top-level CLI parser."""
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ----- train -----------------------------------------------------
    p_train = sub.add_parser("train", help="Start multi-epoch training")
    p_train.add_argument("--pt",  help="train *.pt file; default comes from cfg")
    # p_train.add_argument("--cfg", default="config_A00",    # NEW
    #                      help="import path of the config module")
    p_train.add_argument(
        "--cfg",
        default="config_A00",
        help="either a module import path (e.g. config_A00) or an absolute/relative path to a .py file",
    )

    # ----- test ------------------------------------------------------
    p_test = sub.add_parser("test",  help="Evaluate a checkpoint")
    p_test.add_argument("--ckpt", required=True, help="*.pth checkpoint to load")
    p_test.add_argument("--mode", choices=["simple", "ray"], default="simple")
    p_test.add_argument("--pt_val",  help="validation *.pt; default from cfg")
    # p_test.add_argument("--cfg", default="config_A00")      # NEW
    p_test.add_argument(
        "--cfg",
        default="config_A00",
        help="either a module import path or a path to a .py file",
    )

    return p

def _main(argv: list[str] | None = None,
          cfg: ModuleType | None = None) -> None:
    """
    Entry-point callable from shell *and* other Python code.

    • argv None → use sys.argv[1:]
    • cfg  None → import module named in --cfg / default
    """
    args = _build_parser().parse_args(argv)

    # ---- resolve configuration ----------------------------------
    if cfg is None:
        cfg_arg = args.cfg
        cfg_path = Path(cfg_arg)
        if cfg_path.is_file() and cfg_path.suffix == ".py":
            # load from a .py file on disk
            spec = importlib.util.spec_from_file_location(cfg_path.stem, str(cfg_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            cfg = module
        else:
            # load via import path
            cfg = importlib.import_module(cfg_arg)

    # ─── New: snapshot config & args ────────────────────────────────
    run_dir = Path(cfg.PATH_RUN)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) copy the actual config .py file into the run folder
    cfg_path = Path(inspect.getsourcefile(cfg))
    shutil.copy(cfg_path, run_dir / cfg_path.name)

    # 2) dump the parsed args as JSON for easy reading later
    args_dict = vars(args)
    with open(run_dir / "run_args.json", "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # ---------- resolve dataset paths ------------------------------
    train_pt = Path(cfg.PT_TRAIN or getattr(cfg, "PT_TRAIN", "train.pt"))
    val_pt   = Path(cfg.PT_VAL or getattr(cfg, "PT_VAL",   "val.pt"))

    if not train_pt.is_file():
        raise FileNotFoundError(f"train file {train_pt} not found")
    if not val_pt.is_file():
        raise FileNotFoundError(f"val file   {val_pt} not found")
    
    seed_everything(42)
    
    idx2tk, mat2id, pad_idx, sos_idx, eos_idx = _prepare_token_maps(cfg)
    
    # ---------- dispatch -------------------------------------------
    if args.cmd == "train":
        trainer = OpticalGPTTrainer(
            cfg=cfg,
            num_classes=len(mat2id) + 3,
            d_model=1024,
            n_heads=8,
            n_layers=6,
            max_seq_length=22,
            input_dim=3 * 171,
            dropout=0.1,
            epsilon=0.1,
            pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pt_file=train_pt,
            token_string_map=idx2tk,
            material_to_id=mat2id,
        )
        trainer.train()
    else:   # test
        evaluator = OpticalGPTEvaluator(
            cfg=cfg,
            ckpt_file=args.ckpt,
            pt_file_val=val_pt,
            token_string_map=idx2tk,
            material_to_id=mat2id,
            pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            max_seq_length=22,#22
        )
        
        if args.mode == "ray":
            evaluator.export_results_per_sample(out_name="val_sample_results")
        else:
            v_loss, v_ce, v_mse, v_acc = evaluator.evaluate_simple()
            print(f"✅  val_loss={v_loss:.5f}  acc={v_acc:.3%}")

if __name__ == "__main__":
    _main()
