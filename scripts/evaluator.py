# evaluator.py – stand‑alone validation, analysis & export for Optical‑GPT
# =====================================================================
#  * Loads DataParallel / DDP checkpoints safely (strips `module.`)
#  * evaluate_simple()     – token CE / MSE / accuracy (quick pass)
#  * evaluate_rayflare()   – optical MAE via RayFlare
#  * export_results_per_sample()  – NEW: dumps JSON‑pickle with
#        target_seq, pred_seq, target_spectrum, pred_spectrum, accuracy, mae
#
#  The helper functions _decode_batch(), _reconstruct_sequence(),
#  and _sequence_accuracy() keep things tidy.

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union, Optional
import functools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import config_GPT30N37 as c
from data import SinglePTDataset, pad_collate_fn, create_masks
from model import OptoLlama
from mcd_inference import aggregation


class OpticalGPTEvaluator:
    """Tiny wrapper around *OptoLlama* that only provides **inference** & analysis."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        cfg,
        ckpt_file: Union[str, Path],
        pt_file_val: Union[str, Path],
        token_string_map: Mapping[int, str],
        material_to_id: Mapping[str, int],
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
        max_seq_length: int,
        batch_size: Optional[int] = None,
    ) -> None:
        self.cfg = cfg 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(material_to_id) + 3
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.max_seq_length = max_seq_length
        self.material_to_id = material_to_id
        self.token_string_map = token_string_map

        # ---- model ----------------------------------------------------
        self.model = OptoLlama(
            num_classes=self.num_classes,
            d_model=768,
            n_heads=8,
            n_layers=6,
            max_seq_length=max_seq_length,
            input_dim=3 * 171,
            dropout=0.2,
        ).to(self.device)

        # ---- checkpoint ----------------------------------------------
        ckpt_file = Path(ckpt_file)
        if not ckpt_file.is_file():
            raise FileNotFoundError(ckpt_file)
        map_loc = self._make_map_location()
        raw = torch.load(str(ckpt_file), map_location=map_loc)
        state_dict = self._extract_state_dict(raw)
        cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

        # ---- dataloader ----------------------------------------------
        ds_val = SinglePTDataset(str(pt_file_val))
        collate = functools.partial(
            pad_collate_fn,
            max_seq_length=max_seq_length,
            pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            token_string_map=token_string_map,
            material_to_id=material_to_id,
        )
        self.val_dl = DataLoader(
            ds_val,
            batch_size=batch_size if batch_size is not None else cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
            pin_memory=True,
        )

        self._mse = torch.nn.MSELoss(reduction="mean")
        self.epsilon = 0.1  # label smoothing

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_map_location(self):
        if self.device.type == "cuda":
            idx = self.device.index or 0
            return {"cuda:0": f"cuda:{idx}"}
        return "cpu"

    @staticmethod
    def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in payload and isinstance(payload[k], dict):
                    return payload[k]
        if isinstance(payload, dict):
            return payload
        raise TypeError("Checkpoint format not recognised – expected dict of tensors.")

    # ------------------------------------------------------------------
    # quick val pass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_simple(self) -> Tuple[float, float, float, float]:
        """Return `(loss, ce, mse, token_acc)` averaged over the *val* set."""
        tot_loss = tot_ce = tot_mse = 0.0
        correct = seen = 0

        for spec, mat_seq, thick_seq in self.val_dl:
            spec, mat_seq, thick_seq = (
                spec.to(self.device, dtype=torch.float32),
                mat_seq.to(self.device),
                thick_seq.to(self.device),
            )
            B = spec.size(0)
            input_seq = mat_seq[:, :1]  # <SOS>
            finished = torch.zeros(B, dtype=torch.bool, device=self.device)
            preds, ce_steps, mse_steps = [], [], []

            for _ in range(self.max_seq_length - 1):
                attn_mask, pad_mask = create_masks(input_seq, self.pad_idx, self.model.n_heads)
                attn_mask, pad_mask = attn_mask.to(self.device), pad_mask.to(self.device)
                mat_logit, thick_logit = self.model(spec, input_seq, attn_mask=attn_mask, key_padding_mask=pad_mask)
                pred_tok = mat_logit[:, -1].argmax(-1)
                preds.append(pred_tok)

                tgt_tok = mat_seq[:, input_seq.size(1)]
                tgt_thk = thick_seq[:, input_seq.size(1)]
                gold_1hot = F.one_hot(tgt_tok, self.num_classes).float()
                smoothed = gold_1hot * (1 - self.epsilon) + self.epsilon / self.num_classes
                ce_steps.append(-(smoothed * F.log_softmax(mat_logit[:, -1], -1)).sum(-1).mean())
                mse_steps.append(self._mse(thick_logit[:, -1] / 50.0, tgt_thk / 50.0)/4)

                finished |= pred_tok.eq(self.eos_idx)
                if finished.all():
                    break
                input_seq = torch.cat((input_seq, tgt_tok.unsqueeze(1)), dim=1)

            batch_ce = torch.stack(ce_steps).mean()
            batch_mse = torch.stack(mse_steps).mean()
            batch_loss = batch_ce + batch_mse

            tot_loss += batch_loss.item()
            tot_ce += batch_ce.item()
            tot_mse += batch_mse.item()

            pred_mat = torch.stack(preds, dim=1)
            gold_mat = mat_seq[:, 1 : 1 + pred_mat.size(1)]
            mask = gold_mat.ne(self.pad_idx)
            correct += pred_mat[mask].eq(gold_mat[mask]).sum().item()
            seen += mask.sum().item()

        n_batches = len(self.val_dl)
        return (
            tot_loss / n_batches if n_batches else 0.0,
            tot_ce / n_batches if n_batches else 0.0,
            tot_mse / n_batches if n_batches else 0.0,
            correct / seen if seen else 0.0,
        )

    # ------------------------------------------------------------------
    # NEW: per‑sample export
    # ------------------------------------------------------------------
    def export_results_per_sample(self, *, out_name: str = "results_per_sample", enableMCD: bool = False) -> None:
        """Create a JSON‑pickle with detailed results for every validation item."""
        from utils import save_JSONPICKLE
        from call_rayflare import DBinitNewMats, Call_RayFlare_with_dict
        import numpy as np

        DBinitNewMats(self.cfg)
        mats_sorted = sorted(self.material_to_id.keys())
        pad_tok, sos_tok, eos_tok = len(mats_sorted), len(mats_sorted) + 1, len(mats_sorted) + 2

        results: List[Dict[str, Any]] = []
        if enableMCD:
            self.model.train() 
        else:
            self.model.eval()

        with torch.no_grad():
            for spectra, mat_seq, thick_seq in self.val_dl:
                B = spectra.size(0)
                spectra_gpu = spectra.to(self.device, dtype=torch.float32)
                if enableMCD:
                    pred_seqs = []
                    for i in range(10):
                        pred_seqs.append(self._decode_batch(spectra_gpu))  # list[list[list[str]]]
                    pred_seqs = aggregation(pred_seqs)
                else:
                    pred_seqs = self._decode_batch(spectra_gpu)

                # ⇢ predicted spectra via RayFlare
                pred_specs = [np.asarray(Call_RayFlare_with_dict(self.cfg, st), dtype=np.float32).reshape(-1)
                               for st in pred_seqs]
                tgt_specs = spectra.cpu().numpy().reshape(B, -1)

                mat_seq = mat_seq.cpu()
                thick_seq = thick_seq.cpu()

                for b in range(B):
                    tgt_tokens = self._reconstruct_sequence(mat_seq[b], thick_seq[b], mats_sorted, pad_tok, sos_tok, eos_tok)
                    pred_tokens = pred_seqs[b]
                    acc = self._sequence_accuracy(tgt_tokens, pred_tokens)
                    mae = float(np.abs(pred_specs[b] - tgt_specs[b]).mean())

                    results.append({
                        "target_seq": np.array(tgt_tokens),
                        "pred_seq": np.array(pred_tokens),
                        "target_spectrum": tgt_specs[b],
                        "pred_spectrum": pred_specs[b],
                        "accuracy": acc,
                        "mae": mae,
                    })
        
        out_name = f'val_sample_results_{self.cfg.TARGET}_{self.cfg.RUN_NAME}_E{self.cfg.RESUME_EPOCH}'
        save_JSONPICKLE(self.cfg.PATH_RUN, results, out_name)
        print(f"💾 Saved {len(results)} samples → {self.cfg.PATH_RUN}/{out_name}.json")

    # ------------------------------------------------------------------
    # internal utilities
    # ------------------------------------------------------------------
    def _decode_batch(self, spectra: torch.Tensor) -> List[List[str]]:
        mats_sorted = sorted(self.material_to_id.keys())
        pad_tok, sos_tok, eos_tok = len(mats_sorted), len(mats_sorted) + 1, len(mats_sorted) + 2

        B = spectra.size(0)
        input_seq = torch.full((B, 1), sos_tok, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        stacks: List[List[str]] = [[] for _ in range(B)]

        for _ in range(self.max_seq_length - 1):
            attn_mask, pad_mask = create_masks(input_seq, pad_tok, self.model.n_heads)
            attn_mask, pad_mask = attn_mask.to(self.device), pad_mask.to(self.device)
            mat_logit, thick_logit = self.model(spectra, input_seq, attn_mask=attn_mask, key_padding_mask=pad_mask)
            tok_pred = mat_logit[:, -1].argmax(-1)
            th_pred = thick_logit[:, -1]

            for b in range(B):
                if not finished[b] and tok_pred[b] < len(mats_sorted):
                    stacks[b].append(f"{mats_sorted[tok_pred[b]]}__{th_pred[b].item():.1f}")
            finished |= tok_pred.eq(eos_tok)
            if finished.all():
                break
            input_seq = torch.cat((input_seq, tok_pred.unsqueeze(1)), dim=1)
        return stacks

    @staticmethod
    def _reconstruct_sequence(mat_ids: torch.Tensor, thicks: torch.Tensor, mats_sorted: List[str], pad_tok: int, sos_tok: int, eos_tok: int) -> List[str]:
        seq: List[str] = []
        for mid, th in zip(mat_ids.tolist(), thicks.tolist()):
            if mid == pad_tok:
                continue
            if mid == sos_tok:
                continue
            if mid == eos_tok:
                seq.append("<EOS>")
                break
            seq.append(f"{mats_sorted[mid]}__{th:.1f}")
        return seq

    @staticmethod
    def _sequence_accuracy(target: List[str], pred: List[str]) -> float:
        def _strip(seq):
            return [t for t in seq if t not in ("<PAD>", "<SOS>", "<EOS>")]
        tgt = _strip(target)
        prd = _strip(pred)
        if not tgt:
            return 0.0
        correct = sum(1 for t, p in zip(tgt, prd) if t.split("__")[0] == p.split("__")[0])
        return correct / len(tgt)
