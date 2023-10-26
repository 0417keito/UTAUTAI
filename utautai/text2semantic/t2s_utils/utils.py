import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Sampler
from torch.cuda.amp.grad_scaler import GradScaler
import collections
import logging
import omegaconf
import uuid
import glob
import re
import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import argparse

LRSchedulerType = object
_lhotse_uuid = None

class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")
    
class MetricsTracker(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
            ans.append((k, norm_value))
        return ans

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d    
    
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)
    
def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "cur_epoch":0,
            "log_interval": 100,  # 10: debug 100: train
            "reset_interval": 200,
            "valid_interval": 10000,
        }
    )

    return params

def uuid4():
    if _lhotse_uuid is not None:
        return _lhotse_uuid()
    return uuid.uuid4()

def set_batch_count(model: nn.Module, batch_count: float) -> None:

    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
            
def fix_random_seed(random_seed: int):
    global _lhotse_uuid
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    # Ensure deterministic ID creation
    rd = random.Random()
    rd.seed(random_seed)
    _lhotse_uuid = lambda: uuid.UUID(int=rd.getrandbits(128))

def display_and_save_batch(
    batch: dict,
    exp_dir,
) -> None:

    filename = f"{exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)
    
def save_checkpoint_impl(
    filename: Path,
    model: nn.Module,
    training_dict: AttributeDict,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler = None,
):

    logging.info(f"Saving checkpoint to {filename}")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }

    if training_dict:
        for k, v in training_dict.items():
            assert k not in checkpoint
            checkpoint[k] = v

    torch.save(checkpoint, filename)
    
def save_checkpoint_with_global_batch_idx(
    out_dir: Path,
    global_batch_idx: int,
    model: nn.Module,
    training_dict: AttributeDict,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[Sampler] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"checkpoint-{global_batch_idx}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        training_dict=training_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
    )
    
def find_checkpoints(out_dir: Path, iteration: int = 0) -> List[str]:
    checkpoints = list(glob.glob(f"{out_dir}/checkpoint-[0-9]*.pt"))
    pattern = re.compile(r"checkpoint-([0-9]+).pt")
    iter_checkpoints = []
    for c in checkpoints:
        result = pattern.search(c)
        if not result:
            logging.warn(f"Invalid checkpoint filename {c}")
            continue

        iter_checkpoints.append((int(result.group(1)), c))

    # iter_checkpoints is a list of tuples. Each tuple contains
    # two elements: (iteration_number, checkpoint-iteration_number.pt)

    iter_checkpoints = sorted(iter_checkpoints, reverse=True, key=lambda x: x[0])
    if iteration >= 0:
        ans = [ic[1] for ic in iter_checkpoints if ic[0] >= iteration]
    else:
        ans = [ic[1] for ic in iter_checkpoints if ic[0] <= -iteration]

    return ans
    
def remove_checkpoints(
    out_dir: Path,
    topk: int,
):
    assert topk >= 1, topk
    
    checkpoints = find_checkpoints(out_dir)

    if len(checkpoints) == 0:
        logging.warn(f"No checkpoints found in {out_dir}")
        return

    if len(checkpoints) <= topk:
        return

    to_remove = checkpoints[topk:]
    for c in to_remove:
        os.remove(c)
        
def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")