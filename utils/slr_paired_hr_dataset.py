"""Paired low/high-resolution dataset for GEOTR-SLR.

This wrapper intentionally delegates *both* LR and HR sample construction to
whatever ``datasets.dataloader`` is present in the target project.  This is
important because the user's working tree may carry audited preprocessing
changes that differ from a bundled/public loader by tiny interpolation,
normalisation, colour-order, or mask-conversion details.

LR therefore remains exactly the project's historical data path.  HR is a
second invocation of the same DatasetSegmentation pipeline with a larger
``image_size`` so it is produced from the original source file, never by
upsampling the LR tensor.  During training we replay Python/NumPy/Torch RNG
states for the HR invocation and restore the post-LR states afterwards.  Thus
stochastic spatial augmentation is identical across LR and HR without
advancing the training RNG twice.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator


def _build_random_generator(size: int, cfg=None):
    """Construct the project's RandomGenerator across historical signatures."""
    out = [int(size), int(size)]
    try:
        # The current loader reads augmentation probabilities/ranges from cfg.
        # LR and HR must receive the same cfg as well as the same RNG replay;
        # otherwise the paired branches can silently use different defaults.
        return RandomGenerator(out, cfg=cfg)
    except TypeError:
        try:
            # Historical project variants made cfg positional.
            return RandomGenerator(out, cfg)
        except TypeError:
            # Oldest signature accepted output_size only.
            return RandomGenerator(out)


def _build_val_generator(size: int):
    return ValGenerator([int(size), int(size)])


def _capture_rng_state():
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
    }


def _restore_rng_state(state) -> None:
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.random.set_rng_state(state["torch"])


class SLRPairedResolutionDataset(Dataset):
    """Historical LR sample plus an aligned true-HR view.

    The LR branch is not reimplemented here.  It is the exact current-project
    ``DatasetSegmentation`` path.  HR runs the same path from the source image
    with ``image_size=hr_size``.  This makes the wrapper robust to local audited
    loader changes while keeping Base/Geometry input semantics unchanged.
    """

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text,
        *,
        image_size: int = 224,
        hr_size: int = 448,
        training: bool = False,
        random_rotate_probability: float = 0.5,
        random_rotate_degrees: int = 20,
        post_transform: Optional[Callable] = None,
        cfg=None,
    ) -> None:
        del random_rotate_probability, random_rotate_degrees  # delegated to project loader
        self.image_size = int(image_size)
        self.hr_size = int(hr_size)
        self.training = bool(training)
        self.post_transform = post_transform
        self.cfg = cfg
        if self.image_size <= 0 or self.hr_size <= self.image_size:
            raise ValueError("SLR paired dataset requires 0 < image_size < hr_size")
        if self.hr_size % self.image_size != 0:
            raise ValueError("SLR HR_SIZE must be an integer multiple of DATASET.SIZE")

        lr_transform = _build_random_generator(self.image_size, self.cfg) if self.training else _build_val_generator(self.image_size)
        hr_transform = _build_random_generator(self.hr_size, self.cfg) if self.training else _build_val_generator(self.hr_size)

        # Crucial: both branches use the target project's real loader.  Only
        # image_size differs, so HR is sourced directly from the original file.
        self.lr_dataset = DatasetSegmentation(
            dataset_path,
            task_name,
            row_text,
            lr_transform,
            image_size=self.image_size,
        )
        self.hr_dataset = DatasetSegmentation(
            dataset_path,
            task_name,
            row_text,
            hr_transform,
            image_size=self.hr_size,
        )

    def __len__(self) -> int:
        return len(self.lr_dataset)

    def __getitem__(self, idx: int):
        if len(self.lr_dataset) != len(self.hr_dataset):
            raise RuntimeError("SLR paired LR/HR datasets have different lengths")

        # LR is the canonical historical path and advances RNG exactly once.
        before = _capture_rng_state()
        lr = self.lr_dataset[idx]
        after_lr = _capture_rng_state()

        # Replay the identical stochastic transform for HR, then restore the
        # state that followed LR so the wrapper has no extra RNG side effect.
        _restore_rng_state(before)
        try:
            hr = self.hr_dataset[idx]
        finally:
            _restore_rng_state(after_lr)

        if lr.get("image_name", None) != hr.get("image_name", None):
            raise RuntimeError("SLR paired LR/HR image identity mismatch")
        if "image" not in lr or "image" not in hr:
            raise KeyError("project dataloader must return an 'image' tensor")
        if not torch.is_tensor(lr["image"]) or not torch.is_tensor(hr["image"]):
            raise TypeError("SLR true-HR path requires tensor images after project transforms")
        if tuple(lr["image"].shape[-2:]) != (self.image_size, self.image_size):
            raise RuntimeError(f"unexpected LR shape: {tuple(lr['image'].shape)}")
        if tuple(hr["image"].shape[-2:]) != (self.hr_size, self.hr_size):
            raise RuntimeError(f"unexpected HR shape: {tuple(hr['image'].shape)}")
        if "ground_truth_mask" not in lr or "ground_truth_mask" not in hr:
            raise KeyError("project dataloader must return a 'ground_truth_mask' tensor")
        if not torch.is_tensor(hr["ground_truth_mask"]):
            raise TypeError("SPARC-HR requires a tensor HR ground-truth mask")
        if tuple(hr["ground_truth_mask"].shape[-2:]) != (self.hr_size, self.hr_size):
            raise RuntimeError(
                f"unexpected HR mask shape: {tuple(hr['ground_truth_mask'].shape)}"
            )

        sample = dict(lr)
        sample["image_hr"] = hr["image"]
        sample["ground_truth_mask_hr"] = hr["ground_truth_mask"]
        sample["slr_true_hr"] = torch.tensor(1, dtype=torch.uint8)
        if self.post_transform is not None:
            sample = self.post_transform(sample)
        return sample
