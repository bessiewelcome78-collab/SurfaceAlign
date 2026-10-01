"""Dataset loaders with an exact paper-compatible training path.

The official transform is intentionally unchanged: OpenCV BGR input, a 50%
random rotation in [-20, 20], resize to 224, ImageNet normalization, and a
nearest-neighbour binary mask. The extra InferenceDataset never opens a
ground-truth file, so Test prediction cannot fail merely because a prompt
workbook uses a different mask suffix.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


CLIP_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def to_long_tensor(pic):
    return torch.from_numpy(np.asarray(pic, dtype=np.uint8)).long()


def correct_dims(*images):
    result = [np.expand_dims(x, axis=2) if x.ndim == 2 else x for x in images]
    return result[0] if len(result) == 1 else result


def _required_text(row, key, row_index):
    value = row.get(key) if isinstance(row, Mapping) else None
    if value is None or not str(value).strip():
        raise ValueError(f"Prompt row {row_index} has an empty {key!r} field")
    return str(value).strip()


def _read_image(path, image_size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Cannot decode image: {path}. Run tools/audit_dataset.py before training."
        )
    return cv2.resize(image, (image_size, image_size))


def _read_mask(path, image_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(
            f"Cannot decode ground-truth mask: {path}. Run tools/audit_dataset.py; "
            "for Test inference use InferenceDataset rather than reading labels."
        )
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask >= 127).astype(np.uint8)


def random_rotate(image, label):
    angle = random.randint(-20, 20)
    return image.rotate(angle), label.rotate(angle)


class ValGenerator:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image = sample["image"]
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))
        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        sample["image"] = CLIP_NORMALIZE(F.to_tensor(image))

        if "ground_truth_mask" in sample:
            mask = sample["ground_truth_mask"]
            if isinstance(mask, np.ndarray):
                if mask.ndim == 3 and mask.shape[2] == 1:
                    mask = np.squeeze(mask, axis=2)
                mask = Image.fromarray(mask.astype(np.uint8))
            if mask.size != self.output_size:
                mask = mask.resize(self.output_size, resample=Image.NEAREST)
            sample["ground_truth_mask"] = to_long_tensor(mask)
        return sample


class RandomGenerator:
    """Original augmentation, plus opt-in strong flips outside official mode."""

    def __init__(self, output_size, cfg=None):
        self.output_size = tuple(output_size)
        train_cfg = _cfg_get(cfg, "TRAIN", None)
        self.strong_augmentation = bool(_cfg_get(train_cfg, "USE_STRONG_AUG", False))

    def __call__(self, sample):
        image, mask = sample["image"], sample["ground_truth_mask"]
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if random.random() > 0.5:
            image, mask = random_rotate(image, mask)
        if self.strong_augmentation:
            if random.random() > 0.5:
                image, mask = F.hflip(image), F.hflip(mask)
            if random.random() > 0.5:
                image, mask = F.vflip(image), F.vflip(mask)

        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)
        sample["image"] = CLIP_NORMALIZE(F.to_tensor(image))
        sample["ground_truth_mask"] = to_long_tensor(mask)
        return sample


class DatasetSegmentation(Dataset):
    """Labelled loader for Train and Validation with fail-fast diagnostics."""

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: Sequence[Mapping],
        joint_transform: Callable | None = None,
        one_hot_mask: int = False,
        image_size: int = 224,
    ) -> None:
        self.image_size = int(image_size)
        self.input_path = os.path.join(dataset_path, "img")
        self.output_path = os.path.join(dataset_path, "label")
        self.one_hot_mask = int(one_hot_mask)
        self.task_name = task_name
        self.joint_transform = joint_transform or (lambda x: x)
        self.data_pairs = sorted(
            [
                (
                    _required_text(row, "Image", index),
                    _required_text(row, "Ground Truth", index),
                    _required_text(row, "Description", index),
                )
                for index, row in enumerate(row_text)
            ],
            key=lambda x: x[0],
        )

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_filename, mask_filename, text = self.data_pairs[idx]
        image = _read_image(os.path.join(self.input_path, image_filename), self.image_size)
        mask = _read_mask(os.path.join(self.output_path, mask_filename), self.image_size)
        image, mask = correct_dims(image, mask)
        if self.one_hot_mask:
            mask = torch.zeros(
                (self.one_hot_mask, mask.shape[0], mask.shape[1]), dtype=torch.float32
            ).scatter_(0, torch.as_tensor(mask).long().unsqueeze(0), 1)
        return self.joint_transform(
            {
                "image": image,
                "ground_truth_mask": mask,
                "image_name": image_filename,
                "mask_name": mask_filename,
                "text_prompt": text,
                "dataset_name": self.task_name,
            }
        )


class InferenceDataset(Dataset):
    """Unlabelled Test loader; mask names are identifiers, not files to read."""

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: Sequence[Mapping],
        joint_transform: Callable | None = None,
        image_size: int = 224,
    ) -> None:
        self.image_size = int(image_size)
        self.input_path = os.path.join(dataset_path, "img")
        self.task_name = task_name
        self.joint_transform = joint_transform or ValGenerator((image_size, image_size))
        self.data_pairs = sorted(
            [
                (
                    _required_text(row, "Image", index),
                    _required_text(row, "Ground Truth", index),
                    _required_text(row, "Description", index),
                )
                for index, row in enumerate(row_text)
            ],
            key=lambda x: x[0],
        )

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_filename, mask_filename, text = self.data_pairs[idx]
        image = _read_image(os.path.join(self.input_path, image_filename), self.image_size)
        return self.joint_transform(
            {
                "image": correct_dims(image),
                "image_name": image_filename,
                "mask_name": mask_filename,
                "text_prompt": text,
                "dataset_name": self.task_name,
            }
        )
