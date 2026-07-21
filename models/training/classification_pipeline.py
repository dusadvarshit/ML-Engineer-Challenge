"""Reusable Tiny-ImageNet fine-tuning and inference-preparation pipeline.

The notebook experiment is intentionally split into importable pieces here so a
training run is reproducible in CI/CPU environments and writes serving-friendly
artifacts without embedding local paths in application code.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_CLASSIFIERS = ("resnet50", "efficientnet_b0", "vit_b_16")


@dataclass(frozen=True)
class ClassificationRunConfig:
    """Configurable choices used by a fine-tuning run."""

    model_name: str = "efficientnet_b0"
    version: str = "v1.0.0"
    image_size: int = 224
    epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    use_pretrained_weights: bool = False

    def validate(self) -> None:
        if self.model_name not in SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Unsupported classifier: {self.model_name}")
        if self.image_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size and epochs must be positive.")


class TinyImageNetValidationDataset(Dataset[tuple[Tensor, int]]):
    """Tiny-ImageNet's annotation-based validation split as a PyTorch dataset."""

    def __init__(
        self,
        validation_root: Path,
        class_to_idx: dict[str, int],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        annotation_file = validation_root / "val_annotations.txt"
        images_directory = validation_root / "images"
        if not annotation_file.is_file() or not images_directory.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet validation split requires images/ and val_annotations.txt."
            )
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for line in annotation_file.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Invalid Tiny-ImageNet validation annotation: {line!r}")
            image_name, class_name = fields[:2]
            if class_name not in class_to_idx:
                raise ValueError(f"Validation class {class_name!r} is absent from training classes.")
            self.samples.append((images_directory / image_name, class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        if self.transform is None:
            raise RuntimeError("A transform is required to produce training tensors.")
        return self.transform(rgb_image), label


def resolve_tiny_imagenet_root(root: str | Path) -> Path:
    """Resolve either the dataset directory or its parent without hard-coded paths."""

    candidate_root = Path(root)
    for candidate in (candidate_root, candidate_root / "tiny-imagenet-200"):
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to find Tiny-ImageNet train/val under {candidate_root}.")


def build_classification_transforms(image_size: int = 224) -> tuple[Callable, Callable]:
    """Return ImageNet-normalized train and serving transforms, imported lazily."""

    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ModuleNotFoundError as exc:
        raise RuntimeError("Classification transforms require torchvision.") from exc
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    inference_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, inference_transform


def build_classifier(config: ClassificationRunConfig, num_classes: int, device: str | torch.device = "cpu") -> nn.Module:
    """Build one notebook candidate with a replacement classification head."""

    config.validate()
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    try:
        from torchvision.models import (
            EfficientNet_B0_Weights,
            ResNet50_Weights,
            ViT_B_16_Weights,
            efficientnet_b0,
            resnet50,
            vit_b_16,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("Classifier construction requires torchvision.") from exc

    if config.model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if config.use_pretrained_weights else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config.model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if config.use_pretrained_weights else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if config.use_pretrained_weights else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(device)


def accuracy_at_k(logits: Tensor, labels: Tensor, top_k: Sequence[int] = (1, 5)) -> dict[int, float]:
    """Compute batch top-k accuracy while handling classifiers with fewer than k classes."""

    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError("logits must be [batch, classes] and labels must match its batch size.")
    max_k = min(max(top_k), logits.size(1))
    predictions = logits.topk(max_k, dim=1).indices.t()
    correct = predictions.eq(labels.reshape(1, -1))
    return {
        k: float(correct[: min(k, logits.size(1))].any(dim=0).float().mean().item())
        for k in top_k
    }


def fine_tune_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: AdamW,
    config: ClassificationRunConfig,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Run one mixed-precision-safe training epoch with gradient clipping."""

    resolved_device = torch.device(device)
    scaler = torch.amp.GradScaler("cuda", enabled=resolved_device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_total = 0.0
    count = 0
    for images, labels in loader:
        images, labels = images.to(resolved_device), labels.to(resolved_device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast("cuda") if resolved_device.type == "cuda" else nullcontext()
        with context:
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        loss_total += float(loss.item()) * labels.size(0)
        count += labels.size(0)
    if not count:
        raise ValueError("Training loader produced no batches.")
    return {"train_loss": loss_total / count}


@torch.inference_mode()
def evaluate_classifier(model: nn.Module, loader: Any, device: str | torch.device = "cpu") -> dict[str, float]:
    """Evaluate loss and top-1/top-5 accuracy without requiring a GPU."""

    resolved_device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "count": 0.0}
    for images, labels in loader:
        images, labels = images.to(resolved_device), labels.to(resolved_device)
        logits = model(images)
        scores = accuracy_at_k(logits, labels)
        batch_size = labels.size(0)
        totals["loss"] += float(criterion(logits, labels).item()) * batch_size
        totals["top1"] += scores[1] * batch_size
        totals["top5"] += scores[5] * batch_size
        totals["count"] += batch_size
    if not totals["count"]:
        raise ValueError("Evaluation loader produced no batches.")
    return {
        "val_loss": totals["loss"] / totals["count"],
        "val_top1": totals["top1"] / totals["count"],
        "val_top5": totals["top5"] / totals["count"],
    }


def save_classification_artifact(
    model: nn.Module,
    classes: Sequence[str],
    config: ClassificationRunConfig,
    output_root: str | Path,
    metrics: dict[str, float],
) -> Path:
    """Persist checkpoint, class mapping, config, and metrics in the serving layout."""

    config.validate()
    artifact_dir = Path(output_root) / config.model_name / config.version / "pytorch"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": config.model_name, "version": config.version, "state_dict": model.state_dict()},
        artifact_dir / "best_model.pth",
    )
    (artifact_dir / "classes.json").write_text(json.dumps(list(classes), indent=2))
    (artifact_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2))
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return artifact_dir


def build_optimizer_and_scheduler(model: nn.Module, config: ClassificationRunConfig) -> tuple[AdamW, CosineAnnealingLR]:
    """Use notebook-aligned AdamW and cosine scheduling for fine-tuning."""

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return optimizer, CosineAnnealingLR(optimizer, T_max=config.epochs)
