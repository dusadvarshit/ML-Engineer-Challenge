"""CPU-only tests for reusable training and detection preprocessing pipelines."""

from __future__ import annotations

import json
from io import BytesIO

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.data.detection_preprocessing import (
    normalize_coco_category_name,
    prepare_detr_inputs,
    prepare_retinanet_input,
    prepare_yolo_input,
    xyxy_to_xywh,
)
from models.training.classification_pipeline import (
    ClassificationRunConfig,
    TinyImageNetValidationDataset,
    accuracy_at_k,
    build_optimizer_and_scheduler,
    evaluate_classifier,
    fine_tune_one_epoch,
    resolve_tiny_imagenet_root,
    save_classification_artifact,
)

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    image = Image.new("RGB", (2, 3), color=(255, 128, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_resolve_tiny_imagenet_and_parse_annotation_validation_split(tmp_path) -> None:
    """Both dataset-root forms and official validation annotations are supported."""

    root = tmp_path / "tiny-imagenet-200"
    (root / "train").mkdir(parents=True)
    images = root / "val" / "images"
    images.mkdir(parents=True)
    (root / "val" / "val_annotations.txt").write_text("sample.png\tn0001\t0\t0\t1\t1\n")
    (images / "sample.png").write_bytes(_png_bytes())

    assert resolve_tiny_imagenet_root(tmp_path) == root
    dataset = TinyImageNetValidationDataset(
        root / "val", {"n0001": 7}, transform=lambda _: torch.ones(3, 2, 2)
    )

    image, label = dataset[0]
    assert image.shape == (3, 2, 2)
    assert label == 7


def test_accuracy_training_evaluation_and_artifact_write_are_cpu_safe(tmp_path) -> None:
    """Fine-tuning helpers work against tiny synthetic tensors and write serving files."""

    config = ClassificationRunConfig(epochs=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    images = torch.rand(4, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    train_metrics = fine_tune_one_epoch(model, loader, optimizer, config)
    scheduler.step()
    evaluation_metrics = evaluate_classifier(model, loader)
    artifact_dir = save_classification_artifact(
        model, ["a", "b", "c"], config, tmp_path, {**train_metrics, **evaluation_metrics}
    )

    assert train_metrics["train_loss"] > 0
    assert set(evaluation_metrics) == {"val_loss", "val_top1", "val_top5"}
    assert (artifact_dir / "best_model.pth").is_file()
    assert json.loads((artifact_dir / "classes.json").read_text()) == ["a", "b", "c"]
    assert accuracy_at_k(torch.tensor([[3.0, 1.0, 0.0]]), torch.tensor([0])) == {1: 1.0, 5: 1.0}


def test_detection_preprocessing_adapts_one_payload_for_all_backends() -> None:
    """YOLO, RetinaNet, and DETR preparation share a single safe RGB decode path."""

    payload = _png_bytes()
    assert prepare_yolo_input(payload).size == (2, 3)
    tensor = prepare_retinanet_input(payload)
    assert tensor.shape == (3, 3, 2)
    assert float(tensor.max()) <= 1.0

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images.mode == "RGB"
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones(1, 3, 2, 2), "meta": "keep"}

    detr_inputs = prepare_detr_inputs(payload, Processor())
    assert detr_inputs["pixel_values"].device.type == "cpu"
    assert detr_inputs["meta"] == "keep"
    assert xyxy_to_xywh([1, 2, 5, 7]) == [1.0, 2.0, 4.0, 5.0]
    assert normalize_coco_category_name("pottedplant") == "potted plant"
