"""Shared multi-model export entrypoint for ONNX and TensorRT artifacts."""

from __future__ import annotations

try:
    from models.export_common import ModelExporter, main_for_exporter, main_for_registry
    from models.export_detr_resnet50 import EXPORTER as DETR_EXPORTER
    from models.export_retinanet_resnet50_fpn import EXPORTER as RETINANET_EXPORTER
    from models.export_yolov8n import EXPORTER as YOLO_EXPORTER
except ModuleNotFoundError:
    from export_common import ModelExporter, main_for_exporter, main_for_registry
    from export_detr_resnet50 import EXPORTER as DETR_EXPORTER
    from export_retinanet_resnet50_fpn import EXPORTER as RETINANET_EXPORTER
    from export_yolov8n import EXPORTER as YOLO_EXPORTER


EXPORTERS: dict[str, ModelExporter] = {
    exporter.model_name: exporter
    for exporter in (
        YOLO_EXPORTER,
        DETR_EXPORTER,
        RETINANET_EXPORTER,
    )
}


def main_for_model(model_name: str, argv: list[str] | None = None) -> None:
    exporter = EXPORTERS[model_name]
    main_for_exporter(exporter, argv)


def main(argv: list[str] | None = None) -> None:
    main_for_registry(EXPORTERS, argv)


if __name__ == "__main__":
    main()
