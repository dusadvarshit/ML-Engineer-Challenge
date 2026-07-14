"""YOLOv8n export implementation."""

from __future__ import annotations

from pathlib import Path

try:
    from models.export_common import (
        ExportOptions,
        ExportSource,
        TensorRTShapeProfile,
        main_for_exporter,
    )
    from models.export_utils import ArtifactLayout, move_file, resolve_checkpoint
except ModuleNotFoundError:
    from export_common import (
        ExportOptions,
        ExportSource,
        TensorRTShapeProfile,
        main_for_exporter,
    )
    from export_utils import ArtifactLayout, move_file, resolve_checkpoint


class YoloExporter:
    model_name = "yolov8n"
    description = "Export the YOLOv8n PyTorch checkpoint to ONNX and TensorRT."
    default_image_size = 640
    default_opset = 17

    def resolve_source(self, layout: ArtifactLayout, checkpoint: Path | None) -> ExportSource:
        checkpoint_path = checkpoint or resolve_checkpoint(
            layout.pytorch_dir,
            allowed_suffixes=(".pt", ".pth"),
            preferred_names=("yolov8n.pt",),
        )
        return ExportSource(path=checkpoint_path, stem=checkpoint_path.stem)

    def export_to_onnx(
        self,
        *,
        source: ExportSource,
        layout: ArtifactLayout,
        onnx_path: Path,
        options: ExportOptions,
    ) -> Path:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO export. Install the `ultralytics` package first."
            ) from exc

        model = YOLO(str(source.path))
        exported = model.export(
            format="onnx",
            imgsz=options.image_size,
            dynamic=True,
            simplify=False,
            opset=options.opset,
            device=options.device,
        )

        exported_path = Path(str(exported)) if exported is not None else source.path.with_suffix(".onnx")
        if not exported_path.exists():
            fallback_path = layout.pytorch_dir / f"{source.stem}.onnx"
            if not fallback_path.exists():
                raise FileNotFoundError(
                    f"Ultralytics did not produce an ONNX file for {source.path.name}."
                )
            exported_path = fallback_path

        if exported_path.resolve() != onnx_path.resolve():
            move_file(exported_path, onnx_path)
        return onnx_path

    def build_tensorrt_shape_profile(self, image_size: int) -> TensorRTShapeProfile:
        image_shape = f"1x3x{image_size}x{image_size}"
        return TensorRTShapeProfile(
            min_shapes={"images": image_shape},
            opt_shapes={"images": image_shape},
            max_shapes={"images": image_shape},
        )


EXPORTER = YoloExporter()


if __name__ == "__main__":
    main_for_exporter(EXPORTER)
