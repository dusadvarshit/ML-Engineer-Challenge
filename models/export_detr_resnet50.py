"""DETR ResNet-50 export implementation."""

from __future__ import annotations

from pathlib import Path

try:
    from models.export_common import (
        ExportOptions,
        ExportSource,
        TensorRTShapeProfile,
        main_for_exporter,
        validate_detr_directory,
    )
    from models.export_utils import ArtifactLayout
except ModuleNotFoundError:
    from export_common import (
        ExportOptions,
        ExportSource,
        TensorRTShapeProfile,
        main_for_exporter,
        validate_detr_directory,
    )
    from export_utils import ArtifactLayout


class DetrExporter:
    model_name = "detr_resnet50"
    description = "Export the DETR PyTorch checkpoint to ONNX and TensorRT."
    default_image_size = 800
    default_opset = 17

    def resolve_source(self, layout: ArtifactLayout, checkpoint: Path | None) -> ExportSource:
        source_dir = checkpoint or layout.pytorch_dir
        validate_detr_directory(source_dir)
        return ExportSource(path=source_dir, stem="model")

    def export_to_onnx(
        self,
        *,
        source: ExportSource,
        layout: ArtifactLayout,
        onnx_path: Path,
        options: ExportOptions,
    ) -> Path:
        try:
            import torch
            from torch import nn
            from transformers import AutoModelForObjectDetection
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Torch and transformers are required for DETR export."
            ) from exc

        class DetrOnnxWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(
                self,
                pixel_values: torch.Tensor,
                pixel_mask: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                return outputs.logits, outputs.pred_boxes

        model = AutoModelForObjectDetection.from_pretrained(
            str(source.path),
            local_files_only=True,
        )
        model.eval()

        wrapper = DetrOnnxWrapper(model)
        pixel_values = torch.randn(1, 3, options.image_size, options.image_size, dtype=torch.float32)
        pixel_mask = torch.ones(1, options.image_size, options.image_size, dtype=torch.bool)

        torch.onnx.export(
            wrapper,
            (pixel_values, pixel_mask),
            onnx_path,
            export_params=True,
            opset_version=options.opset,
            do_constant_folding=True,
            input_names=["pixel_values", "pixel_mask"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {2: "height", 3: "width"},
                "pixel_mask": {1: "height", 2: "width"},
                "logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
            },
        )
        return onnx_path

    def build_tensorrt_shape_profile(self, image_size: int) -> TensorRTShapeProfile:
        image_shape = f"1x3x{image_size}x{image_size}"
        mask_shape = f"1x{image_size}x{image_size}"
        return TensorRTShapeProfile(
            min_shapes={"pixel_values": image_shape, "pixel_mask": mask_shape},
            opt_shapes={"pixel_values": image_shape, "pixel_mask": mask_shape},
            max_shapes={"pixel_values": image_shape, "pixel_mask": mask_shape},
        )


EXPORTER = DetrExporter()


if __name__ == "__main__":
    main_for_exporter(EXPORTER)
