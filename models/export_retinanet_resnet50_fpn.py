"""RetinaNet ResNet-50 FPN export implementation."""

from __future__ import annotations

from pathlib import Path

try:
    from models.export_common import (
        ExportOptions,
        ExportSource,
        TensorRTShapeProfile,
        main_for_exporter,
    )
    from models.export_utils import ArtifactLayout, extract_state_dict, resolve_checkpoint
except ModuleNotFoundError:
    from export_common import (
        ExportOptions,
        ExportSource,
        TensorRTShapeProfile,
        main_for_exporter,
    )
    from export_utils import ArtifactLayout, extract_state_dict, resolve_checkpoint


class RetinaNetExporter:
    model_name = "retinanet_resnet50_fpn"
    description = "Export the RetinaNet PyTorch checkpoint to ONNX and TensorRT."
    default_image_size = 640
    default_opset = 17

    def resolve_source(self, layout: ArtifactLayout, checkpoint: Path | None) -> ExportSource:
        checkpoint_path = checkpoint or resolve_checkpoint(
            layout.pytorch_dir,
            allowed_suffixes=(".pth", ".pt"),
            preferred_names=("retinanet_resnet50_fpn_coco-eeacb38b.pth",),
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
            import torch
            from torch import nn
            from torchvision.models.detection import retinanet_resnet50_fpn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Torch and torchvision are required for RetinaNet export."
            ) from exc

        class RetinaNetOnnxWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                detections = self.model([images[0]])[0]
                return (
                    detections["boxes"].to(dtype=torch.float32),
                    detections["scores"].to(dtype=torch.float32),
                    detections["labels"].to(dtype=torch.int32),
                )

        checkpoint = torch.load(source.path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)

        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None)
        model.load_state_dict(state_dict)
        model.eval()

        wrapper = RetinaNetOnnxWrapper(model)
        dummy_input = torch.randn(1, 3, options.image_size, options.image_size, dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=options.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes={
                "images": {2: "height", 3: "width"},
                "boxes": {0: "num_detections"},
                "scores": {0: "num_detections"},
                "labels": {0: "num_detections"},
            },
        )
        return onnx_path

    def build_tensorrt_shape_profile(self, image_size: int) -> TensorRTShapeProfile:
        image_shape = f"1x3x{image_size}x{image_size}"
        return TensorRTShapeProfile(
            min_shapes={"images": image_shape},
            opt_shapes={"images": image_shape},
            max_shapes={"images": image_shape},
        )


EXPORTER = RetinaNetExporter()


if __name__ == "__main__":
    main_for_exporter(EXPORTER)
