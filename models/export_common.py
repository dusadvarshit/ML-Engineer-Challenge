"""Shared export pipeline used by the model-specific export scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    from models.export_utils import (
        ArtifactLayout,
        build_artifact_layout,
        convert_onnx_to_tensorrt,
        ensure_export_directories,
        quantize_onnx_to_int8,
    )
except ModuleNotFoundError:
    from export_utils import (
        ArtifactLayout,
        build_artifact_layout,
        convert_onnx_to_tensorrt,
        ensure_export_directories,
        quantize_onnx_to_int8,
    )


DEFAULT_TASK = "object_detection"


@dataclass(frozen=True)
class ExportSource:
    """Resolved source input for one export run."""

    path: Path
    stem: str


@dataclass(frozen=True)
class TensorRTShapeProfile:
    """Per-model trtexec input shape profile."""

    min_shapes: dict[str, str]
    opt_shapes: dict[str, str]
    max_shapes: dict[str, str]


@dataclass(frozen=True)
class ExportOptions:
    """Normalized export options shared across models."""

    version: str
    pytorch_dir: Path | None
    onnx_dir: Path | None
    tensorrt_dir: Path | None
    quantized_dir: Path | None
    checkpoint: Path | None
    image_size: int
    opset: int
    device: str
    skip_onnx: bool
    skip_tensorrt: bool
    skip_quantized_onnx: bool
    trt_precision: str
    workspace_mib: int


class ModelExporter(Protocol):
    """Model-specific hooks used by the shared export pipeline."""

    model_name: str
    description: str
    default_image_size: int
    default_opset: int

    def resolve_source(self, layout: ArtifactLayout, checkpoint: Path | None) -> ExportSource:
        ...

    def export_to_onnx(
        self,
        *,
        source: ExportSource,
        layout: ArtifactLayout,
        onnx_path: Path,
        options: ExportOptions,
    ) -> Path:
        ...

    def build_tensorrt_shape_profile(self, image_size: int) -> TensorRTShapeProfile:
        ...


def validate_detr_directory(pytorch_dir: Path) -> None:
    """Ensure the Hugging Face artifact directory contains the required files."""

    required_files = (
        pytorch_dir / "config.json",
        pytorch_dir / "model.safetensors",
        pytorch_dir / "preprocessor_config.json",
    )
    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing DETR model artifacts in {pytorch_dir}: {', '.join(missing)}"
        )


def build_parser(
    exporter: ModelExporter,
    *,
    include_model_argument: bool,
    model_names: list[str] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=exporter.description)
    if include_model_argument:
        if model_names is None:
            raise ValueError("model_names must be provided when include_model_argument is True.")
        parser.add_argument("model", choices=model_names, help="Model export profile to use.")
    parser.add_argument("--version", default="v1.0.0", help="Artifact version to export.")
    parser.add_argument("--pytorch-dir", type=Path, help="Override the PyTorch artifact directory.")
    parser.add_argument("--onnx-dir", type=Path, help="Override the ONNX artifact directory.")
    parser.add_argument("--tensorrt-dir", type=Path, help="Override the TensorRT artifact directory.")
    parser.add_argument("--quantized-dir", type=Path, help="Override the INT8 ONNX artifact directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the checkpoint path, or the DETR artifact directory.",
    )
    parser.add_argument(
        "--image-size",
        "--imgsz",
        dest="image_size",
        type=int,
        default=exporter.default_image_size,
        help="Square input size used for ONNX export and TensorRT shape profiles.",
    )
    parser.add_argument("--opset", type=int, default=exporter.default_opset, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Used by the YOLO path and ignored by other exporters.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export and reuse an existing ONNX artifact.",
    )
    parser.add_argument(
        "--skip-quantized-onnx",
        action="store_true",
        help="Skip INT8 ONNX export and reuse an existing quantized artifact.",
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT export.",
    )
    parser.add_argument(
        "--trt-precision",
        choices=("fp32", "fp16", "int8"),
        default="int8",
        help="TensorRT engine precision. Defaults to INT8 to match the challenge requirements.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Compatibility flag for the old scripts. Equivalent to --trt-precision fp16.",
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=4096,
        help="TensorRT workspace size in MiB.",
    )
    return parser


def namespace_to_options(args: argparse.Namespace) -> ExportOptions:
    trt_precision = "fp16" if args.fp16 else args.trt_precision
    return ExportOptions(
        version=args.version,
        pytorch_dir=args.pytorch_dir,
        onnx_dir=args.onnx_dir,
        tensorrt_dir=args.tensorrt_dir,
        quantized_dir=args.quantized_dir,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        opset=args.opset,
        device=args.device,
        skip_onnx=args.skip_onnx,
        skip_tensorrt=args.skip_tensorrt,
        skip_quantized_onnx=args.skip_quantized_onnx,
        trt_precision=trt_precision,
        workspace_mib=args.workspace_mib,
    )


def run_export(exporter: ModelExporter, options: ExportOptions) -> None:
    layout = build_artifact_layout(
        task=DEFAULT_TASK,
        model_name=exporter.model_name,
        version=options.version,
        pytorch_dir=options.pytorch_dir,
        onnx_dir=options.onnx_dir,
        tensorrt_dir=options.tensorrt_dir,
        quantized_dir=options.quantized_dir,
    )
    ensure_export_directories(layout)

    source = exporter.resolve_source(layout, options.checkpoint)
    onnx_path = layout.onnx_dir / f"{source.stem}.onnx"
    quantized_onnx_path = layout.quantized_dir / f"{source.stem}.int8.onnx"
    engine_path = layout.tensorrt_dir / f"{source.stem}.{options.trt_precision}.engine"

    if not options.skip_onnx:
        exporter.export_to_onnx(
            source=source,
            layout=layout,
            onnx_path=onnx_path,
            options=options,
        )
    elif not onnx_path.exists():
        raise FileNotFoundError(f"Expected existing ONNX artifact at {onnx_path}.")

    if not options.skip_quantized_onnx:
        quantize_onnx_to_int8(
            source_path=onnx_path,
            destination_path=quantized_onnx_path,
        )
    elif not quantized_onnx_path.exists():
        raise FileNotFoundError(f"Expected existing INT8 ONNX artifact at {quantized_onnx_path}.")

    if not options.skip_tensorrt:
        shape_profile = exporter.build_tensorrt_shape_profile(options.image_size)
        onnx_input = quantized_onnx_path if options.trt_precision == "int8" else onnx_path
        convert_onnx_to_tensorrt(
            onnx_path=onnx_input,
            engine_path=engine_path,
            min_shapes=shape_profile.min_shapes,
            opt_shapes=shape_profile.opt_shapes,
            max_shapes=shape_profile.max_shapes,
            precision=options.trt_precision,
            workspace_mib=options.workspace_mib,
        )

    print(f"ONNX artifact: {onnx_path}")
    print(f"INT8 ONNX artifact: {quantized_onnx_path}")
    if not options.skip_tensorrt:
        print(f"TensorRT artifact: {engine_path}")


def main_for_exporter(exporter: ModelExporter, argv: list[str] | None = None) -> None:
    parser = build_parser(exporter, include_model_argument=False)
    options = namespace_to_options(parser.parse_args(argv))
    run_export(exporter, options)


def main_for_registry(
    exporters: dict[str, ModelExporter],
    argv: list[str] | None = None,
) -> None:
    argv = list(argv) if argv is not None else None
    model_names = sorted(exporters)
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("model", choices=model_names)
    known_args, _ = base_parser.parse_known_args(argv)

    exporter = exporters[known_args.model]
    parser = build_parser(
        exporter,
        include_model_argument=True,
        model_names=model_names,
    )
    args = parser.parse_args(argv)
    options = namespace_to_options(args)
    run_export(exporter, options)
