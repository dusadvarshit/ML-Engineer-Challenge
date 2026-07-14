"""Shared helpers for exporting model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "models" / "artifacts"


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved artifact directories for one model version."""

    model_name: str
    version: str
    task: str
    pytorch_dir: Path
    onnx_dir: Path
    tensorrt_dir: Path
    quantized_dir: Path


def build_artifact_layout(
    *,
    task: str,
    model_name: str,
    version: str,
    pytorch_dir: str | Path | None = None,
    onnx_dir: str | Path | None = None,
    tensorrt_dir: str | Path | None = None,
    quantized_dir: str | Path | None = None,
) -> ArtifactLayout:
    """Build the default artifact directory layout for a model version."""

    version_root = ARTIFACTS_ROOT / task / model_name / version
    return ArtifactLayout(
        model_name=model_name,
        version=version,
        task=task,
        pytorch_dir=Path(pytorch_dir) if pytorch_dir is not None else version_root / "pytorch",
        onnx_dir=Path(onnx_dir) if onnx_dir is not None else version_root / "onnx",
        tensorrt_dir=Path(tensorrt_dir) if tensorrt_dir is not None else version_root / "tensorrt",
        quantized_dir=(
            Path(quantized_dir) if quantized_dir is not None else version_root / "quantized"
        ),
    )


def ensure_export_directories(layout: ArtifactLayout) -> None:
    """Create export directories when they do not exist yet."""

    layout.onnx_dir.mkdir(parents=True, exist_ok=True)
    layout.tensorrt_dir.mkdir(parents=True, exist_ok=True)
    layout.quantized_dir.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(
    pytorch_dir: Path,
    *,
    allowed_suffixes: tuple[str, ...],
    preferred_names: tuple[str, ...] = (),
) -> Path:
    """Resolve a checkpoint file inside a PyTorch artifact directory."""

    if not pytorch_dir.exists():
        raise FileNotFoundError(f"PyTorch artifact directory does not exist: {pytorch_dir}")

    for preferred_name in preferred_names:
        candidate = pytorch_dir / preferred_name
        if candidate.exists() and candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in pytorch_dir.iterdir()
        if path.is_file() and path.suffix in allowed_suffixes and path.name != ".gitkeep"
    )
    if not candidates:
        suffix_list = ", ".join(allowed_suffixes)
        raise FileNotFoundError(
            f"No checkpoint with suffix {suffix_list} found in {pytorch_dir}."
        )

    return candidates[0]


def extract_state_dict(checkpoint: object) -> object:
    """Return a model state dict from a raw checkpoint payload."""

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    return checkpoint


def move_file(source: Path, destination: Path) -> Path:
    """Move an exported file to its final artifact location."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def run_subprocess(command: list[str]) -> None:
    """Run a subprocess and surface stderr when the command fails."""

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    details = stderr or stdout or "no output captured"
    raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{details}")


def quantize_onnx_to_int8(
    *,
    source_path: Path,
    destination_path: Path,
    calibration_reader: object | None = None,
) -> Path:
    """Create an INT8 ONNX artifact from an existing ONNX export."""

    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX INT8 export requires `onnxruntime`. Install the project export dependencies first."
        ) from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_reader is None:
        quantize_dynamic(
            str(source_path),
            str(destination_path),
            per_channel=True,
            weight_type=QuantType.QInt8,
        )
        return destination_path

    quantize_static(
        str(source_path),
        str(destination_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return destination_path


def convert_onnx_to_tensorrt(
    *,
    onnx_path: Path,
    engine_path: Path,
    min_shapes: dict[str, str] | None = None,
    opt_shapes: dict[str, str] | None = None,
    max_shapes: dict[str, str] | None = None,
    precision: str = "int8",
    workspace_mib: int = 4096,
) -> Path:
    """Build a TensorRT engine from an ONNX export with trtexec."""

    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")

    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        raise RuntimeError(
            "TensorRT export requires the `trtexec` binary. "
            "Run this inside the TensorRT container or install TensorRT locally."
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
    ]
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
    if workspace_mib > 0:
        command.append(f"--memPoolSize=workspace:{workspace_mib}")
    if min_shapes:
        command.append(f"--minShapes={format_shape_arguments(min_shapes)}")
    if opt_shapes:
        command.append(f"--optShapes={format_shape_arguments(opt_shapes)}")
    if max_shapes:
        command.append(f"--maxShapes={format_shape_arguments(max_shapes)}")

    run_subprocess(command)
    return engine_path


def format_shape_arguments(shapes: dict[str, str]) -> str:
    """Render trtexec input shape arguments."""

    return ",".join(f"{name}:{shape}" for name, shape in shapes.items())
