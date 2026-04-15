"""Standalone validation launcher for trained AMFE checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amfe.evaluation import (
    _format_model_complexity_summary,
    build_validation_overrides,
    load_evaluation_context,
    validate_trained_detector,
    validation_result_to_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse the standalone evaluation arguments."""

    parser = argparse.ArgumentParser(description="Validate a trained AMFE checkpoint on a YOLO detection dataset.")
    parser.add_argument("--weights", required=True, help="Path to the trained checkpoint (.pt).")
    parser.add_argument(
        "--config",
        help="Optional training config YAML. When provided, evaluation defaults reuse its data/model/training fields.",
    )
    parser.add_argument("--data", help="Optional dataset YAML override. Required if --config is not provided.")
    parser.add_argument(
        "--model-config",
        help="Optional model config YAML used for dataset/model consistency checks when --config is not provided.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, help="Evaluation image size override.")
    parser.add_argument("--batch", type=int, help="Evaluation batch size override.")
    parser.add_argument("--workers", type=int, help="Evaluation dataloader worker override.")
    parser.add_argument("--device", help="Evaluation device override, e.g. cpu, 0, 0,1.")
    half_group = parser.add_mutually_exclusive_group()
    half_group.add_argument("--half", action="store_true", help="Enable FP16 evaluation.")
    half_group.add_argument("--no-half", action="store_true", help="Disable FP16 evaluation.")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-format predictions JSON when supported.")
    parser.add_argument("--project", default="runs/eval", help="Validation output root directory.")
    parser.add_argument("--name", default="eval", help="Validation run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing save directory.")
    parser.add_argument("--conf", type=float, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--fps", action="store_true", help="Run the synthetic post-validation FPS benchmark.")
    parser.add_argument("--fps-batch-size", type=int, default=1, help="Batch size for the optional FPS benchmark.")
    parser.add_argument("--fps-warmup-iters", type=int, default=10, help="Warmup iterations for the FPS benchmark.")
    parser.add_argument("--fps-timed-iters", type=int, default=30, help="Timed iterations for the FPS benchmark.")
    parser.add_argument("--fps-imgsz", type=int, help="Image size override for the FPS benchmark.")
    parser.add_argument("--fps-use-amp", action="store_true", help="Use AMP for the optional FPS benchmark.")
    return parser.parse_args()


def _resolve_evaluation_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, dict[str, Any]]:
    """Resolve dataset/model config inputs from CLI flags and optional training config."""

    training_cfg: dict[str, Any] = {}
    model_config: Path | None = Path(args.model_config).resolve() if args.model_config else None

    if args.config:
        context = load_evaluation_context(args.config)
        data_yaml = context.data_yaml
        if context.model_config is not None:
            model_config = context.model_config
        training_cfg = context.training
    elif args.data:
        data_yaml = Path(args.data).resolve()
    else:
        raise ValueError("Standalone evaluation requires either --config or --data.")

    if args.data:
        data_yaml = Path(args.data).resolve()

    return data_yaml, model_config, training_cfg


def main() -> None:
    """Run standalone checkpoint validation and print a compact summary."""

    args = parse_args()
    data_yaml, model_config, training_cfg = _resolve_evaluation_inputs(args)
    half_override = True if args.half else False if args.no_half else None
    overrides = build_validation_overrides(
        weights=args.weights,
        data_yaml=data_yaml,
        training_cfg=training_cfg,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        half=half_override,
        plots=args.plots,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    fps_benchmark = None
    if args.fps:
        fps_benchmark = {
            "enabled": True,
            "batch_size": args.fps_batch_size,
            "warmup_iters": args.fps_warmup_iters,
            "timed_iters": args.fps_timed_iters,
            "use_amp": args.fps_use_amp,
        }
        if args.fps_imgsz is not None:
            fps_benchmark["imgsz"] = args.fps_imgsz

    result = validate_trained_detector(
        weights=args.weights,
        data_yaml=data_yaml,
        model_config=model_config,
        validator_overrides=overrides,
        fps_benchmark=fps_benchmark,
    )
    if result.model_complexity is not None:
        print(
            _format_model_complexity_summary(
                params_m=float(result.model_complexity["params_m"]),
                flops_g=result.model_complexity["flops_g"],
            )
        )
    print("Validation completed:", validation_result_to_dict(result))


if __name__ == "__main__":
    main()
