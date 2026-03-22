"""Utilities for converting VisDrone-style raw annotations to YOLO detect format."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import yaml

VISDRONE_CATEGORY_NAMES: dict[int, str] = {
    0: "ignored-region",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# The current dataset contains raw ids 0..11. Following the standard
# VisDrone-DET convention, ids 1..10 are trainable object classes, while
# ids 0 and 11 are ignored during YOLO label conversion.
VISDRONE_TO_YOLO_CLASS_ID: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}

VISDRONE_YOLO_CLASS_NAMES: dict[int, str] = {
    yolo_id: VISDRONE_CATEGORY_NAMES[raw_id] for raw_id, yolo_id in VISDRONE_TO_YOLO_CLASS_ID.items()
}

IGNORED_RAW_CATEGORY_IDS = {0, 11}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SourceSample:
    """A single source image/annotation pair discovered under the raw dataset root."""

    stem: str
    image_path: Path
    annotation_path: Path
    sequence_id: str


@dataclass(frozen=True)
class RawVisDroneAnnotation:
    """One raw VisDrone-style annotation row."""

    bbox_left: float
    bbox_top: float
    width: float
    height: float
    score: int
    category_id: int
    truncation: int
    occlusion: int


@dataclass(frozen=True)
class ConversionStats:
    """Summary returned after a conversion run."""

    source_root: Path
    dataset_root: Path
    total_images: int
    train_images: int
    val_images: int
    raw_category_counts: dict[int, int]
    kept_category_counts: dict[int, int]
    ignored_category_counts: dict[int, int]
    written_label_rows: int
    empty_label_files: int
    val_sequence_ids: tuple[str, ...]
    dataset_yaml: Path


@dataclass(frozen=True)
class ValidationStats:
    """Summary returned after validating a YOLO detection dataset."""

    dataset_root: Path
    image_count: int
    label_count: int
    empty_label_files: int
    annotation_rows: int
    split_image_counts: dict[str, int]


def parse_visdrone_annotation(line: str, *, file_path: Path, line_number: int) -> RawVisDroneAnnotation:
    """Parse one comma-separated raw annotation row and validate its structure."""

    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 8:
        raise ValueError(
            f"{file_path}:{line_number} expected 8 comma-separated values, found {len(parts)}: {line!r}"
        )

    try:
        bbox_left, bbox_top, width, height = (float(parts[index]) for index in range(4))
        score = int(parts[4])
        category_id = int(parts[5])
        truncation = int(parts[6])
        occlusion = int(parts[7])
    except ValueError as exc:
        raise ValueError(f"{file_path}:{line_number} contains non-numeric fields: {line!r}") from exc

    if bbox_left < 0 or bbox_top < 0:
        raise ValueError(f"{file_path}:{line_number} contains a negative top-left coordinate: {line!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"{file_path}:{line_number} contains a non-positive bounding box size: {line!r}")

    return RawVisDroneAnnotation(
        bbox_left=bbox_left,
        bbox_top=bbox_top,
        width=width,
        height=height,
        score=score,
        category_id=category_id,
        truncation=truncation,
        occlusion=occlusion,
    )


def discover_source_samples(source_root: Path) -> list[SourceSample]:
    """Discover source samples in a flat ``images/`` + ``annotations/`` dataset root."""

    images_dir = source_root / "images"
    annotations_dir = source_root / "annotations"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images directory was not found: {images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Expected annotations directory was not found: {annotations_dir}")

    image_by_stem: dict[str, Path] = {}
    for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
        suffix = image_path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if image_path.stem in image_by_stem:
            raise ValueError(f"Duplicate image stem discovered under {images_dir}: {image_path.stem}")
        image_by_stem[image_path.stem] = image_path

    annotation_by_stem: dict[str, Path] = {}
    for annotation_path in sorted(annotations_dir.glob("*.txt")):
        if annotation_path.stem in annotation_by_stem:
            raise ValueError(f"Duplicate annotation stem discovered under {annotations_dir}: {annotation_path.stem}")
        annotation_by_stem[annotation_path.stem] = annotation_path

    missing_images = sorted(stem for stem in annotation_by_stem if stem not in image_by_stem)
    missing_annotations = sorted(stem for stem in image_by_stem if stem not in annotation_by_stem)
    if missing_images:
        raise ValueError(f"Found annotations without matching images. First missing stems: {missing_images[:10]}")
    if missing_annotations:
        raise ValueError(
            f"Found images without matching annotations. First missing stems: {missing_annotations[:10]}"
        )

    samples: list[SourceSample] = []
    for stem in sorted(image_by_stem):
        sequence_id = stem.split("_")[0]
        samples.append(
            SourceSample(
                stem=stem,
                image_path=image_by_stem[stem],
                annotation_path=annotation_by_stem[stem],
                sequence_id=sequence_id,
            )
        )
    if not samples:
        raise ValueError(f"No source samples were discovered under {source_root}")
    return samples


def build_sequence_split_map(samples: list[SourceSample], *, val_every_n_groups: int = 5) -> dict[str, str]:
    """Build a deterministic train/val split map at the sequence level."""

    if val_every_n_groups < 2:
        raise ValueError("val_every_n_groups must be at least 2.")

    sequence_ids = sorted({sample.sequence_id for sample in samples})
    if len(sequence_ids) < 2:
        raise ValueError("At least two sequence groups are required to create train/val splits.")

    val_sequence_ids = {sequence_id for index, sequence_id in enumerate(sequence_ids) if index % val_every_n_groups == 0}
    train_sequence_ids = set(sequence_ids) - val_sequence_ids
    if not train_sequence_ids or not val_sequence_ids:
        raise ValueError(
            "Deterministic split produced an empty train or val partition. "
            "Adjust the sequence grouping or val_every_n_groups."
        )

    split_map = {sequence_id: "val" if sequence_id in val_sequence_ids else "train" for sequence_id in sequence_ids}
    return split_map


def convert_annotation_row(
    annotation: RawVisDroneAnnotation,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float] | None:
    """Convert one raw row to YOLO format or return ``None`` when it should be ignored."""

    if annotation.category_id in VISDRONE_TO_YOLO_CLASS_ID:
        if annotation.score != 1:
            raise ValueError(
                f"Trainable category {annotation.category_id} must have score 1, got {annotation.score}."
            )
    elif annotation.category_id in IGNORED_RAW_CATEGORY_IDS:
        if annotation.score != 0:
            raise ValueError(
                f"Ignored category {annotation.category_id} must have score 0, got {annotation.score}."
            )
        return None
    else:
        raise ValueError(
            f"Raw category id {annotation.category_id} is not covered by the documented VisDrone mapping."
        )

    if annotation.bbox_left + annotation.width > image_width:
        raise ValueError("Bounding box exceeds image width; refusing to clamp silently.")
    if annotation.bbox_top + annotation.height > image_height:
        raise ValueError("Bounding box exceeds image height; refusing to clamp silently.")

    x_center = (annotation.bbox_left + (annotation.width / 2.0)) / image_width
    y_center = (annotation.bbox_top + (annotation.height / 2.0)) / image_height
    norm_width = annotation.width / image_width
    norm_height = annotation.height / image_height
    values = (x_center, y_center, norm_width, norm_height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Normalized YOLO coordinates are outside [0, 1]: {values}")
    if norm_width <= 0.0 or norm_height <= 0.0:
        raise ValueError(f"Normalized YOLO width/height must be positive: {values}")

    yolo_class_id = VISDRONE_TO_YOLO_CLASS_ID[annotation.category_id]
    return (yolo_class_id, x_center, y_center, norm_width, norm_height)


def write_dataset_yaml(dataset_root: Path) -> Path:
    """Write the Ultralytics dataset YAML under the converted dataset root."""

    dataset_yaml = dataset_root / "dataset.yaml"
    payload = {
        "path": dataset_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": VISDRONE_YOLO_CLASS_NAMES,
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def convert_visdrone_dataset(
    source_root: Path,
    output_root: Path,
    *,
    dataset_name: str = "visdrone_yolo",
    val_every_n_groups: int = 5,
) -> ConversionStats:
    """Convert a flat VisDrone-style dataset into a YOLO detection dataset."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    dataset_root = output_root / dataset_name
    if dataset_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing converted dataset: {dataset_root}. "
            "Remove it explicitly before re-running conversion."
        )

    samples = discover_source_samples(source_root)
    split_map = build_sequence_split_map(samples, val_every_n_groups=val_every_n_groups)
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=False)

    raw_category_counts: Counter[int] = Counter()
    kept_category_counts: Counter[int] = Counter()
    ignored_category_counts: Counter[int] = Counter()
    written_label_rows = 0
    empty_label_files = 0
    split_image_counts = Counter()

    for sample in samples:
        split = split_map[sample.sequence_id]
        split_image_counts[split] += 1

        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise ValueError(f"Failed to read image for conversion: {sample.image_path}")
        image_height, image_width = image.shape[:2]

        converted_rows: list[str] = []
        for line_number, line in enumerate(sample.annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            annotation = parse_visdrone_annotation(line, file_path=sample.annotation_path, line_number=line_number)
            raw_category_counts[annotation.category_id] += 1
            converted = convert_annotation_row(annotation, image_width=image_width, image_height=image_height)
            if converted is None:
                ignored_category_counts[annotation.category_id] += 1
                continue
            class_id, x_center, y_center, width, height = converted
            kept_category_counts[annotation.category_id] += 1
            converted_rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_output_path = dataset_root / "labels" / split / f"{sample.stem}.txt"
        label_output_path.write_text("\n".join(converted_rows), encoding="utf-8")
        if not converted_rows:
            empty_label_files += 1
        written_label_rows += len(converted_rows)

        image_output_path = dataset_root / "images" / split / sample.image_path.name
        shutil.copy2(sample.image_path, image_output_path)

    dataset_yaml = write_dataset_yaml(dataset_root)
    val_sequence_ids = tuple(sorted(sequence_id for sequence_id, split in split_map.items() if split == "val"))
    return ConversionStats(
        source_root=source_root,
        dataset_root=dataset_root,
        total_images=len(samples),
        train_images=split_image_counts["train"],
        val_images=split_image_counts["val"],
        raw_category_counts=dict(sorted(raw_category_counts.items())),
        kept_category_counts=dict(sorted(kept_category_counts.items())),
        ignored_category_counts=dict(sorted(ignored_category_counts.items())),
        written_label_rows=written_label_rows,
        empty_label_files=empty_label_files,
        val_sequence_ids=val_sequence_ids,
        dataset_yaml=dataset_yaml,
    )


def _load_num_classes_from_dataset_yaml(dataset_yaml: Path) -> int:
    """Load the class count from a dataset YAML generated by this module."""

    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        normalized = {int(key): value for key, value in names.items()}
        expected_keys = list(range(len(normalized)))
        if sorted(normalized) != expected_keys:
            raise ValueError(f"{dataset_yaml} contains non-contiguous class ids: {sorted(normalized)}")
        return len(normalized)
    if isinstance(names, list):
        return len(names)
    raise ValueError(f"{dataset_yaml} must define names as a list or id->name mapping.")


def validate_yolo_dataset(dataset_root: Path) -> ValidationStats:
    """Validate a converted YOLO detection dataset and return summary statistics."""

    dataset_root = dataset_root.resolve()
    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Expected dataset YAML was not found: {dataset_yaml}")

    num_classes = _load_num_classes_from_dataset_yaml(dataset_yaml)
    epsilon = 2e-6
    split_image_counts: dict[str, int] = {}
    image_count = 0
    label_count = 0
    empty_label_files = 0
    annotation_rows = 0

    for split in ("train", "val"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Expected images split directory was not found: {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Expected labels split directory was not found: {labels_dir}")

        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
        label_paths = sorted(labels_dir.glob("*.txt"))
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        if missing_labels:
            raise ValueError(f"Split {split} is missing label files for image stems: {missing_labels[:10]}")
        if extra_labels:
            raise ValueError(f"Split {split} contains label files without images: {extra_labels[:10]}")

        split_image_counts[split] = len(image_paths)
        image_count += len(image_paths)
        label_count += len(label_paths)

        for label_path in label_paths:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                empty_label_files += 1
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number} expected 5 whitespace-separated values, found {len(parts)}."
                    )
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = (float(parts[index]) for index in range(1, 5))
                except ValueError as exc:
                    raise ValueError(f"{label_path}:{line_number} contains non-numeric values: {line!r}") from exc

                if class_id < 0 or class_id >= num_classes:
                    raise ValueError(f"{label_path}:{line_number} contains invalid class id {class_id}.")
                if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                    raise ValueError(f"{label_path}:{line_number} contains non-normalized coordinates: {line!r}")
                if width <= 0.0 or height <= 0.0:
                    raise ValueError(f"{label_path}:{line_number} contains non-positive width/height: {line!r}")
                if x_center - (width / 2.0) < -epsilon or x_center + (width / 2.0) > 1.0 + epsilon:
                    raise ValueError(f"{label_path}:{line_number} bbox extends outside normalized x range: {line!r}")
                if y_center - (height / 2.0) < -epsilon or y_center + (height / 2.0) > 1.0 + epsilon:
                    raise ValueError(f"{label_path}:{line_number} bbox extends outside normalized y range: {line!r}")
                annotation_rows += 1

    return ValidationStats(
        dataset_root=dataset_root,
        image_count=image_count,
        label_count=label_count,
        empty_label_files=empty_label_files,
        annotation_rows=annotation_rows,
        split_image_counts=split_image_counts,
    )
