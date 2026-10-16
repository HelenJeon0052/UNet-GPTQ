from __future__ import annotations
import csv
from collections import defaultdict



from dataclasses import dataclass
from pathlib import Path

import torch

from unet_gptq.datas.fastmri_dataset import _crop_shape

@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    label: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "label": self.label,
        }


def load_fastmri_plus_annotations(
    csv_path: str | Path,
) -> dict[tuple[str, int], list[BoundingBox]]:
    """Load the official columns: file,slice,study_level,x,y,width,height,label."""
    result: dict[tuple[str, int], list[BoundingBox]] = defaultdict(list)
    required = {"file", "slice", "x", "y", "width", "height", "label"}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing FastMRI+ columns: {sorted(missing)}")
        for row in reader:
            file_id = Path(row["file"].strip()).stem
            slice_index = int(row["slice"])
            result[(file_id, slice_index)].append(
                BoundingBox(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                    label=row["label"].strip(),
                )
            )
    return dict(result)

def boxes_to_roi_mask(
    boxes: list[dict] | list[BoundingBox],
    source_shape: tuple[int, int],
    crop_size: tuple[int, int],
    *,
    annotations_flipped: bool = True
) -> tuple[torch.Tensor, list[dict[str, int | str]]]:
    source_h, source_w = source_shape
    
    crop_h, crop_w = _crop_shape(crop_size)
    if crop_h > source_h or crop_w > source_w:
        raise ValueError(f"{(crop_h, crop_w)} must be larger {source_shape}")

    top = (source_h - crop_h) // 2
    left = (source_w - crop_w) // 2
    roi = torch.zeros((1, crop_h, crop_w), dtype=torch.bool)
    transformed: list[dict[str, int | str]] = []

    for item in boxes:
        box = item if isinstance(item, BoundingBox) else BoundingBox(**item)
        x_0 = box.x
        y_0 = source_h - (box.y + box.height) if annotations_flipped else box.y
        x_1 = x_0 + box.width
        y_1 = y_0 + box.height

        x_0 = max(0, min(crop_w, int(round(x_0 - left))))
        x_1 = max(0, min(crop_w, int(round(x_1 - left))))
        y_0 = max(0, min(crop_h, int(round(y_0 - top))))
        y_1 = max(0, min(crop_h, int(round(y_1 - top))))
        if x_1 <= x_0 or y_1 <= y_0:
            continue
        roi[:, y_0:y_1, x_0:x_1] = True
        transformed.append(
            {"x_0": x_0, "y_0": y_0, "x_1": x_1, "y_1": y_1, "label": box.label}
        )

    return roi, transformed