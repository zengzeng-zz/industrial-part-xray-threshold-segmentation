from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_mask(image_path: Path, threshold: int = 127) -> np.ndarray:
    """Load an image as a binary mask where foreground pixels are defects."""
    grayscale = Image.open(image_path).convert("L")
    pixels = np.array(grayscale, dtype=np.uint8)
    return pixels > threshold


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Extract 8-connected foreground components without external CV libs."""
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[list[tuple[int, int]]] = []
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            component: list[tuple[int, int]] = []

            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))

                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if not mask[next_row, next_col] or visited[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            components.append(component)

    return components


def estimate_perimeter(component_pixels: Iterable[tuple[int, int]], mask: np.ndarray) -> int:
    """Approximate perimeter by counting exposed 4-neighborhood edges."""
    component_set = set(component_pixels)
    perimeter = 0
    for row, col in component_set:
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not (0 <= next_row < mask.shape[0] and 0 <= next_col < mask.shape[1]):
                perimeter += 1
                continue
            if (next_row, next_col) not in component_set:
                perimeter += 1
    return perimeter


def summarize_component(component_pixels: list[tuple[int, int]], mask: np.ndarray, image_name: str, component_id: int) -> dict[str, int | str]:
    rows = [row for row, _ in component_pixels]
    cols = [col for _, col in component_pixels]

    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)

    area = len(component_pixels)
    perimeter = estimate_perimeter(component_pixels, mask)

    return {
        "image_name": image_name,
        "component_id": component_id,
        "area_px": area,
        "perimeter_px": perimeter,
        "bbox_x": min_col,
        "bbox_y": min_row,
        "bbox_width": max_col - min_col + 1,
        "bbox_height": max_row - min_row + 1,
    }


def analyze_image(image_path: Path, min_area: int) -> list[dict[str, int | str]]:
    mask = load_mask(image_path)
    components = connected_components(mask)
    summaries: list[dict[str, int | str]] = []

    for component_id, component in enumerate(components, start=1):
        if len(component) < min_area:
            continue
        summaries.append(summarize_component(component, mask, image_path.name, component_id))

    return summaries


def save_rows(rows: list[dict[str, int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "component_id",
        "area_px",
        "perimeter_px",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze binary defect masks and export contour statistics.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing segmentation result images.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file path.")
    parser.add_argument("--min-area", type=int, default=10, help="Ignore components smaller than this many pixels.")
    parser.add_argument(
        "--filename-keyword",
        type=str,
        default="_mask",
        help="Only analyze images whose filename contains this keyword. Use empty string to analyze all images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if args.filename_keyword:
        filtered_paths = [path for path in image_paths if args.filename_keyword in path.stem]
        if filtered_paths:
            image_paths = filtered_paths
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")

    all_rows: list[dict[str, int | str]] = []
    for image_path in image_paths:
        all_rows.extend(analyze_image(image_path, args.min_area))

    save_rows(all_rows, args.output)
    print(f"Saved {len(all_rows)} component rows to {args.output}")


if __name__ == "__main__":
    main()
