from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from contour_analysis import connected_components


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment weld defects from preprocessed X-ray images using weld-band constrained detection."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing preprocessed grayscale images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/images/segmentation"),
        help="Directory for masks, annotations, and comparisons.",
    )
    parser.add_argument(
        "--band-ratio",
        type=float,
        default=0.42,
        help="Controls how aggressively the bright weld band is selected from row means.",
    )
    parser.add_argument(
        "--band-padding",
        type=int,
        default=12,
        help="Extra pixels added above and below the detected weld band.",
    )
    parser.add_argument(
        "--inspection-band-scale",
        type=float,
        default=0.5,
        help="Use only the central portion of the weld band for defect detection.",
    )
    parser.add_argument(
        "--blackhat-kernel",
        type=int,
        default=31,
        help="Odd kernel size used to highlight dark defects inside the weld band.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=18,
        help="Ignore connected components smaller than this many pixels.",
    )
    parser.add_argument(
        "--max-height-ratio",
        type=float,
        default=0.45,
        help="Reject components that are too tall relative to the weld-band height.",
    )
    return parser.parse_args()


def supported_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")
    return image_paths


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def load_grayscale(image_path: Path) -> np.ndarray:
    pixels = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return pixels


def detect_weld_band(grayscale: np.ndarray, band_ratio: float, band_padding: int) -> tuple[int, int]:
    row_means = grayscale.mean(axis=1)
    peak = float(row_means.max())
    base = float(row_means.mean())
    threshold = base + band_ratio * (peak - base)
    mask = row_means >= threshold

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_selected in enumerate(mask):
        if is_selected and start is None:
            start = index
        elif not is_selected and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))

    if not spans:
        center = grayscale.shape[0] // 2
        half = max(20, grayscale.shape[0] // 8)
        return max(0, center - half), min(grayscale.shape[0] - 1, center + half)

    start, end = max(spans, key=lambda span: span[1] - span[0])
    return max(0, start - band_padding), min(grayscale.shape[0] - 1, end + band_padding)


def build_weld_band_mask(shape: tuple[int, int], top: int, bottom: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top : bottom + 1, :] = 255
    return mask


def detect_inspection_band(top: int, bottom: int, scale: float) -> tuple[int, int]:
    band_height = bottom - top + 1
    inner_height = max(12, int(band_height * scale))
    inner_height = min(inner_height, band_height)
    center = (top + bottom) // 2
    inner_top = max(top, center - inner_height // 2)
    inner_bottom = min(bottom, inner_top + inner_height - 1)
    inner_top = max(top, inner_bottom - inner_height + 1)
    return inner_top, inner_bottom


def compute_defect_candidates(grayscale: np.ndarray, weld_band_mask: np.ndarray, blackhat_kernel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_size = ensure_odd(blackhat_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.bitwise_and(blackhat, weld_band_mask)

    _, otsu_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_mask = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -3,
    )
    combined = cv2.bitwise_and(otsu_mask, adaptive_mask)
    return blackhat, otsu_mask, combined


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
    return closed > 0


def compute_threshold_masks(grayscale: np.ndarray, weld_band_mask: np.ndarray) -> dict[str, np.ndarray]:
    masked_gray = cv2.bitwise_and(grayscale, weld_band_mask)
    _, global_mask = cv2.threshold(masked_gray, 135, 255, cv2.THRESH_BINARY_INV)
    _, otsu_mask = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_mask = cv2.adaptiveThreshold(
        masked_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    threshold_masks = {
        "global": cv2.bitwise_and(global_mask, weld_band_mask),
        "otsu": cv2.bitwise_and(otsu_mask, weld_band_mask),
        "adaptive": cv2.bitwise_and(adaptive_mask, weld_band_mask),
    }
    return threshold_masks


def compute_edge_map(grayscale: np.ndarray, weld_band_mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(grayscale, 40, 120)
    return cv2.bitwise_and(edges, weld_band_mask)


def filter_components(mask: np.ndarray, min_area: int, band_height: int, max_height_ratio: float) -> np.ndarray:
    filtered = np.zeros_like(mask, dtype=bool)
    max_height = max(8, int(band_height * max_height_ratio))

    for component in connected_components(mask):
        if len(component) < min_area:
            continue

        rows = [row for row, _ in component]
        cols = [col for _, col in component]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1

        if height > max_height:
            continue
        if width > mask.shape[1] * 0.2 and height < 6:
            continue

        for row, col in component:
            filtered[row, col] = True

    return filtered


def component_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for component in connected_components(mask):
        if len(component) < min_area:
            continue
        rows = [row for row, _ in component]
        cols = [col for _, col in component]
        boxes.append((min(cols), min(rows), max(cols), max(rows)))
    return boxes


def annotate_image(
    grayscale: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    band_top: int,
    band_bottom: int,
    inspect_top: int,
    inspect_bottom: int,
) -> Image.Image:
    annotated = Image.fromarray(grayscale, mode="L").convert("RGB")
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((0, band_top, grayscale.shape[1] - 1, band_bottom), outline=(0, 255, 255), width=2)
    draw.rectangle((0, inspect_top, grayscale.shape[1] - 1, inspect_bottom), outline=(0, 255, 0), width=2)

    for index, (left, top, right, bottom) in enumerate(boxes, start=1):
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=2)
        draw.text((left, max(0, top - 12)), f"D{index}", fill=(255, 0, 0))

    return annotated


def save_outputs(
    image_name: str,
    grayscale: np.ndarray,
    weld_band_mask: np.ndarray,
    blackhat: np.ndarray,
    threshold_masks: dict[str, np.ndarray],
    edge_map: np.ndarray,
    filtered_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    output_dir: Path,
    band_top: int,
    band_bottom: int,
    inspect_top: int,
    inspect_bottom: int,
) -> None:
    thresholds_dir = output_dir / "thresholds"
    edges_dir = output_dir / "edges"
    comparisons_dir = output_dir / "comparisons"
    for directory in (thresholds_dir, edges_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_image = Image.fromarray((filtered_mask.astype(np.uint8) * 255), mode="L")
    annotated = annotate_image(grayscale, boxes, band_top, band_bottom, inspect_top, inspect_bottom)
    roi_preview = cv2.addWeighted(grayscale, 0.82, weld_band_mask, 0.18, 0)
    roi_preview_image = Image.fromarray(roi_preview, mode="L").convert("RGB")
    blackhat_image = Image.fromarray(blackhat, mode="L").convert("RGB")

    mask_image.save(output_dir / f"{image_name}_weld_mask.png")
    annotated.save(output_dir / f"{image_name}_weld_annotated.png")

    comparison = Image.new("RGB", (annotated.width * 3, annotated.height))
    comparison.paste(roi_preview_image, (0, 0))
    comparison.paste(blackhat_image, (annotated.width, 0))
    comparison.paste(annotated, (annotated.width * 2, 0))
    comparison.save(output_dir / f"{image_name}_weld_comparison.png")

    for method_name, mask in threshold_masks.items():
        Image.fromarray(mask, mode="L").save(thresholds_dir / f"{image_name}_{method_name}_mask.png")

    Image.fromarray(edge_map, mode="L").save(edges_dir / f"{image_name}_edge.png")

    panel_images = [
        Image.fromarray(grayscale, mode="L").convert("RGB"),
        Image.fromarray(threshold_masks["global"], mode="L").convert("RGB"),
        Image.fromarray(threshold_masks["otsu"], mode="L").convert("RGB"),
        Image.fromarray(threshold_masks["adaptive"], mode="L").convert("RGB"),
        Image.fromarray(edge_map, mode="L").convert("RGB"),
    ]
    panel = Image.new("RGB", (annotated.width * len(panel_images), annotated.height))
    for index, panel_image in enumerate(panel_images):
        panel.paste(panel_image, (annotated.width * index, 0))
    panel.save(comparisons_dir / f"{image_name}_threshold_comparison.png")


def main() -> None:
    args = parse_args()
    image_paths = supported_images(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        grayscale = load_grayscale(image_path)
        image_name = image_path.stem

        band_top, band_bottom = detect_weld_band(grayscale, args.band_ratio, args.band_padding)
        inspect_top, inspect_bottom = detect_inspection_band(
            band_top,
            band_bottom,
            args.inspection_band_scale,
        )
        weld_band_mask = build_weld_band_mask(grayscale.shape, inspect_top, inspect_bottom)
        blackhat, _, combined_mask = compute_defect_candidates(grayscale, weld_band_mask, args.blackhat_kernel)
        threshold_masks = compute_threshold_masks(grayscale, weld_band_mask)
        edge_map = compute_edge_map(grayscale, weld_band_mask)
        cleaned_mask = clean_mask(combined_mask)
        filtered_mask = filter_components(
            cleaned_mask,
            args.min_area,
            band_height=inspect_bottom - inspect_top + 1,
            max_height_ratio=args.max_height_ratio,
        )
        boxes = component_boxes(filtered_mask, args.min_area)

        save_outputs(
            image_name,
            grayscale,
            weld_band_mask,
            blackhat,
            threshold_masks,
            edge_map,
            filtered_mask,
            boxes,
            args.output_dir,
            band_top,
            band_bottom,
            inspect_top,
            inspect_bottom,
        )
        print(
            f"{image_path.name}: weld_band=({band_top}, {band_bottom}), "
            f"inspect_band=({inspect_top}, {inspect_bottom}), defects={len(boxes)}"
        )


if __name__ == "__main__":
    main()
