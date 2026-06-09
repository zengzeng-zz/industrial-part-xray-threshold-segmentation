from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess X-ray images with denoising and contrast enhancement."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed grayscale images.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("results/images/preprocess"),
        help="Directory for preprocess result images.",
    )
    parser.add_argument(
        "--median-size",
        type=int,
        default=5,
        help="Median filter kernel size. Use odd values such as 3 or 5.",
    )
    parser.add_argument(
        "--nlm-h",
        type=float,
        default=9.0,
        help="Filter strength for non-local means denoising.",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.5,
        help="Clip limit for CLAHE contrast enhancement.",
    )
    return parser.parse_args()


def supported_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")
    return image_paths


def preprocess_pixels(
    image: np.ndarray,
    median_size: int,
    nlm_h: float,
    clahe_clip_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grayscale = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(grayscale, median_size)
    denoised = cv2.fastNlMeansDenoising(median, None, h=nlm_h, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return grayscale, denoised, normalized


def build_preview(raw_pixels: np.ndarray, processed_pixels: np.ndarray) -> Image.Image:
    raw_image = Image.fromarray(raw_pixels)
    processed_image = Image.fromarray(processed_pixels)
    canvas = Image.new("L", (raw_image.width + processed_image.width, max(raw_image.height, processed_image.height)), color=255)
    canvas.paste(raw_image, (0, 0))
    canvas.paste(processed_image, (raw_image.width, 0))
    return canvas


def main() -> None:
    args = parse_args()
    image_paths = supported_images(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    grayscale_dir = args.preview_dir / "grayscale"
    denoised_dir = args.preview_dir / "denoised"
    enhanced_dir = args.preview_dir / "enhanced"
    comparison_dir = args.preview_dir / "comparison"

    for directory in (grayscale_dir, denoised_dir, enhanced_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        raw_pixels = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if raw_pixels is None:
            raise ValueError(f"Failed to read image: {image_path}")

        grayscale, denoised, processed = preprocess_pixels(
            raw_pixels,
            args.median_size,
            args.nlm_h,
            args.clahe_clip_limit,
        )
        processed_name = f"{image_path.stem}_processed.png"
        preview_name = f"{image_path.stem}_preview.png"
        grayscale_name = f"{image_path.stem}_grayscale.png"
        denoised_name = f"{image_path.stem}_denoised.png"
        enhanced_name = f"{image_path.stem}_enhanced.png"

        Image.fromarray(processed).save(args.output_dir / processed_name)
        Image.fromarray(grayscale).save(grayscale_dir / grayscale_name)
        Image.fromarray(denoised).save(denoised_dir / denoised_name)
        Image.fromarray(processed).save(enhanced_dir / enhanced_name)
        build_preview(raw_pixels, processed).save(comparison_dir / preview_name)

    print(f"Processed {len(image_paths)} images into {args.output_dir}")


if __name__ == "__main__":
    main()
