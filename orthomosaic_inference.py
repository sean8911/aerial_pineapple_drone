"""Count and geo-locate pineapples across a drone orthomosaic.

Tiles the orthomosaic, runs a trained YOLO model per tile, counts
detections, and writes georeferenced boxes to a shapefile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import xy
from rasterio.windows import Window
from shapely.geometry import Polygon
from ultralytics import YOLO

TILE_SIZE = 1024
RGB_BANDS = 3
CONF_THRESHOLD = 0.352
INFERENCE_IMAGE_SIZE = 1024
DETECT_COLUMNS = ["class", "cx", "cy", "w", "h"]


def tile_orthomosaic(source: Path, tile_dir: Path, tile_size: int = TILE_SIZE) -> int:
    """Write non-empty full-size RGB tiles as GeoTIFFs, preserving geotransforms.

    Partial edge windows and all-black nodata windows are skipped.
    """
    if not source.is_file():
        raise FileNotFoundError(f"orthomosaic not found: {source}")
    tile_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with rasterio.open(source) as src:
        meta = src.meta.copy()
        for n in range(src.width // tile_size):
            for m in range(src.height // tile_size):
                window = Window(n * tile_size, m * tile_size, tile_size, tile_size)
                patch = src.read(window=window)[:RGB_BANDS]
                if patch.max() == 0 or patch.shape[1:] != (tile_size, tile_size):
                    continue
                written += 1
                meta.update(
                    driver="GTiff",
                    count=RGB_BANDS,
                    height=tile_size,
                    width=tile_size,
                    transform=src.window_transform(window),
                )
                with rasterio.open(tile_dir / f"tile_{written}.tif", "w", **meta) as dst:
                    dst.write(patch)
    return written


def tiles_to_jpg(tile_dir: Path, jpg_dir: Path) -> None:
    """Convert GeoTIFF tiles to RGB JPGs for inference."""
    jpg_dir.mkdir(parents=True, exist_ok=True)
    for tile_path in sorted(tile_dir.glob("*.tif")):
        with rasterio.open(tile_path) as src:
            rgb = src.read()[:RGB_BANDS].transpose(1, 2, 0)
        Image.fromarray(rgb).save(jpg_dir / f"{tile_path.stem}.jpg")


def detect(model_path: Path, jpg_dir: Path) -> tuple[YOLO, list, Path]:
    """Run detection on every JPG tile and return the model, results, and label dir."""
    if not model_path.is_file():
        raise FileNotFoundError(f"model weights not found: {model_path}")
    model = YOLO(str(model_path))
    results = model.predict(
        str(jpg_dir),
        save_txt=True,
        conf=CONF_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        show_conf=False,
        show_labels=False,
    )
    label_dir = Path(results[0].save_dir) / "labels" if results else jpg_dir
    return model, results, label_dir


def count_by_class(model: YOLO, results: list) -> dict[str, int]:
    """Tally detections per class name across all tiles."""
    counts: dict[str, int] = {}
    for result in results:
        detections = result.boxes
        if detections is None:
            continue
        for class_id in detections.cls.tolist():
            name = model.names[int(class_id)]
            counts[name] = counts.get(name, 0) + 1
    return counts


def box_corners_px(
    cx: float, cy: float, w: float, h: float, tile_size: int
) -> list[tuple[float, float]]:
    """Return the four (col, row) pixel corners of a normalized YOLO box.

    Corners run clockwise from the top-left in tile pixel space.
    """
    half_w = w * tile_size / 2
    half_h = h * tile_size / 2
    col = cx * tile_size
    row = cy * tile_size
    return [
        (col - half_w, row - half_h),
        (col + half_w, row - half_h),
        (col + half_w, row + half_h),
        (col - half_w, row + half_h),
    ]


def georeference_detections(
    label_dir: Path, tile_dir: Path, tile_size: int = TILE_SIZE
) -> tuple[list[Polygon], list[int], object]:
    """Convert per-tile detection labels into map-coordinate polygons.

    Each label line holds a class id and a normalized YOLO box. The matching
    tile's affine transform maps pixel corners to map coordinates.
    """
    polygons: list[Polygon] = []
    classes: list[int] = []
    crs = None
    for label_path in sorted(label_dir.glob("*.txt")):
        tile_path = tile_dir / f"{label_path.stem}.tif"
        if not tile_path.is_file():
            continue
        with rasterio.open(tile_path) as tile:
            crs = tile.crs
            frame = pd.read_csv(label_path, sep=" ", header=None, names=DETECT_COLUMNS)
            for _, row in frame.iterrows():
                corners = [
                    xy(tile.transform, py, px)
                    for px, py in box_corners_px(
                        row["cx"], row["cy"], row["w"], row["h"], tile_size
                    )
                ]
                polygons.append(Polygon(corners))
                classes.append(int(row["class"]))
    return polygons, classes, crs


def save_counts(counts: dict[str, int], out_path: Path) -> None:
    pd.DataFrame(sorted(counts.items()), columns=["class", "count"]).to_csv(
        out_path, index=False
    )


def save_detections(
    polygons: list[Polygon], classes: list[int], crs: object, out_path: Path
) -> None:
    gpd.GeoDataFrame({"class": classes}, geometry=polygons, crs=crs).to_file(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orthomosaic", type=Path, help="georeferenced source .tif")
    parser.add_argument("weights", type=Path, help="trained YOLO checkpoint")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_dir = args.work_dir / "tiles"
    jpg_dir = args.work_dir / "tiles_jpg"

    tiles = tile_orthomosaic(args.orthomosaic, tile_dir)
    print(f"wrote {tiles} tiles")
    tiles_to_jpg(tile_dir, jpg_dir)

    model, results, label_dir = detect(args.weights, jpg_dir)
    counts = count_by_class(model, results)
    print(f"counts: {counts}")
    save_counts(counts, args.work_dir / "class_counts.csv")

    polygons, classes, crs = georeference_detections(label_dir, tile_dir)
    save_detections(polygons, classes, crs, args.work_dir / "detections.shp")
    print(f"georeferenced {len(polygons)} detections")


if __name__ == "__main__":
    main()
