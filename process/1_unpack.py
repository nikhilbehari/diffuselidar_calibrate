"""Unpack raw .pkl capture files into structured directories.

Extracts aligned RGB, depth, SPAD histograms, and metadata from the pkl
stream produced by run_calibration.py.

Usage: python 1_unpack.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
PKL_PATH = Path("path/to/your_capture_data.pkl")
OUTPUT_DIR = Path("path/to/processed_output")

DEPTH_CMAP = "magma"
DEPTH_VMIN = 0.0
DEPTH_VMAX = 3000.0
# ───────────────────────────────────────────────────────────────────────────────


def apply_numpy_pickle_compat():
    try:
        import numpy.core.numeric as _numeric
        sys.modules["numpy._core.numeric"] = _numeric
    except Exception:
        pass


def load_pickle_stream(path: Path) -> list:
    objs = []
    with open(path, "rb") as f:
        while True:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break
    return objs


def json_safe(obj, arrays_dir: Path, prefix: str):
    if isinstance(obj, np.ndarray):
        name = f"{prefix}.npy"
        np.save(arrays_dir / name, obj)
        return {"__type__": "ndarray", "path": name,
                "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v, arrays_dir, f"{prefix}.{k}")
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, arrays_dir, f"{prefix}[{i}]")
                for i, v in enumerate(obj)]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return {"__type__": type(obj).__name__, "repr": repr(obj)[:200]}


def extract_histogram(capture: dict) -> np.ndarray | None:
    for k, v in capture.items():
        if isinstance(v, np.ndarray) and "HISTOGRAM" in str(k).upper():
            return v
    for v in capture.values():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 128:
            return v
    return None


def save_rgb(img: np.ndarray, path: Path):
    plt.imsave(path, img[:, :, ::-1])


def save_depth(depth: np.ndarray, npy_path: Path, png_path: Path):
    np.save(npy_path, depth)
    norm = np.clip(depth, DEPTH_VMIN, DEPTH_VMAX)
    norm = (norm - DEPTH_VMIN) / (DEPTH_VMAX - DEPTH_VMIN)
    plt.imsave(png_path, norm, cmap=DEPTH_CMAP)


def unpack(pkl_path: Path, output_dir: Path):
    apply_numpy_pickle_compat()

    objs = load_pickle_stream(pkl_path)
    if not objs:
        raise RuntimeError(f"No objects in {pkl_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "aligned_rgb"
    depth_dir = output_dir / "aligned_depth"
    hist_dir = output_dir / "spad_histogram"
    meta_arrays_dir = output_dir / "metadata_arrays"
    for d in [rgb_dir, depth_dir, hist_dir, meta_arrays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    start = 0
    if isinstance(objs[0], dict) and "metadata" in objs[0]:
        metadata = objs[0]["metadata"]
        safe = json_safe(metadata, meta_arrays_dir, "metadata")
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(safe, f, indent=2)
        start = 1
        print(f"  Saved metadata.json")

    n_captures = 0
    for i, cap in enumerate(objs[start:]):
        if not isinstance(cap, dict):
            continue

        idx_str = f"{i:04d}"

        rs = cap.get("realsense_data")
        if isinstance(rs, dict):
            rgb = rs.get("aligned_rgb_image")
            if rgb is not None:
                save_rgb(rgb, rgb_dir / f"{idx_str}.png")

            depth = rs.get("aligned_depth_image")
            if depth is not None:
                save_depth(depth,
                           depth_dir / f"{idx_str}.npy",
                           depth_dir / f"{idx_str}.png")

        hist = extract_histogram(cap)
        if hist is not None:
            np.save(hist_dir / f"{idx_str}.npy", hist)

        n_captures += 1

    print(f"  Unpacked {n_captures} captures → {output_dir}")
    return n_captures


def main():
    if not PKL_PATH.exists():
        raise FileNotFoundError(
            f"PKL file not found: {PKL_PATH}\n"
            "Set PKL_PATH at the top of this script to your capture .pkl file."
        )

    print(f"Unpacking: {PKL_PATH}")
    unpack(PKL_PATH, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
