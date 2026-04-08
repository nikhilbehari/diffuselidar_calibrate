"""Compute the spatial calibration from processed captures.

Background-subtracts SPAD histograms and computes per-pixel peak response
at each scan position. Outputs a calibration .npz.

Usage: python 3_calibrate.py
"""

import json
import time
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
# Processed directories from Step 1 (unpack)
LARGEPATCH_DIR = Path("path/to/processed_largepatch")
BACKGROUND_DIR = Path("path/to/processed_nopatch")

# Centers JSON from Step 2 (find_centers)
CENTERS_JSON = Path("path/to/all_centers.json")

# Output
OUTPUT_DIR = Path("calibration_output")

# Range mode: "longrange" or "shortrange"
RANGE_MODE = "longrange"
# ───────────────────────────────────────────────────────────────────────────────

BIN_RANGES = {
    "longrange": (15, 35),
    "shortrange": (45, 75),
}

PRINT_EVERY = 200


def bins_for_range(mode: str) -> tuple[int, int]:
    if mode not in BIN_RANGES:
        raise ValueError(f"Unknown range mode: {mode!r}. Use 'longrange' or 'shortrange'.")
    return BIN_RANGES[mode]


def list_npy_ids(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.npy")}


def list_png_ids(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.png")}


def save_calibration_npz(path: Path, responses, xs, ys, cap_ids, meta: dict):
    meta = dict(meta)
    meta["responses_shape"] = list(responses.shape)
    meta["n_captures"] = int(len(cap_ids))
    np.savez_compressed(
        path,
        responses=responses.astype(np.float32),
        xs=xs.astype(np.float32),
        ys=ys.astype(np.float32),
        cap_ids=cap_ids,
        meta_json=np.array(json.dumps(meta, sort_keys=True), dtype="<U"),
    )


def calibrate(
    largepatch_dir: Path,
    background_dir: Path,
    centers_json: Path,
    output_dir: Path,
    range_mode: str,
):
    bin_min, bin_max = bins_for_range(range_mode)
    sl = slice(bin_min, bin_max + 1)

    lp_hist_dir = largepatch_dir / "spad_histogram"
    lp_rgb_dir = largepatch_dir / "aligned_rgb"
    bg_hist_dir = background_dir / "spad_histogram"

    for d, name in [(lp_hist_dir, "largepatch spad_histogram"),
                    (lp_rgb_dir, "largepatch aligned_rgb"),
                    (bg_hist_dir, "background spad_histogram")]:
        if not d.exists():
            raise FileNotFoundError(f"Missing {name}: {d}")

    with open(centers_json, "r") as f:
        centers = json.load(f)["all_centers_by_filename"]

    lp_hist_ids = list_npy_ids(lp_hist_dir)
    lp_rgb_ids = list_png_ids(lp_rgb_dir)
    bg_hist_ids = list_npy_ids(bg_hist_dir)
    center_ids = {Path(k).stem for k in centers.keys()}

    cap_ids = sorted(lp_hist_ids & lp_rgb_ids & bg_hist_ids & center_ids)
    if not cap_ids:
        raise RuntimeError("No valid capture IDs found across all sources.")

    xs = np.empty(len(cap_ids), dtype=np.float32)
    ys = np.empty(len(cap_ids), dtype=np.float32)
    for i, cid in enumerate(cap_ids):
        c = centers[f"{cid}.png"]
        xs[i] = float(c["x"])
        ys[i] = float(c["y"])

    sample = np.load(lp_hist_dir / f"{cap_ids[0]}.npy")
    if sample.ndim != 3:
        raise RuntimeError(f"Expected 3D histogram, got ndim={sample.ndim}")
    Sy, Sx, T = sample.shape
    if (Sy, Sx) != (3, 3):
        raise RuntimeError(f"Expected 3x3 histogram grid, got ({Sy}, {Sx})")
    if bin_max >= T:
        raise RuntimeError(f"bin_max={bin_max} exceeds histogram bins T={T}")

    print(f"Captures: {len(cap_ids)} | Histogram: ({Sy}, {Sx}, {T}) | "
          f"Bins: [{bin_min}, {bin_max}]")

    responses_sub = np.empty((Sy, Sx, len(cap_ids)), dtype=np.float32)
    responses_raw = np.empty((Sy, Sx, len(cap_ids)), dtype=np.float32)

    t0 = time.time()
    for k, cid in enumerate(cap_ids):
        lp = np.load(lp_hist_dir / f"{cid}.npy").astype(np.int32)
        bg = np.load(bg_hist_dir / f"{cid}.npy").astype(np.int32)

        responses_raw[:, :, k] = lp[:, :, sl].max(axis=2).astype(np.float32)

        # Background-subtracted: clip(largepatch - background, >= 0), then peak
        diff = np.maximum(lp[:, :, sl] - bg[:, :, sl], 0)
        responses_sub[:, :, k] = diff.max(axis=2).astype(np.float32)

        if (k + 1) % PRINT_EVERY == 0 or (k + 1) == len(cap_ids):
            dt = time.time() - t0
            rate = (k + 1) / dt if dt > 0 else float("inf")
            print(f"  [{k + 1}/{len(cap_ids)}] {rate:.0f} captures/s")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap_ids_arr = np.array(cap_ids, dtype="<U8")

    base_meta = {
        "range_mode": range_mode,
        "bin_min": bin_min,
        "bin_max": bin_max,
        "largepatch_dir": str(largepatch_dir),
        "background_dir": str(background_dir),
        "centers_json": str(centers_json),
    }

    out_sub = output_dir / f"spad_calib_3x3_{range_mode}_bins{bin_min}_{bin_max}.npz"
    save_calibration_npz(out_sub, responses_sub, xs, ys, cap_ids_arr,
                         {**base_meta, "background_subtracted": True,
                          "response": "max(clip(largepatch - background, >=0)) over bin range"})
    print(f"  Saved: {out_sub}")

    out_raw = output_dir / f"spad_calib_3x3_{range_mode}_bins{bin_min}_{bin_max}_nobgsub.npz"
    save_calibration_npz(out_raw, responses_raw, xs, ys, cap_ids_arr,
                         {**base_meta, "background_subtracted": False,
                          "response": "max(largepatch) over bin range (no background subtraction)"})
    print(f"  Saved: {out_raw}")

    print(f"\nBackground-subtracted: min={responses_sub.min():.3g}, "
          f"max={responses_sub.max():.3g}")
    print(f"Raw: min={responses_raw.min():.3g}, max={responses_raw.max():.3g}")


def main():
    print(f"=== Calibrating 3x3 {RANGE_MODE} ===")
    t0 = time.time()
    calibrate(LARGEPATCH_DIR, BACKGROUND_DIR, CENTERS_JSON, OUTPUT_DIR, RANGE_MODE)
    print(f"\nDone. Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
