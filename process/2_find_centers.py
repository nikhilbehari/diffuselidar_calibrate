"""Find the retroreflective patch center in each RGB frame.

Phase 1: GUI corner labeling → bilinear interpolation of all grid positions.
Phase 2: Hough circle refinement with contour fallback.
Phase 3: Outlier correction with manual GUI fallback.

Usage: python 2_find_centers.py
"""

import csv
import json
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Configuration ──────────────────────────────────────────────────────────────
# Directory of aligned RGB images from Step 1
IMAGE_DIR = Path("path/to/processed_largepatch/aligned_rgb")

# Output directory for center labels
OUTPUT_DIR = Path("path/to/centers_output")

# Grid dimensions (must match run_calibration.py settings)
GRID_COLS = 80
GRID_ROWS = 45

# The grid starts at file index 1 (index 0 is the pre-grid initial capture)
START_INDEX = 1

# Expected patch radius in pixels (tune if your setup differs)
EXPECTED_RADIUS = 17.75
DIAMETER_PX = 2 * EXPECTED_RADIUS
R_TOL = 8                  # radius tolerance for Hough detection
ROI_HALF = 90              # half-size of ROI crop around seed point

# Outlier detection
OUTLIER_WINDOW = 2          # neighbor window for rolling median
MIN_CONF_FOR_SMOOTH = 0.35  # min confidence to include in neighbor median
MIN_CONF_ACCEPT = 0.28      # below this → flag as outlier

# Tighter re-detection for outliers
RERUN_ROI_HALF = 70
RERUN_R_TOL = 5

# Display
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 900
# ───────────────────────────────────────────────────────────────────────────────


# ── Utilities ──────────────────────────────────────────────────────────────────

@dataclass
class Point:
    x: float
    y: float


def zpad(i: int, w: int = 4) -> str:
    return str(i).zfill(w)


def snake_index(row: int, col: int, cols: int, start: int) -> int:

    base = start + row * cols
    return base + (col if row % 2 == 0 else (cols - 1 - col))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bilerp(bl: Point, br: Point, tl: Point, tr: Point, u: float, v: float) -> Point:

    x0, y0 = lerp(bl.x, br.x, u), lerp(bl.y, br.y, u)
    x1, y1 = lerp(tl.x, tr.x, u), lerp(tl.y, tr.y, u)
    return Point(x=lerp(x0, x1, v), y=lerp(y0, y1, v))


def list_images(image_dir: Path) -> list[str]:

    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
    return [f for f in files if f != "0000.png"]


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_centers(output_dir: Path, results: dict[str, dict], fields: list[str]):

    save_json(output_dir / "all_centers.json",
              {"all_centers_by_filename": results})
    rows = list(results.values())
    save_csv(output_dir / "all_centers.csv", rows, fields)


# ── Phase 1: Corner Labeling GUI ───────────────────────────────────────────────

CORNER_TASKS = [
    ("BL", 0, 0),
    ("BR", 0, GRID_COLS - 1),
    ("TL", GRID_ROWS - 1, 0),
    ("TR", GRID_ROWS - 1, GRID_COLS - 1),
]


class CornerLabelGUI:


    def __init__(self, root: tk.Tk, image_dir: Path):
        self.root = root
        self.image_dir = image_dir
        self.tasks = CORNER_TASKS
        self.idx = 0
        self.points: dict[str, Optional[Point]] = {t[0]: None for t in self.tasks}
        self.scale = 1.0
        self.photo = None
        self.dot_id = None
        self.interpolated: dict[str, dict] | None = None

        self._build_ui()
        self._load_current()

    def _build_ui(self):
        self.root.title("Corner Labeling — Click patch center in each corner image")
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)

        self.info = tk.Label(right, text="", justify=tk.LEFT, anchor="w")
        self.info.pack(fill=tk.X, padx=10, pady=(10, 6))

        self.progress = tk.Label(right, text="", justify=tk.LEFT, anchor="w")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))

        btns = tk.Frame(right)
        btns.pack(fill=tk.X, padx=10, pady=6)
        self.prev_btn = tk.Button(btns, text="Prev", command=self._prev)
        self.prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.next_btn = tk.Button(btns, text="Next", command=self._next)
        self.next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        actions = tk.Frame(right)
        actions.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(actions, text="Clear Dot", command=self._clear).pack(fill=tk.X)
        tk.Button(actions, text="Interpolate + Continue",
                  command=self._interpolate_and_close).pack(fill=tk.X, pady=(8, 0))

        self.status = tk.Label(right, text="", justify=tk.LEFT, anchor="w", fg="gray")
        self.status.pack(fill=tk.X, padx=10, pady=(10, 10))

    def _current_task(self):
        return self.tasks[self.idx]

    def _image_path(self, row, col):
        idx = snake_index(row, col, GRID_COLS, START_INDEX)
        return self.image_dir / f"{zpad(idx)}.png"

    def _load_current(self):
        key, row, col = self._current_task()
        path = self._image_path(row, col)
        idx = snake_index(row, col, GRID_COLS, START_INDEX)

        self.info.config(text=f"Corner: {key}  (row={row}, col={col})\n"
                              f"File: {path.name}  (index={idx})")
        done = sum(1 for v in self.points.values() if v is not None)
        self.progress.config(text=f"Labeled: {done}/{len(self.tasks)}")
        self.prev_btn.config(state=tk.NORMAL if self.idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.idx < len(self.tasks) - 1
                             else tk.DISABLED)

        if not path.exists():
            self.canvas.delete("all")
            self.status.config(text=f"Missing: {path}")
            return

        img = Image.open(path).convert("RGB")
        w, h = img.size
        self.scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        dw, dh = int(w * self.scale), int(h * self.scale)
        disp = img.resize((dw, dh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self.canvas.config(width=dw, height=dh)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.dot_id = None

        pt = self.points.get(key)
        if pt is not None:
            self._draw_dot(pt.x * self.scale, pt.y * self.scale)
        self.status.config(text="")

    def _draw_dot(self, xd, yd):
        r = 5
        if self.dot_id is not None:
            self.canvas.delete(self.dot_id)
        self.dot_id = self.canvas.create_oval(xd - r, yd - r, xd + r, yd + r,
                                               outline="red", width=2)

    def _on_click(self, event):
        key = self._current_task()[0]
        self.points[key] = Point(x=event.x / self.scale, y=event.y / self.scale)
        self._draw_dot(event.x, event.y)
        self.status.config(text="Dot set.")
        done = sum(1 for v in self.points.values() if v is not None)
        self.progress.config(text=f"Labeled: {done}/{len(self.tasks)}")
        if self.idx < len(self.tasks) - 1:
            self._next()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_current()

    def _next(self):
        if self.idx < len(self.tasks) - 1:
            self.idx += 1
            self._load_current()

    def _clear(self):
        key = self._current_task()[0]
        self.points[key] = None
        if self.dot_id is not None:
            self.canvas.delete(self.dot_id)
            self.dot_id = None
        self.status.config(text="Cleared.")

    def _interpolate_and_close(self):
        missing = [k for k, v in self.points.items() if v is None]
        if missing:
            self.status.config(text=f"Missing corners: {', '.join(missing)}")
            return

        bl, br, tl, tr = (self.points["BL"], self.points["BR"],
                           self.points["TL"], self.points["TR"])

        result = {}
        for r in range(GRID_ROWS):
            v = 0.0 if GRID_ROWS == 1 else r / (GRID_ROWS - 1)
            for c in range(GRID_COLS):
                u = 0.0 if GRID_COLS == 1 else c / (GRID_COLS - 1)
                pt = bilerp(bl, br, tl, tr, u, v)
                idx = snake_index(r, c, GRID_COLS, START_INDEX)
                fn = f"{zpad(idx)}.png"
                result[fn] = {"row": r, "col": c, "index": idx,
                              "filename": fn, "x": pt.x, "y": pt.y}

        self.interpolated = result
        self.root.quit()
        self.root.destroy()

    def run(self) -> dict[str, dict] | None:
        self.root.mainloop()
        return self.interpolated


def run_phase1(image_dir: Path, output_dir: Path) -> dict[str, dict]:

    rough_path = output_dir / "rough_centers.json"

    if rough_path.exists():
        print("  Phase 1: Found existing rough_centers.json, skipping GUI.")
        with open(rough_path, "r") as f:
            return json.load(f)["all_centers_by_filename"]

    print("  Phase 1: Opening corner labeling GUI...")
    print("           Click the patch center in each of the 4 corner images,")
    print("           then click 'Interpolate + Continue'.")

    root = tk.Tk()
    gui = CornerLabelGUI(root, image_dir)
    result = gui.run()

    if result is None:
        raise RuntimeError("Corner labeling was cancelled.")

    save_json(rough_path, {"all_centers_by_filename": result})
    print(f"  Phase 1: Saved {len(result)} rough centers → {rough_path}")
    return result


# ── Phase 2: Hough Circle Refinement ───────────────────────────────────────────

def crop_roi(img: np.ndarray, cx: float, cy: float, half: int):

    h, w = img.shape[:2]
    x0 = max(0, min(int(round(cx)) - half, w - 1))
    y0 = max(0, min(int(round(cy)) - half, h - 1))
    x1 = max(0, min(int(round(cx)) + half, w - 1))
    y1 = max(0, min(int(round(cy)) + half, h - 1))
    return img[y0:y1 + 1, x0:x1 + 1].copy(), (x0, y0)


def preprocess(roi_bgr: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.bilateralFilter(gray, 7, 50, 50)


def detect_edges(gray: np.ndarray) -> np.ndarray:

    v = np.median(gray)
    lo, hi = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(edges, kernel, iterations=1)


def circle_edge_support(edges: np.ndarray, cx: float, cy: float,
                        r: float, band: int = 2, samples: int = 360) -> float:

    if r <= 0:
        return 0.0
    h, w = edges.shape[:2]
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    px = (cx + r * np.cos(angles)).round().astype(int)
    py = (cy + r * np.sin(angles)).round().astype(int)

    mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not mask.any():
        return 0.0

    hits = 0
    for xi, yi in zip(px[mask], py[mask]):
        x0, x1 = max(0, xi - band), min(w - 1, xi + band)
        y0, y1 = max(0, yi - band), min(h - 1, yi + band)
        if np.any(edges[y0:y1 + 1, x0:x1 + 1] > 0):
            hits += 1
    return hits / mask.sum()


def score_circle(support: float, r: float, r0: float,
                 dist: float, w_r: float = 0.06, w_d: float = 0.0025) -> float:

    return support - w_r * abs(r - r0) - w_d * dist


def hough_detect(gray: np.ndarray, edges: np.ndarray, seed_roi: tuple[float, float],
                 r_min: int, r_max: int, r0: float):

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.1,
        minDist=max(10, min(gray.shape[:2]) // 2),
        param1=140, param2=22, minRadius=r_min, maxRadius=r_max)
    if circles is None:
        return None

    sx, sy = seed_roi
    best = None
    for x, y, r in circles[0].astype(float):
        sup = circle_edge_support(edges, x, y, r)
        sc = score_circle(sup, r, r0, float(np.hypot(x - sx, y - sy)))
        if best is None or sc > best[0]:
            best = (sc, x, y, r, sup)
    return (best[1], best[2], best[3], best[4]) if best else None


def contour_detect(edges: np.ndarray, seed_roi: tuple[float, float],
                   r_min: int, r_max: int, r0: float):

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sx, sy = seed_roi
    best = None

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y, r = float(x), float(y), float(r)
        if r < r_min or r > r_max:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 1:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.35:
            continue

        sup = circle_edge_support(edges, x, y, r)
        sc = score_circle(sup, r, r0, float(np.hypot(x - sx, y - sy))) + 0.15 * circ
        if best is None or sc > best[0]:
            best = (sc, x, y, r, sup)

    return (best[1], best[2], best[3], best[4]) if best else None


def detect_circle(bgr: np.ndarray, seed_xy: tuple[float, float],
                  roi_half: int, r_tol: int,
                  expected_r: float = EXPECTED_RADIUS):

    roi, (x0, y0) = crop_roi(bgr, seed_xy[0], seed_xy[1], roi_half)
    gray = preprocess(roi)
    edges = detect_edges(gray)

    r0 = expected_r
    r_min = int(max(1, round(r0 - r_tol)))
    r_max = int(round(r0 + r_tol))
    seed_roi = (seed_xy[0] - x0, seed_xy[1] - y0)

    result = hough_detect(gray, edges, seed_roi, r_min, r_max, r0)
    if result is None:
        result = contour_detect(edges, seed_roi, r_min, r_max, r0)

    if result is None:
        return None
    rx, ry, rr, conf = result
    return (x0 + rx, y0 + ry, rr, conf)


def run_phase2(image_dir: Path, seeds: dict[str, dict]) -> tuple[
    list[str], list[float], list[float], list[float], list[float], list[str]
]:

    files = list_images(image_dir)
    if not files:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    xs, ys, rs, confs, methods = [], [], [], [], []

    for k, fn in enumerate(files):
        bgr = cv2.imread(str(image_dir / fn), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        seed = seeds.get(fn)
        if seed is None:
            h, w = bgr.shape[:2]
            seed_xy = (w / 2.0, h / 2.0)
        else:
            seed_xy = (float(seed["x"]), float(seed["y"]))

        det = detect_circle(bgr, seed_xy, ROI_HALF, R_TOL)

        if det is None:
            x, y, r, conf, method = seed_xy[0], seed_xy[1], EXPECTED_RADIUS, 0.0, "NONE"
        else:
            x, y, r, conf = det
            method = "HOUGH"

        xs.append(x)
        ys.append(y)
        rs.append(r)
        confs.append(conf)
        methods.append(method)

        if (k + 1) % 200 == 0 or (k + 1) == len(files):
            print(f"    [{k + 1}/{len(files)}]")

    return files, xs, ys, rs, confs, methods


# ── Phase 3: Outlier Correction ────────────────────────────────────────────────

class ManualCenterGUI:


    def __init__(self, root: tk.Tk, bgr: np.ndarray, title: str, default_r: float):
        self.root = root
        self.bgr = bgr
        self.default_r = default_r
        self.scale = 1.0
        self.photo = None
        self.dot_id = None
        self.circle_id = None
        self.center: tuple[float, float] | None = None

        self.root.title(title)
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._click)

        bar = tk.Frame(self.root)
        bar.pack(fill=tk.X)
        self.info = tk.Label(bar, text="Click the patch center, then Confirm.",
                             anchor="w")
        self.info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=6)
        self.confirm_btn = tk.Button(bar, text="Confirm", command=self._confirm,
                                     state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.RIGHT, padx=8, pady=6)

        self._render()

    def _render(self):
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        w, h = img.size
        self.scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        dw, dh = int(w * self.scale), int(h * self.scale)
        disp = img.resize((dw, dh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.config(width=dw, height=dh)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _click(self, event):
        xd, yd = float(event.x), float(event.y)
        self.center = (xd / self.scale, yd / self.scale)
        r = 5
        if self.dot_id:
            self.canvas.delete(self.dot_id)
        if self.circle_id:
            self.canvas.delete(self.circle_id)
        self.dot_id = self.canvas.create_oval(xd - r, yd - r, xd + r, yd + r,
                                               outline="yellow", width=2)
        rr = self.default_r * self.scale
        self.circle_id = self.canvas.create_oval(xd - rr, yd - rr, xd + rr, yd + rr,
                                                  outline="lime", width=2)
        self.confirm_btn.config(state=tk.NORMAL)
        self.info.config(text=f"Center: ({self.center[0]:.1f}, {self.center[1]:.1f})")

    def _confirm(self):
        self.root.quit()

    def run(self) -> tuple[float, float] | None:
        self.root.mainloop()
        self.root.destroy()
        return self.center


def rolling_neighbor_median(xs, ys, rs, confs, i, window, min_conf):

    n = len(xs)
    lo, hi = max(0, i - window), min(n - 1, i + window)
    idxs = [j for j in range(lo, hi + 1) if j != i and confs[j] >= min_conf]
    if len(idxs) < 2:
        return None
    return (float(np.median([xs[j] for j in idxs])),
            float(np.median([ys[j] for j in idxs])),
            float(np.median([rs[j] for j in idxs])))


def run_phase3(image_dir: Path, files, xs, ys, rs, confs, methods):

    n = len(files)

    # Compute residuals vs neighbors
    pos_resids = []
    r_resids = []
    neighbor_meds = []

    for i in range(n):
        med = rolling_neighbor_median(xs, ys, rs, confs, i,
                                      OUTLIER_WINDOW, MIN_CONF_FOR_SMOOTH)
        neighbor_meds.append(med)
        if med is None:
            pos_resids.append(0.0)
            r_resids.append(0.0)
        else:
            mx, my, mr = med
            pos_resids.append(float(np.hypot(xs[i] - mx, ys[i] - my)))
            r_resids.append(abs(rs[i] - mr))

    # Adaptive thresholds (median + 6 * MAD)
    pr = np.array(pos_resids)
    rr = np.array(r_resids)
    pos_thresh = float(np.median(pr) + 6 * max(1e-6, np.median(np.abs(pr - np.median(pr)))))
    r_thresh = float(np.median(rr) + 6 * max(1e-6, np.median(np.abs(rr - np.median(rr)))))

    rerun_count = 0
    for i, fn in enumerate(files):
        med = neighbor_meds[i]
        if med is None:
            continue

        is_low = confs[i] < MIN_CONF_ACCEPT
        is_jump = pos_resids[i] > max(6.0, pos_thresh) or r_resids[i] > max(1.2, r_thresh)
        is_failed = methods[i] == "NONE"

        if not (is_low or is_jump or is_failed):
            continue

        bgr = cv2.imread(str(image_dir / fn), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        mx, my, mr = med

        # Try tighter re-detection seeded from neighbor median
        det = detect_circle(bgr, (mx, my), RERUN_ROI_HALF, RERUN_R_TOL)

        if det is not None:
            x, y, r, conf = det
            if conf < confs[i]:
                continue  # original was better
        else:
            # Manual fallback
            root = tk.Tk()
            gui = ManualCenterGUI(root, bgr,
                                  f"Manual: {fn}  ({i + 1}/{n})", EXPECTED_RADIUS)
            picked = gui.run()
            if picked is None:
                x, y = mx, my  # fall back to neighbor median
            else:
                x, y = picked
            r = EXPECTED_RADIUS
            conf = 1.0

        xs[i] = x
        ys[i] = y
        rs[i] = r
        confs[i] = conf
        methods[i] = "RERUN" if det is not None else "MANUAL"
        rerun_count += 1

    print(f"  Phase 3: Corrected {rerun_count} outliers")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

CSV_FIELDS = ["filename", "x", "y", "r", "method", "confidence", "seed_dist", "r_err"]


def main():
    image_dir = IMAGE_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            "Set IMAGE_DIR at the top of this script to your aligned_rgb/ directory."
        )

    # Phase 1: Rough labeling (GUI)
    print("Phase 1: Rough corner labeling")
    seeds = run_phase1(image_dir, output_dir)

    # Phase 2: Hough refinement
    print("Phase 2: Hough circle refinement")
    files, xs, ys, rs, confs, methods = run_phase2(image_dir, seeds)
    print(f"  Detected {sum(1 for m in methods if m != 'NONE')}/{len(files)} circles")

    # Phase 3: Outlier correction
    print("Phase 3: Outlier correction")
    run_phase3(image_dir, files, xs, ys, rs, confs, methods)

    # Build final results
    results = {}
    for i, fn in enumerate(files):
        seed = seeds.get(fn, {"x": xs[i], "y": ys[i]})
        seed_x, seed_y = float(seed["x"]), float(seed["y"])
        results[fn] = {
            "filename": fn,
            "x": float(xs[i]),
            "y": float(ys[i]),
            "r": float(rs[i]),
            "method": methods[i],
            "confidence": float(confs[i]),
            "seed_dist": float(np.hypot(xs[i] - seed_x, ys[i] - seed_y)),
            "r_err": float(abs(rs[i] - EXPECTED_RADIUS)),
        }

    save_centers(output_dir, results, CSV_FIELDS)
    print(f"\nSaved {len(results)} centers → {output_dir / 'all_centers.json'}")
    print(f"                          → {output_dir / 'all_centers.csv'}")


if __name__ == "__main__":
    main()
