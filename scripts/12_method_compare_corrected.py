"""
Four-method candidate tongue-coating region comparison (corrected version).

Required packages:
    pip install numpy pandas opencv-python scikit-learn scikit-image

Before running, edit PROJECT_ROOT below if your project is stored elsewhere.

Main correction:
    Standard FCM updates centers before recomputing memberships and checking
    convergence. Automatic audits detect the old one-iteration failure mode
    and the old all-identical K-means/FCM masks.
"""

from pathlib import Path
import os
import json
import time
import random
import warnings
from datetime import datetime

os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", message="One of the clusters is empty.*", category=UserWarning)

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage.segmentation import slic

# ============================================================
# 0. Paths
# ============================================================
import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))
INPUT_DIR = PROJECT_ROOT / "data" / "final_images"
FORMAL_D_MASK_DIR = PROJECT_ROOT / "data" / "final_candidate_regions" / "candidate_coating_masks"

# New output directory: the old FCM-bug results are kept unchanged for audit.
RESULT_DIR = PROJECT_ROOT / "results" / "12_method_comparison_A_B_C_D_full877_corrected"
MASK_ROOT = RESULT_DIR / "masks"
OVERLAY_ROOT = RESULT_DIR / "overlays"
CONTACT_DIR = RESULT_DIR / "contact_sheets"

METHOD_NAMES = {
    "A": "KMeans",
    "B": "FCM",
    "C": "Spatial_FCM",
    "D": "SLIC_Spatial_FCM",
}

for p in [RESULT_DIR, MASK_ROOT, OVERLAY_ROOT, CONTACT_DIR]:
    p.mkdir(parents=True, exist_ok=True)
for key in METHOD_NAMES:
    (MASK_ROOT / key).mkdir(parents=True, exist_ok=True)
    (OVERLAY_ROOT / key).mkdir(parents=True, exist_ok=True)

FULL_MANIFEST_CSV = RESULT_DIR / "comparison_877_manifest.csv"
BLIND_MAPPING_CSV = PROJECT_ROOT / "results" / "11_blind_validation_88" / "blind_validation_88_INTERNAL_mapping.csv"
VISUAL_MANIFEST_CSV = RESULT_DIR / "visualization_blind_88_manifest.csv"
METRICS_CSV = RESULT_DIR / "comparison_metrics_long.csv"
SUMMARY_CSV = RESULT_DIR / "method_summary.csv"
PAIRWISE_CSV = RESULT_DIR / "pairwise_region_consistency.csv"
SUMMARY_TXT = RESULT_DIR / "method_comparison_summary.txt"
CONFIG_JSON = RESULT_DIR / "method_comparison_config.json"
FAILED_CSV = RESULT_DIR / "failed_cases.csv"
TABLE2_CSV = RESULT_DIR / "table2_structural_characteristics_corrected.csv"
AUDIT_CSV = RESULT_DIR / "implementation_audit_corrected.csv"

# ============================================================
# 1. Frozen parameters
# ============================================================
EXPECTED_N = 877
BLIND_N = 88
RANDOM_SEED = 42
WORK_MAX_SIDE = 384
ROI_THRESHOLD = 12

SLIC_N_SEGMENTS = 500
SLIC_COMPACTNESS = 7.0
SLIC_SIGMA = 1.0

N_CLUSTERS = 2
FCM_M = 2.0
FCM_MAX_ITER = 200
FCM_TOL = 1e-5
SPATIAL_LAMBDA = 0.90
UNCERTAIN_MEMBERSHIP_THRESHOLD = 0.60

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ============================================================
# 2. IO helpers
# ============================================================
def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_png_unicode(path, image):
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        return False
    encoded.tofile(str(path))
    return True

# ============================================================
# 3. ROI and resize
# ============================================================
def largest_component(mask):
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def extract_tongue_roi(image):
    mask = (image.max(axis=2) > ROI_THRESHOLD).astype(np.uint8)
    mask = largest_component(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return largest_component(mask)


def resize_to_work(image, roi):
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= WORK_MAX_SIDE:
        return image.copy(), roi.copy()
    scale = WORK_MAX_SIDE / max_side
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img, (m > 0).astype(np.uint8)

# ============================================================
# 4. Pixel features and radial map
# ============================================================
def build_pixel_features(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    L = lab[:, :, 0] * 100.0 / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    S = hsv[:, :, 1] / 255.0
    raw = np.stack([L, a, b, S], axis=-1)
    scaled = np.stack([
        np.clip(L / 100.0, 0, 1),
        np.clip(a / 40.0, -2, 2),
        np.clip(b / 50.0, -2, 2),
        np.clip(S, 0, 1),
    ], axis=-1)
    return raw, scaled


def normalized_radial_map(roi):
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return np.zeros(roi.shape, dtype=np.float32)
    cx, cy = xs.mean(), ys.mean()
    equivalent_radius = max(np.sqrt(len(xs) / np.pi), 1e-6)
    yy, xx = np.indices(roi.shape)
    return (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / equivalent_radius).astype(np.float32)

# ============================================================
# 5. FCM core
# ============================================================
def membership_from_centers(X, centers, m):
    eps = 1e-12
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    d = np.maximum(d, eps)
    exponent = 2.0 / (m - 1.0)
    ratio = d[:, :, None] / d[:, None, :]
    return 1.0 / np.sum(ratio ** exponent, axis=2)


def init_centers_kmeans(X):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    km.fit(X)
    return km.cluster_centers_.astype(np.float64)


def run_fcm(X):
    """Standard FCM with convergence checked after a genuine update."""
    centers = init_centers_kmeans(X)
    U_prev = membership_from_centers(X, centers, FCM_M)
    converged = False
    final_delta = np.nan

    for iteration in range(1, FCM_MAX_ITER + 1):
        # 1) Update centers from the current memberships.
        weights = U_prev ** FCM_M
        centers_new = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], 1e-12)

        # 2) Recompute memberships from the UPDATED centers.
        U_new = membership_from_centers(X, centers_new, FCM_M)

        # 3) Only now is the membership change meaningful.
        final_delta = float(np.max(np.abs(U_new - U_prev)))
        centers = centers_new
        U_prev = U_new

        if final_delta < FCM_TOL:
            converged = True
            break

    return U_prev, iteration, converged, final_delta


def pixel_spatial_support(U_base, roi):
    h, w = roi.shape
    U_map = np.zeros((h, w, N_CLUSTERS), dtype=np.float64)
    U_map[roi > 0] = U_base
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    neighbor_count = cv2.filter2D(roi.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    support_map = np.zeros_like(U_map)
    for k in range(N_CLUSTERS):
        neighbor_sum = cv2.filter2D(U_map[:, :, k], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        support_map[:, :, k] = neighbor_sum / np.maximum(neighbor_count, 1.0)
    isolated = (neighbor_count <= 0) & (roi > 0)
    if np.any(isolated):
        support_map[isolated] = U_map[isolated]
    return support_map[roi > 0]


def run_spatial_fcm_pixels(X, roi):
    centers = init_centers_kmeans(X)
    U_prev = membership_from_centers(X, centers, FCM_M)
    converged = False
    final_delta = np.nan
    for iteration in range(1, FCM_MAX_ITER + 1):
        U_base = membership_from_centers(X, centers, FCM_M)
        h = np.maximum(pixel_spatial_support(U_base, roi), 1e-12)
        fused = U_base * (h ** SPATIAL_LAMBDA)
        fused /= np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)
        weights = fused ** FCM_M
        centers_new = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], 1e-12)
        final_delta = float(np.max(np.abs(fused - U_prev)))
        centers, U_prev = centers_new, fused
        if final_delta < FCM_TOL:
            converged = True
            break
    return U_prev, iteration, converged, final_delta

# ============================================================
# 6. Semantic mapping shared by A/B/C
# ============================================================
def semantic_score_for_hard_labels(hard, raw_vectors, radial_vectors):
    scores = []
    for k in range(N_CLUSTERS):
        idx = (hard == k)
        if not np.any(idx):
            scores.append(-np.inf)
            continue
        mean_raw = raw_vectors[idx].mean(axis=0)
        mean_r = float(radial_vectors[idx].mean())
        L, a, b, S = [float(x) for x in mean_raw]
        score = (
            1.00 * (L / 100.0)
            - 1.35 * (a / 40.0)
            + 0.45 * (b / 50.0)
            + 0.25 * (1.0 - S)
            + 0.35 * (1.0 - np.clip(mean_r / 1.2, 0, 1))
        )
        scores.append(float(score))
    return int(np.argmax(scores)), scores

# ============================================================
# 7. Postprocessing shared by all methods
# ============================================================
def postprocess_mask(coating, roi):
    coating = (coating > 0).astype(np.uint8)
    roi = (roi > 0).astype(np.uint8)
    roi_area = int(roi.sum())
    min_area = max(20, int(round(0.001 * roi_area)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(coating, connectivity=8)
    cleaned = np.zeros_like(coating)
    for label in range(1, n):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned[roi == 0] = 0
    return cleaned


def work_mask_to_original(mask_work, roi_original):
    h, w = roi_original.shape
    mask = cv2.resize(mask_work.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return postprocess_mask(mask, roi_original)

# ============================================================
# 8. A: pixel KMeans
# ============================================================
def method_A_kmeans(image, roi):
    img_w, roi_w = resize_to_work(image, roi)
    raw, scaled = build_pixel_features(img_w)
    radial = normalized_radial_map(roi_w)
    X = scaled[roi_w > 0].astype(np.float64)
    raw_v = raw[roi_w > 0].astype(np.float64)
    radial_v = radial[roi_w > 0].astype(np.float64)
    km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    hard = km.fit_predict(X)
    coating_cluster, scores = semantic_score_for_hard_labels(hard, raw_v, radial_v)
    mask_work = np.zeros(roi_w.shape, dtype=np.uint8)
    mask_work[roi_w > 0] = (hard == coating_cluster).astype(np.uint8)
    coating = work_mask_to_original(mask_work, roi)
    body = ((roi > 0) & (coating == 0)).astype(np.uint8)
    return {
        "coating": coating, "body": body,
        "iterations": int(km.n_iter_), "converged": True,
        "final_delta": np.nan, "uncertain_fraction": np.nan,
        "cluster_scores": scores,
    }

# ============================================================
# 9. B: pixel FCM
# ============================================================
def method_B_fcm(image, roi):
    img_w, roi_w = resize_to_work(image, roi)
    raw, scaled = build_pixel_features(img_w)
    radial = normalized_radial_map(roi_w)
    X = scaled[roi_w > 0].astype(np.float64)
    raw_v = raw[roi_w > 0].astype(np.float64)
    radial_v = radial[roi_w > 0].astype(np.float64)
    U, iterations, converged, final_delta = run_fcm(X)
    hard = np.argmax(U, axis=1)
    coating_cluster, scores = semantic_score_for_hard_labels(hard, raw_v, radial_v)
    mask_work = np.zeros(roi_w.shape, dtype=np.uint8)
    mask_work[roi_w > 0] = (hard == coating_cluster).astype(np.uint8)
    coating = work_mask_to_original(mask_work, roi)
    body = ((roi > 0) & (coating == 0)).astype(np.uint8)
    uncertain_fraction = float((np.max(U, axis=1) < UNCERTAIN_MEMBERSHIP_THRESHOLD).mean())
    return {
        "coating": coating, "body": body,
        "iterations": iterations, "converged": converged,
        "final_delta": final_delta, "uncertain_fraction": uncertain_fraction,
        "cluster_scores": scores,
    }

# ============================================================
# 10. C: pixel Spatial FCM
# ============================================================
def method_C_spatial_fcm(image, roi):
    img_w, roi_w = resize_to_work(image, roi)
    raw, scaled = build_pixel_features(img_w)
    radial = normalized_radial_map(roi_w)
    X = scaled[roi_w > 0].astype(np.float64)
    raw_v = raw[roi_w > 0].astype(np.float64)
    radial_v = radial[roi_w > 0].astype(np.float64)
    U, iterations, converged, final_delta = run_spatial_fcm_pixels(X, roi_w)
    hard = np.argmax(U, axis=1)
    coating_cluster, scores = semantic_score_for_hard_labels(hard, raw_v, radial_v)
    mask_work = np.zeros(roi_w.shape, dtype=np.uint8)
    mask_work[roi_w > 0] = (hard == coating_cluster).astype(np.uint8)
    coating = work_mask_to_original(mask_work, roi)
    body = ((roi > 0) & (coating == 0)).astype(np.uint8)
    uncertain_fraction = float((np.max(U, axis=1) < UNCERTAIN_MEMBERSHIP_THRESHOLD).mean())
    return {
        "coating": coating, "body": body,
        "iterations": iterations, "converged": converged,
        "final_delta": final_delta, "uncertain_fraction": uncertain_fraction,
        "cluster_scores": scores,
    }

# ============================================================
# 11. D: SLIC + superpixel Spatial FCM (formal method)
# ============================================================
def run_slic(image, roi):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return slic(
        rgb,
        n_segments=SLIC_N_SEGMENTS,
        compactness=SLIC_COMPACTNESS,
        sigma=SLIC_SIGMA,
        start_label=0,
        mask=roi.astype(bool),
        enforce_connectivity=True,
        channel_axis=-1,
        convert2lab=True,
    )


def superpixel_features(labels, roi, raw, scaled, radial):
    valid_labels = np.unique(labels[roi > 0])
    valid_labels = valid_labels[valid_labels >= 0]
    X, raw_means, radial_means, areas = [], [], [], []
    label_to_index = {}
    for label in valid_labels:
        m = (labels == label) & (roi > 0)
        n = int(m.sum())
        if n == 0:
            continue
        label_to_index[int(label)] = len(X)
        X.append(scaled[m].mean(axis=0))
        raw_means.append(raw[m].mean(axis=0))
        radial_means.append(float(radial[m].mean()))
        areas.append(n)
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(raw_means, dtype=np.float64),
        np.asarray(radial_means, dtype=np.float64),
        np.asarray(areas, dtype=np.float64),
        label_to_index,
    )


def build_adjacency(labels, roi, label_to_index):
    adjacency = [set() for _ in range(len(label_to_index))]
    left, right = labels[:, :-1], labels[:, 1:]
    valid_h = (roi[:, :-1] > 0) & (roi[:, 1:] > 0) & (left != right) & (left >= 0) & (right >= 0)
    pairs_h = np.stack([left[valid_h], right[valid_h]], axis=1) if np.any(valid_h) else np.empty((0, 2), dtype=np.int32)
    up, down = labels[:-1, :], labels[1:, :]
    valid_v = (roi[:-1, :] > 0) & (roi[1:, :] > 0) & (up != down) & (up >= 0) & (down >= 0)
    pairs_v = np.stack([up[valid_v], down[valid_v]], axis=1) if np.any(valid_v) else np.empty((0, 2), dtype=np.int32)
    if len(pairs_h) + len(pairs_v) > 0:
        pairs = np.unique(np.sort(np.concatenate([pairs_h, pairs_v], axis=0), axis=1), axis=0)
        for a, b in pairs:
            a, b = int(a), int(b)
            if a in label_to_index and b in label_to_index:
                ia, ib = label_to_index[a], label_to_index[b]
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)
    return adjacency


def superpixel_spatial_support(U_base, adjacency):
    h = np.zeros_like(U_base)
    for i, neighbors in enumerate(adjacency):
        if len(neighbors) == 0:
            h[i] = U_base[i]
        else:
            idx = np.fromiter(neighbors, dtype=np.int64)
            h[i] = U_base[idx].mean(axis=0)
    return h


def run_spatial_fcm_superpixels(X, adjacency):
    centers = init_centers_kmeans(X)
    U_prev = membership_from_centers(X, centers, FCM_M)
    converged = False
    final_delta = np.nan
    for iteration in range(1, FCM_MAX_ITER + 1):
        U_base = membership_from_centers(X, centers, FCM_M)
        h = np.maximum(superpixel_spatial_support(U_base, adjacency), 1e-12)
        fused = U_base * (h ** SPATIAL_LAMBDA)
        fused /= np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)
        weights = fused ** FCM_M
        centers_new = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], 1e-12)
        final_delta = float(np.max(np.abs(fused - U_prev)))
        centers, U_prev = centers_new, fused
        if final_delta < FCM_TOL:
            converged = True
            break
    return U_prev, iteration, converged, final_delta


def semantic_score_superpixels(hard, raw_means, radial_means, areas):
    scores = []
    for k in range(N_CLUSTERS):
        idx = (hard == k)
        if not np.any(idx):
            scores.append(-np.inf)
            continue
        w = areas[idx] / areas[idx].sum()
        mean_raw = np.sum(raw_means[idx] * w[:, None], axis=0)
        mean_r = float(np.sum(radial_means[idx] * w))
        L, a, b, S = [float(x) for x in mean_raw]
        score = (
            1.00 * (L / 100.0)
            - 1.35 * (a / 40.0)
            + 0.45 * (b / 50.0)
            + 0.25 * (1.0 - S)
            + 0.35 * (1.0 - np.clip(mean_r / 1.2, 0, 1))
        )
        scores.append(float(score))
    return int(np.argmax(scores)), scores


def method_D_slic_spatial_fcm(image, roi):
    img_w, roi_w = resize_to_work(image, roi)
    raw, scaled = build_pixel_features(img_w)
    radial = normalized_radial_map(roi_w)
    labels = run_slic(img_w, roi_w)
    X, raw_means, radial_means, areas, label_to_index = superpixel_features(labels, roi_w, raw, scaled, radial)
    if len(X) < 2:
        raise RuntimeError("D: valid superpixels < 2")
    adjacency = build_adjacency(labels, roi_w, label_to_index)
    U, iterations, converged, final_delta = run_spatial_fcm_superpixels(X, adjacency)
    hard = np.argmax(U, axis=1)
    coating_cluster, scores = semantic_score_superpixels(hard, raw_means, radial_means, areas)
    mask_work = np.zeros(roi_w.shape, dtype=np.uint8)
    for label, idx in label_to_index.items():
        if hard[idx] == coating_cluster:
            mask_work[labels == label] = 1
    mask_work[roi_w == 0] = 0
    coating = work_mask_to_original(mask_work, roi)
    body = ((roi > 0) & (coating == 0)).astype(np.uint8)
    uncertain_fraction = float(areas[np.max(U, axis=1) < UNCERTAIN_MEMBERSHIP_THRESHOLD].sum() / max(areas.sum(), 1.0))
    return {
        "coating": coating, "body": body,
        "iterations": iterations, "converged": converged,
        "final_delta": final_delta, "uncertain_fraction": uncertain_fraction,
        "cluster_scores": scores, "n_superpixels": int(len(X)),
    }

# ============================================================
# 12. Structural metrics (no ground truth)
# ============================================================
def component_metrics(mask):
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return 0, 0.0, 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest_fraction = float(areas.max() / area)
    tiny_threshold = max(20, int(round(0.001 * area)))
    tiny_fraction = float(areas[areas < tiny_threshold].sum() / area) if np.any(areas < tiny_threshold) else 0.0
    return int(len(areas)), largest_fraction, tiny_fraction


def perimeter_metrics(mask):
    binary = (mask > 0).astype(np.uint8)
    area = float(binary.sum())
    if area <= 0:
        return 0.0, np.nan, np.nan
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    normalized_perimeter = float(perimeter / max(np.sqrt(area), 1e-12))
    compactness = float(4.0 * np.pi * area / max(perimeter ** 2, 1e-12))
    return perimeter, normalized_perimeter, compactness


def dice_score(a, b):
    a, b = (a > 0), (b > 0)
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    return 1.0 if den == 0 else float(2.0 * inter / den)


def iou_score(a, b):
    a, b = (a > 0), (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)

# ============================================================
# 13. Visualization
# ============================================================
def overlay_mask(image, mask):
    out = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def fit_image(image, width=240, height=190):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas


def build_contact_pages(visual_df, image_lookup, metrics_df, cases_per_page=10):
    labels = ["Original", "A KMeans", "B FCM", "C Spatial FCM", "D SLIC+Spatial FCM"]
    method_keys = [None, "A", "B", "C", "D"]
    thumb_w, thumb_h, text_h = 240, 190, 46
    row_h = thumb_h + text_h
    page_w = thumb_w * 5
    n_pages = int(np.ceil(len(visual_df) / cases_per_page))

    for page in range(n_pages):
        subset = visual_df.iloc[page*cases_per_page:(page+1)*cases_per_page]
        sheet = np.full((len(subset) * row_h, page_w, 3), 255, dtype=np.uint8)
        for r, (_, srow) in enumerate(subset.iterrows()):
            image_id = srow["image_id"]
            image = imread_unicode(image_lookup[image_id])
            if image is None:
                continue
            for c, key in enumerate(method_keys):
                if key is None:
                    view = image
                else:
                    mask_img = imread_unicode(MASK_ROOT / key / f"{image_id}.png")
                    mask = (mask_img[:, :, 0] > 0).astype(np.uint8)
                    view = overlay_mask(image, mask)
                thumb = fit_image(view, thumb_w, thumb_h)
                x0, y0 = c * thumb_w, r * row_h
                sheet[y0:y0+thumb_h, x0:x0+thumb_w] = thumb
                cv2.putText(sheet, labels[c], (x0+5, y0+17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)
            y_text = r * row_h + thumb_h + 18
            label_id = str(srow.get("blind_id", f"#{int(srow['sample_order']):02d}"))
            cv2.putText(sheet, f"{label_id}", (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)
            drow = metrics_df[(metrics_df.image_id == image_id) & (metrics_df.method == "D")].iloc[0]
            cv2.putText(sheet, f"D coating={drow.coating_ratio:.3f} | D converged={bool(drow.converged)}", (5, y_text+19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)
        out_path = CONTACT_DIR / f"comparison_page_{page+1:02d}.jpg"
        ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            encoded.tofile(str(out_path))

# ============================================================
# 14. Dataset scan + frozen 88-case visualization subset
# ============================================================
image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
if len(image_paths) != EXPECTED_N:
    raise RuntimeError(f"Expected {EXPECTED_N} formal images, found {len(image_paths)}")

image_lookup = {p.stem: p for p in image_paths}
all_ids = sorted(image_lookup.keys())

# Full 877-case manifest: ALL cases enter A/B/C/D quantitative comparison.
full_manifest_df = pd.DataFrame({
    "dataset_order": range(1, EXPECTED_N + 1),
    "image_id": all_ids,
    "input_path": [str(image_lookup[x]) for x in all_ids],
})
full_manifest_df.to_csv(FULL_MANIFEST_CSV, index=False, encoding="utf-8-sig")

# Frozen 88-case subset from Step 11 is used only for contact-sheet visualization
# and later human/VLM review. It does NOT determine the quantitative A-D sample.
if not BLIND_MAPPING_CSV.exists():
    raise FileNotFoundError(f"Frozen blind mapping not found: {BLIND_MAPPING_CSV}")

blind_df = pd.read_csv(BLIND_MAPPING_CSV, dtype=str)
if "blind_id" not in blind_df.columns or "image_id" not in blind_df.columns:
    raise RuntimeError("blind_validation_88_INTERNAL_mapping.csv must contain blind_id and image_id")

blind_df["blind_id"] = blind_df["blind_id"].astype(str)
blind_df["image_id"] = blind_df["image_id"].astype(str)
blind_df = blind_df.sort_values("blind_id").reset_index(drop=True)

if len(blind_df) != BLIND_N or blind_df["image_id"].nunique() != BLIND_N:
    raise RuntimeError(f"Frozen blind subset must contain exactly {BLIND_N} unique cases; found {len(blind_df)} rows / {blind_df['image_id'].nunique()} unique")

missing_blind = sorted(set(blind_df["image_id"]) - set(all_ids))
if missing_blind:
    raise RuntimeError("Frozen blind cases missing from formal 877 dataset:\n" + "\n".join(missing_blind))

visual_df = pd.DataFrame({
    "sample_order": range(1, BLIND_N + 1),
    "blind_id": blind_df["blind_id"],
    "image_id": blind_df["image_id"],
    "input_path": [str(image_lookup[x]) for x in blind_df["image_id"]],
})
visual_df.to_csv(VISUAL_MANIFEST_CSV, index=False, encoding="utf-8-sig")

# ============================================================
# 15. Save config
# ============================================================
config = {
    "created_at": datetime.now().isoformat(),
    "formal_dataset_n": EXPECTED_N,
    "comparison_n": EXPECTED_N,
    "comparison_rule": "all 877 frozen original images are included in A/B/C/D quantitative comparison",
    "visualization_subset_n": BLIND_N,
    "visualization_subset_source": str(BLIND_MAPPING_CSV),
    "visualization_rule": "the frozen Step-11 88-case subset is used only for contact-sheet visualization and later human/VLM review",
    "random_seed": RANDOM_SEED,
    "work_max_side": WORK_MAX_SIDE,
    "roi": "max RGB > 12 + largest 8-connected component + 5x5 closing",
    "features": ["CIELAB L*", "CIELAB a*", "CIELAB b*", "HSV saturation"],
    "feature_scaling": ["L*/100", "clip(a*/40,-2,2)", "clip(b*/50,-2,2)", "S"],
    "semantic_mapping": "same coating semantic score for A/B/C/D",
    "common_postprocessing": "remove components < max(20 px, 0.1% ROI) + 5x5 closing",
    "A": "pixel KMeans, k=2",
    "B": "pixel FCM, k=2; corrected update order (centers -> memberships -> delta)",
    "C": "pixel Spatial FCM with 4-neighbor support, k=2",
    "D": "SLIC + superpixel Spatial FCM, k=2",
    "slic": {"n_segments": SLIC_N_SEGMENTS, "compactness": SLIC_COMPACTNESS, "sigma": SLIC_SIGMA},
    "fcm": {"m": FCM_M, "max_iter": FCM_MAX_ITER, "tol": FCM_TOL, "spatial_lambda": SPATIAL_LAMBDA},
    "important": "No metric in this experiment is segmentation accuracy because expert pixel-level ground truth is unavailable. Structural metrics describe spatial coherence/fragmentation only.",
}
with open(CONFIG_JSON, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# ============================================================
# 16. Run A-D on ALL 877 formal cases
# ============================================================
method_functions = {
    "A": method_A_kmeans,
    "B": method_B_fcm,
    "C": method_C_spatial_fcm,
    "D": method_D_slic_spatial_fcm,
}

records = []
failed = []
start_all = time.time()
print("=" * 110)
print("A/B/C/D full-dataset method comparison")
print("=" * 110)
print(f"Formal dataset: {EXPECTED_N}")
print(f"Quantitative comparison: ALL {EXPECTED_N} formal cases")
print(f"Frozen visualization / human-VLM subset: {BLIND_N}")
print(f"Random seed: {RANDOM_SEED}")
print("A=KMeans | B=FCM | C=Spatial FCM | D=SLIC+Spatial FCM")
print("=" * 110)

for sample_order, image_id in enumerate(all_ids, start=1):
    image = imread_unicode(image_lookup[image_id])
    if image is None:
        failed.append({"sample_order": sample_order, "image_id": image_id, "method": "ALL", "error": "image read failed"})
        continue
    roi = extract_tongue_roi(image)
    if roi.sum() == 0:
        failed.append({"sample_order": sample_order, "image_id": image_id, "method": "ALL", "error": "empty ROI"})
        continue
    roi_area = float(roi.sum())

    for method_key, func in method_functions.items():
        t0 = time.time()
        try:
            result = func(image, roi)
            coating, body = result["coating"], result["body"]
            coating_area = float(coating.sum())
            body_area = float(body.sum())
            c_comp, c_largest, c_tiny = component_metrics(coating)
            b_comp, b_largest, b_tiny = component_metrics(body)
            perimeter, norm_perimeter, compactness = perimeter_metrics(coating)

            mask_path = MASK_ROOT / method_key / f"{image_id}.png"
            overlay_path = OVERLAY_ROOT / method_key / f"{image_id}.png"
            if not imwrite_png_unicode(mask_path, coating * 255):
                raise RuntimeError("mask save failed")
            if not imwrite_png_unicode(overlay_path, overlay_mask(image, coating)):
                raise RuntimeError("overlay save failed")

            formal_d_dice = np.nan
            if method_key == "D":
                formal_mask_path = FORMAL_D_MASK_DIR / f"{image_id}.png"
                formal_img = imread_unicode(formal_mask_path)
                if formal_img is None:
                    raise RuntimeError(f"formal D mask missing: {formal_mask_path}")
                formal_mask = (formal_img[:, :, 0] > 0).astype(np.uint8)
                formal_d_dice = dice_score(coating, formal_mask)

            records.append({
                "sample_order": sample_order,
                "image_id": image_id,
                "method": method_key,
                "method_name": METHOD_NAMES[method_key],
                "success": True,
                "roi_area": int(roi_area),
                "coating_area": int(coating_area),
                "body_area": int(body_area),
                "coating_ratio": coating_area / roi_area,
                "body_ratio": body_area / roi_area,
                "coating_components": c_comp,
                "coating_largest_component_fraction": c_largest,
                "coating_tiny_component_fraction": c_tiny,
                "body_components": b_comp,
                "body_largest_component_fraction": b_largest,
                "body_tiny_component_fraction": b_tiny,
                "coating_perimeter": perimeter,
                "coating_normalized_perimeter": norm_perimeter,
                "coating_compactness": compactness,
                "iterations": result["iterations"],
                "converged": result["converged"],
                "final_delta": result["final_delta"],
                "uncertain_fraction": result["uncertain_fraction"],
                "formal_D_reproduction_dice": formal_d_dice,
                "runtime_seconds": time.time() - t0,
                "error": "",
            })
            print(f"[{sample_order:03d}/{EXPECTED_N}] {image_id} | {method_key} OK | {time.time()-t0:.2f}s | Coat={coating_area/roi_area:.3f} | Conv={result['converged']}")
        except Exception as e:
            failed.append({"sample_order": sample_order, "image_id": image_id, "method": method_key, "error": str(e)})
            print(f"[{sample_order:03d}/{EXPECTED_N}] {image_id} | {method_key} FAILED | {e}")

    pd.DataFrame(records).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed).to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")

metrics_df = pd.DataFrame(records)
failed_df = pd.DataFrame(failed)

# Final complete snapshots (the loop above also saves checkpoints).
metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
failed_df.to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")

# ============================================================
# 17. Pairwise consistency among methods (descriptive only)
# ============================================================
pairwise_records = []
method_keys = ["A", "B", "C", "D"]
for image_id in all_ids:
    masks = {}
    for k in method_keys:
        mimg = imread_unicode(MASK_ROOT / k / f"{image_id}.png")
        if mimg is not None:
            masks[k] = (mimg[:, :, 0] > 0).astype(np.uint8)
    for i in range(len(method_keys)):
        for j in range(i+1, len(method_keys)):
            a, b = method_keys[i], method_keys[j]
            if a in masks and b in masks:
                pairwise_records.append({
                    "image_id": image_id,
                    "method_1": a,
                    "method_2": b,
                    "dice": dice_score(masks[a], masks[b]),
                    "iou": iou_score(masks[a], masks[b]),
                })
pairwise_df = pd.DataFrame(pairwise_records)
pairwise_df.to_csv(PAIRWISE_CSV, index=False, encoding="utf-8-sig")

# ============================================================
# 17A. Corrected-FCM implementation audit
# ============================================================
fcm_df = metrics_df[metrics_df["method"] == "B"].copy()
fcm_iteration_median = float(fcm_df["iterations"].median()) if len(fcm_df) else np.nan
fcm_iteration_max = int(fcm_df["iterations"].max()) if len(fcm_df) else 0
fcm_one_iteration_n = int((fcm_df["iterations"] <= 1).sum()) if len(fcm_df) else 0
fcm_zero_delta_n = int(
    np.isclose(fcm_df["final_delta"], 0.0, rtol=0.0, atol=0.0).sum()
) if len(fcm_df) else 0

if len(pairwise_df):
    ab_df = pairwise_df[
        (pairwise_df["method_1"] == "A") &
        (pairwise_df["method_2"] == "B")
    ].copy()
else:
    ab_df = pd.DataFrame()

if len(ab_df):
    ab_exact_mask = (
        np.isclose(ab_df["dice"], 1.0, rtol=0.0, atol=1e-12) &
        np.isclose(ab_df["iou"], 1.0, rtol=0.0, atol=1e-12)
    )
    ab_exact_same_n = int(ab_exact_mask.sum())
    ab_mean_dice = float(ab_df["dice"].mean())
    ab_median_dice = float(ab_df["dice"].median())
else:
    ab_exact_same_n = 0
    ab_mean_dice = np.nan
    ab_median_dice = np.nan

fcm_count_ok = len(fcm_df) == EXPECTED_N
fcm_not_all_one_iteration = not (
    len(fcm_df) == EXPECTED_N and fcm_one_iteration_n == EXPECTED_N
)
ab_count_ok = len(ab_df) == EXPECTED_N
ab_not_all_identical = not (
    len(ab_df) == EXPECTED_N and ab_exact_same_n == EXPECTED_N
)
fcm_audit_ok = (
    fcm_count_ok and fcm_not_all_one_iteration and
    ab_count_ok and ab_not_all_identical
)

audit_df = pd.DataFrame([
    {
        "check": "FCM result count",
        "value": len(fcm_df),
        "expected": EXPECTED_N,
        "passed": fcm_count_ok,
    },
    {
        "check": "FCM median iterations",
        "value": fcm_iteration_median,
        "expected": "descriptive; old faulty run was 1",
        "passed": fcm_not_all_one_iteration,
    },
    {
        "check": "FCM iterations <= 1",
        "value": fcm_one_iteration_n,
        "expected": f"not {EXPECTED_N}/{EXPECTED_N}",
        "passed": fcm_not_all_one_iteration,
    },
    {
        "check": "FCM final_delta exactly zero",
        "value": fcm_zero_delta_n,
        "expected": "descriptive only",
        "passed": True,
    },
    {
        "check": "A-B pairwise result count",
        "value": len(ab_df),
        "expected": EXPECTED_N,
        "passed": ab_count_ok,
    },
    {
        "check": "A-B exactly identical masks",
        "value": ab_exact_same_n,
        "expected": f"not {EXPECTED_N}/{EXPECTED_N}",
        "passed": ab_not_all_identical,
    },
    {
        "check": "A-B mean Dice",
        "value": ab_mean_dice,
        "expected": "descriptive only",
        "passed": True,
    },
    {
        "check": "A-B median Dice",
        "value": ab_median_dice,
        "expected": "descriptive only",
        "passed": True,
    },
])
audit_df.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")

print("\n" + "=" * 110)
print("CORRECTED FCM IMPLEMENTATION AUDIT")
print("=" * 110)
print(f"FCM records                 : {len(fcm_df)}/{EXPECTED_N}")
print(f"FCM median iterations       : {fcm_iteration_median:.1f}")
print(f"FCM maximum iterations      : {fcm_iteration_max}")
print(f"FCM iterations <= 1         : {fcm_one_iteration_n}/{len(fcm_df)}")
print(f"FCM final_delta == 0         : {fcm_zero_delta_n}/{len(fcm_df)}")
print(f"A-B exact same masks         : {ab_exact_same_n}/{len(ab_df)}")
print(f"A-B mean Dice                : {ab_mean_dice:.6f}")
print(f"A-B median Dice              : {ab_median_dice:.6f}")
print(f"FCM implementation audit     : {'PASS' if fcm_audit_ok else 'FAILED'}")
print("=" * 110)

# ============================================================
# 18. Method-level descriptive summary
# ============================================================
summary_rows = []
for key in method_keys:
    g = metrics_df[metrics_df.method == key].copy()
    if len(g) == 0:
        continue
    summary_rows.append({
        "method": key,
        "method_name": METHOD_NAMES[key],
        "n": len(g),
        "converged_n": int(g["converged"].fillna(False).sum()),
        "median_coating_ratio": g["coating_ratio"].median(),
        "iqr_coating_ratio": g["coating_ratio"].quantile(.75) - g["coating_ratio"].quantile(.25),
        "median_coating_components": g["coating_components"].median(),
        "median_largest_component_fraction": g["coating_largest_component_fraction"].median(),
        "median_normalized_perimeter": g["coating_normalized_perimeter"].median(),
        "median_compactness": g["coating_compactness"].median(),
        "median_uncertain_fraction": g["uncertain_fraction"].median(),
        "median_runtime_seconds": g["runtime_seconds"].median(),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")


def median_q1_q3(series, decimals=4):
    """Format a numeric series as median (Q1-Q3) for the manuscript table."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "Not applicable"
    median = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    if decimals == 0:
        return f"{median:.0f} ({q1:.0f}-{q3:.0f})"
    return f"{median:.{decimals}f} ({q1:.{decimals}f}-{q3:.{decimals}f})"


# Paper-ready corrected values corresponding to Table 2.
table2_rows = []
for key in method_keys:
    g = metrics_df[metrics_df["method"] == key].copy()
    if len(g) == 0:
        continue

    display_name = {
        "A": "K-means",
        "B": "FCM",
        "C": "Spatial FCM",
        "D": "SR-SFCM",
    }[key]

    table2_rows.append({
        "Method": display_name,
        "Tongue coating coverage ratio, median (Q1-Q3)": median_q1_q3(
            g["coating_ratio"], 4
        ),
        "Number of connected components, median (Q1-Q3)": median_q1_q3(
            g["coating_components"], 0
        ),
        "Largest component fraction, median (Q1-Q3)": median_q1_q3(
            g["coating_largest_component_fraction"], 4
        ),
        "Normalized perimeter, median (Q1-Q3)": median_q1_q3(
            g["coating_normalized_perimeter"], 4
        ),
        "Uncertainty fraction, median (Q1-Q3)": (
            "Not applicable"
            if key == "A"
            else median_q1_q3(g["uncertain_fraction"], 4)
        ),
    })

table2_df = pd.DataFrame(table2_rows)
table2_df.to_csv(TABLE2_CSV, index=False, encoding="utf-8-sig")

# ============================================================
# 19. D reproduction check
# ============================================================
d_df = metrics_df[metrics_df.method == "D"].copy()
d_repro_min = float(d_df["formal_D_reproduction_dice"].min()) if len(d_df) else np.nan
d_repro_median = float(d_df["formal_D_reproduction_dice"].median()) if len(d_df) else np.nan

# ============================================================
# 20. Contact pages
# ============================================================
build_contact_pages(visual_df, image_lookup, metrics_df, cases_per_page=10)

# ============================================================
# 21. Final summary
# ============================================================
expected_records = EXPECTED_N * 4
success_records = len(metrics_df)
all_methods_complete = (success_records == expected_records and len(failed_df) == 0)
all_fuzzy_converged = True
for key in ["B", "C", "D"]:
    g = metrics_df[metrics_df.method == key]
    if len(g) != EXPECTED_N or not bool(g["converged"].all()):
        all_fuzzy_converged = False

# D should reproduce the formal Step 05 output nearly exactly.
d_reproduction_ok = bool(len(d_df) == EXPECTED_N and np.isfinite(d_repro_min) and d_repro_min >= 0.999999)
final_pass = (
    all_methods_complete and
    all_fuzzy_converged and
    d_reproduction_ok and
    fcm_audit_ok
)

total_runtime = time.time() - start_all

lines = [
    "=" * 110,
    "A/B/C/D Candidate-Region Method Comparison",
    "=" * 110,
    f"Frozen formal dataset: {EXPECTED_N}",
    f"Quantitative comparison cases: {EXPECTED_N} (all frozen formal cases)",
    f"Frozen visualization/human-VLM subset: {BLIND_N}",
    f"Random seed: {RANDOM_SEED}",
    f"Expected method runs: {expected_records}",
    f"Successful method runs: {success_records}",
    f"Failed method runs: {len(failed_df)}",
    "",
    "METHOD DEFINITIONS",
    "-" * 110,
    "A: pixel-level KMeans (k=2)",
    "B: pixel-level FCM (k=2)",
    "C: pixel-level Spatial FCM with 4-neighbor membership support (k=2)",
    "D: SLIC + superpixel-level Spatial FCM (formal method, k=2)",
    "",
    "All four methods use the same ROI, pixel features, semantic cluster mapping, and final postprocessing.",
    "",
    "CONVERGENCE",
    "-" * 110,
    f"B/C/D all converged: {all_fuzzy_converged}",
    "",
    "CORRECTED FCM IMPLEMENTATION AUDIT",
    "-" * 110,
    f"FCM result count: {len(fcm_df)}/{EXPECTED_N}",
    f"FCM median iterations: {fcm_iteration_median:.1f}",
    f"FCM maximum iterations: {fcm_iteration_max}",
    f"FCM iterations <= 1: {fcm_one_iteration_n}/{len(fcm_df)}",
    f"FCM final_delta exactly zero: {fcm_zero_delta_n}/{len(fcm_df)}",
    f"A-B exact same masks: {ab_exact_same_n}/{len(ab_df)}",
    f"A-B mean Dice: {ab_mean_dice:.9f}",
    f"A-B median Dice: {ab_median_dice:.9f}",
    f"FCM implementation audit passed: {fcm_audit_ok}",
    "",
    "FORMAL D REPRODUCTION CHECK",
    "-" * 110,
    f"D vs Step-05 formal-mask minimum Dice: {d_repro_min:.9f}",
    f"D vs Step-05 formal-mask median Dice: {d_repro_median:.9f}",
    f"D reproduction check passed: {d_reproduction_ok}",
    "",
    "METHOD SUMMARY",
    "-" * 110,
    summary_df.to_string(index=False),
    "",
    "IMPORTANT INTERPRETATION",
    "-" * 110,
    "1. These are no-ground-truth structural comparisons; they do not measure segmentation accuracy.",
    "2. Component count, largest-component fraction, normalized perimeter, and compactness describe spatial coherence/fragmentation only.",
    "3. A smoother or more connected mask is not automatically more medically correct.",
    "4. All 877 frozen formal cases enter the quantitative A/B/C/D comparison; there is no result-based subsampling.",
    "5. The frozen 88-case Step-11 subset is used only for visualization and later human/VLM review, and must not be altered according to results.",
    "6. The formal 877-case dataset and Step-05 D outputs remain unchanged.",
    "",
    f"Total runtime: {total_runtime:.2f} s",
    "",
    "FINAL CHECK: PASS" if final_pass else "FINAL CHECK: FAILED",
    "=" * 110,
]
with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n" + "=" * 110)
print("A/B/C/D full-dataset comparison completed")
print("=" * 110)
print(f"Successful method runs: {success_records}/{expected_records}")
print(f"Failures: {len(failed_df)}")
print(f"B/C/D all converged: {all_fuzzy_converged}")
print(f"D formal reproduction min Dice: {d_repro_min:.9f}")
print(f"FCM implementation audit: {'PASS' if fcm_audit_ok else 'FAILED'}")
print("\nMethod summary:")
print(summary_df.to_string(index=False))
print(f"\nSummary: {SUMMARY_TXT}")
print(f"Metrics: {METRICS_CSV}")
print(f"Paper-ready Table 2: {TABLE2_CSV}")
print(f"Implementation audit: {AUDIT_CSV}")
print(f"Contact sheets: {CONTACT_DIR}")
print("\nFINAL CHECK: PASS" if final_pass else "\nFINAL CHECK: FAILED")
print("=" * 110)

# Fail loudly if either signature of the old FCM implementation reappears.
if fcm_count_ok and not fcm_not_all_one_iteration:
    raise RuntimeError(
        "FCM audit failed: all 877 cases finished within one iteration. "
        "Recheck the FCM update/convergence implementation."
    )
if ab_count_ok and not ab_not_all_identical:
    raise RuntimeError(
        "FCM audit failed: K-means and FCM produced identical masks for all "
        "877 cases. Recheck the FCM pipeline and output paths."
    )
