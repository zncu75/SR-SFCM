"""Inference kernels extracted from 12_method_compare_corrected.py.

Scientific parameters and kernel bodies are preserved. No experiment
or statistical-analysis code is included. Images use OpenCV BGR order.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic

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


