"""Functional mask-generation kernels from original step 08B; no statistics."""
import time
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic

EXPECTED_N = 877


RANDOM_SEED = 42


WORK_MAX_SIDE = 384


ROI_THRESHOLD = 12


SLIC_N_SEGMENTS = 500


SLIC_COMPACTNESS = 7.0


SLIC_SIGMA = 1.0


FCM_CLUSTERS = 2


FCM_M = 2.0


FCM_MAX_ITER = 200


FCM_TOL = 1e-5


SPATIAL_LAMBDA = 0.90


UNCERTAIN_MEMBERSHIP_THRESHOLD = 0.60


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_png_unicode(path, image):
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def largest_component(mask):
    binary = (mask > 0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if n_labels <= 1:
        return np.zeros_like(binary)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    return (labels == largest_label).astype(np.uint8)


def extract_tongue_roi(image):
    mask = (image.max(axis=2) > ROI_THRESHOLD).astype(np.uint8)
    mask = largest_component(mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = largest_component(mask)

    return mask


def resize_to_work(image, roi):
    h, w = image.shape[:2]
    max_side = max(h, w)

    if max_side <= WORK_MAX_SIDE:
        return image.copy(), roi.copy()

    scale = WORK_MAX_SIDE / max_side

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image_work = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    roi_work = cv2.resize(
        roi,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    roi_work = (roi_work > 0).astype(np.uint8)

    return image_work, roi_work


def build_pixel_features(image):
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    ).astype(np.float32)

    L = lab[:, :, 0] * 100.0 / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    S = hsv[:, :, 1] / 255.0

    raw_features = np.stack(
        [L, a, b, S],
        axis=-1,
    )

    scaled_features = np.stack(
        [
            np.clip(L / 100.0, 0, 1),
            np.clip(a / 40.0, -2, 2),
            np.clip(b / 50.0, -2, 2),
            np.clip(S, 0, 1),
        ],
        axis=-1,
    )

    return raw_features, scaled_features


def normalized_radial_map(roi):
    ys, xs = np.where(roi > 0)

    if len(xs) == 0:
        return np.zeros(
            roi.shape,
            dtype=np.float32,
        )

    cx = float(xs.mean())
    cy = float(ys.mean())

    area = float(len(xs))
    equivalent_radius = max(
        np.sqrt(area / np.pi),
        1e-6,
    )

    yy, xx = np.indices(roi.shape)

    radius = np.sqrt(
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
    )

    return (radius / equivalent_radius).astype(np.float32)


def run_slic(image, roi):
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    labels = slic(
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

    return labels


def superpixel_features(
    labels,
    roi,
    raw_features,
    scaled_features,
    radial_map,
):
    valid_labels = np.unique(
        labels[roi > 0]
    )

    valid_labels = valid_labels[
        valid_labels >= 0
    ]

    features = []
    raw_means = []
    radial_means = []
    areas = []
    label_to_index = {}

    for label in valid_labels:
        mask = (
            (labels == label)
            &
            (roi > 0)
        )

        n_pixels = int(mask.sum())

        if n_pixels == 0:
            continue

        current_index = len(features)
        label_to_index[int(label)] = current_index

        features.append(
            scaled_features[mask].mean(axis=0)
        )

        raw_means.append(
            raw_features[mask].mean(axis=0)
        )

        radial_means.append(
            float(radial_map[mask].mean())
        )

        areas.append(n_pixels)

    X = np.asarray(
        features,
        dtype=np.float64,
    )

    raw_means = np.asarray(
        raw_means,
        dtype=np.float64,
    )

    radial_means = np.asarray(
        radial_means,
        dtype=np.float64,
    )

    areas = np.asarray(
        areas,
        dtype=np.float64,
    )

    return (
        X,
        raw_means,
        radial_means,
        areas,
        label_to_index,
    )


def build_adjacency(
    labels,
    roi,
    label_to_index,
):
    n = len(label_to_index)

    adjacency = [
        set()
        for _ in range(n)
    ]

    left = labels[:, :-1]
    right = labels[:, 1:]

    valid_h = (
        (roi[:, :-1] > 0)
        &
        (roi[:, 1:] > 0)
        &
        (left != right)
        &
        (left >= 0)
        &
        (right >= 0)
    )

    if np.any(valid_h):
        pairs_h = np.stack(
            [
                left[valid_h],
                right[valid_h],
            ],
            axis=1,
        )
    else:
        pairs_h = np.empty(
            (0, 2),
            dtype=np.int32,
        )

    upper = labels[:-1, :]
    lower = labels[1:, :]

    valid_v = (
        (roi[:-1, :] > 0)
        &
        (roi[1:, :] > 0)
        &
        (upper != lower)
        &
        (upper >= 0)
        &
        (lower >= 0)
    )

    if np.any(valid_v):
        pairs_v = np.stack(
            [
                upper[valid_v],
                lower[valid_v],
            ],
            axis=1,
        )
    else:
        pairs_v = np.empty(
            (0, 2),
            dtype=np.int32,
        )

    if len(pairs_h) + len(pairs_v) > 0:
        pairs = np.concatenate(
            [pairs_h, pairs_v],
            axis=0,
        )

        pairs = np.sort(
            pairs,
            axis=1,
        )

        pairs = np.unique(
            pairs,
            axis=0,
        )

        for label_a, label_b in pairs:
            label_a = int(label_a)
            label_b = int(label_b)

            if (
                label_a not in label_to_index
                or
                label_b not in label_to_index
            ):
                continue

            index_a = label_to_index[label_a]
            index_b = label_to_index[label_b]

            adjacency[index_a].add(index_b)
            adjacency[index_b].add(index_a)

    return adjacency


def membership_from_centers(
    X,
    centers,
    m,
):
    eps = 1e-12

    distances = np.linalg.norm(
        X[:, None, :]
        -
        centers[None, :, :],
        axis=2,
    )

    distances = np.maximum(
        distances,
        eps,
    )

    exponent = 2.0 / (m - 1.0)

    ratio = (
        distances[:, :, None]
        /
        distances[:, None, :]
    )

    denominator = np.sum(
        ratio ** exponent,
        axis=2,
    )

    return 1.0 / denominator


def spatial_support(
    U_base,
    adjacency,
):
    h = np.zeros_like(U_base)

    for index, neighbors in enumerate(adjacency):
        if len(neighbors) == 0:
            h[index] = U_base[index]
            continue

        neighbor_index = np.fromiter(
            neighbors,
            dtype=np.int64,
        )

        h[index] = (
            U_base[neighbor_index]
            .mean(axis=0)
        )

    return h


def spatial_fcm(
    X,
    adjacency,
):
    if len(X) < 2:
        raise RuntimeError(
            "有效超像素不足2个。"
        )

    kmeans = KMeans(
        n_clusters=FCM_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10,
    )

    kmeans.fit(X)

    centers = (
        kmeans.cluster_centers_
        .astype(np.float64)
    )

    U_prev = membership_from_centers(
        X,
        centers,
        FCM_M,
    )

    converged = False
    final_delta = np.nan

    for iteration in range(
        1,
        FCM_MAX_ITER + 1,
    ):
        U_base = membership_from_centers(
            X,
            centers,
            FCM_M,
        )

        h = spatial_support(
            U_base,
            adjacency,
        )

        h = np.maximum(
            h,
            1e-12,
        )

        fused = (
            U_base
            *
            (
                h
                **
                SPATIAL_LAMBDA
            )
        )

        fused = (
            fused
            /
            np.maximum(
                fused.sum(
                    axis=1,
                    keepdims=True,
                ),
                1e-12,
            )
        )

        weights = fused ** FCM_M

        centers_new = (
            weights.T @ X
        ) / np.maximum(
            weights.sum(axis=0)[:, None],
            1e-12,
        )

        delta = float(
            np.max(
                np.abs(
                    fused - U_prev
                )
            )
        )

        final_delta = delta
        centers = centers_new
        U_prev = fused

        if delta < FCM_TOL:
            converged = True
            break

    return (
        U_prev,
        centers,
        iteration,
        converged,
        final_delta,
    )


def semantic_mapping(
    U,
    raw_means,
    radial_means,
    areas,
):
    hard_labels = np.argmax(
        U,
        axis=1,
    )

    cluster_scores = []
    cluster_stats = []

    for cluster_id in range(
        FCM_CLUSTERS
    ):
        cluster_mask = (
            hard_labels
            ==
            cluster_id
        )

        if not np.any(cluster_mask):
            cluster_scores.append(-np.inf)
            cluster_stats.append(None)
            continue

        cluster_areas = areas[cluster_mask]
        cluster_weights = (
            cluster_areas
            /
            cluster_areas.sum()
        )

        mean_raw = np.sum(
            raw_means[cluster_mask]
            *
            cluster_weights[:, None],
            axis=0,
        )

        mean_radial = float(
            np.sum(
                radial_means[cluster_mask]
                *
                cluster_weights
            )
        )

        L = float(mean_raw[0])
        a = float(mean_raw[1])
        b = float(mean_raw[2])
        S = float(mean_raw[3])

        centrality = (
            1.0
            -
            np.clip(
                mean_radial / 1.2,
                0,
                1,
            )
        )

        score = (
            1.00 * (L / 100.0)
            -
            1.35 * (a / 40.0)
            +
            0.45 * (b / 50.0)
            +
            0.25 * (1.0 - S)
            +
            0.35 * centrality
        )

        cluster_scores.append(
            float(score)
        )

        cluster_stats.append(
            {
                "L": L,
                "a": a,
                "b": b,
                "S": S,
                "radial": mean_radial,
                "score": float(score),
            }
        )

    coating_cluster = int(
        np.argmax(cluster_scores)
    )

    return (
        coating_cluster,
        hard_labels,
        cluster_scores,
        cluster_stats,
    )


def labels_to_coating_mask(
    labels,
    roi,
    label_to_index,
    hard_labels,
    coating_cluster,
):
    coating_mask = np.zeros(
        roi.shape,
        dtype=np.uint8,
    )

    for (
        superpixel_label,
        feature_index,
    ) in label_to_index.items():
        if (
            hard_labels[feature_index]
            ==
            coating_cluster
        ):
            coating_mask[
                labels == superpixel_label
            ] = 1

    coating_mask[roi == 0] = 0

    return coating_mask


def postprocess_mask(
    coating,
    roi,
):
    coating = (coating > 0).astype(np.uint8)
    roi = (roi > 0).astype(np.uint8)

    roi_area = int(roi.sum())

    min_area = max(
        20,
        int(
            round(
                0.001 * roi_area
            )
        ),
    )

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            coating,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        coating
    )

    for label_id in range(
        1,
        n_labels,
    ):
        component_area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if component_area >= min_area:
            cleaned[
                labels == label_id
            ] = 1

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
    )

    cleaned[roi == 0] = 0

    return cleaned


def segment_candidate_regions(
    image,
    roi_input,
    output_shape,
):
    branch_start = time.time()

    image_work, roi_work = (
        resize_to_work(
            image,
            roi_input,
        )
    )

    if roi_work.sum() == 0:
        raise RuntimeError(
            "工作分辨率ROI为空。"
        )

    raw_features, scaled_features = (
        build_pixel_features(
            image_work
        )
    )

    radial_map = (
        normalized_radial_map(
            roi_work
        )
    )

    labels = run_slic(
        image_work,
        roi_work,
    )

    (
        X,
        raw_means,
        radial_means,
        areas,
        label_to_index,
    ) = superpixel_features(
        labels,
        roi_work,
        raw_features,
        scaled_features,
        radial_map,
    )

    if len(X) < 2:
        raise RuntimeError(
            f"SLIC有效超像素不足2个：{len(X)}"
        )

    adjacency = build_adjacency(
        labels,
        roi_work,
        label_to_index,
    )

    (
        U,
        centers,
        iterations,
        converged,
        final_delta,
    ) = spatial_fcm(
        X,
        adjacency,
    )

    (
        coating_cluster,
        hard_labels,
        cluster_scores,
        cluster_stats,
    ) = semantic_mapping(
        U,
        raw_means,
        radial_means,
        areas,
    )

    coating_work = (
        labels_to_coating_mask(
            labels,
            roi_work,
            label_to_index,
            hard_labels,
            coating_cluster,
        )
    )

    output_h, output_w = output_shape

    coating_output = cv2.resize(
        coating_work,
        (output_w, output_h),
        interpolation=cv2.INTER_NEAREST,
    )

    roi_output = cv2.resize(
        roi_input,
        (output_w, output_h),
        interpolation=cv2.INTER_NEAREST,
    )

    roi_output = (
        roi_output > 0
    ).astype(np.uint8)

    coating_output = (
        postprocess_mask(
            coating_output,
            roi_output,
        )
    )

    body_output = (
        (
            roi_output > 0
        )
        &
        (
            coating_output == 0
        )
    ).astype(np.uint8)

    max_membership = np.max(
        U,
        axis=1,
    )

    uncertain_mask = (
        max_membership
        <
        UNCERTAIN_MEMBERSHIP_THRESHOLD
    )

    if areas.sum() > 0:
        uncertain_fraction = float(
            areas[uncertain_mask].sum()
            /
            areas.sum()
        )
    else:
        uncertain_fraction = np.nan

    runtime_seconds = (
        time.time()
        -
        branch_start
    )

    return {
        "coating": coating_output,
        "body": body_output,
        "iterations": int(iterations),
        "converged": bool(converged),
        "final_delta": float(final_delta),
        "n_superpixels": int(len(X)),
        "uncertain_fraction": float(
            uncertain_fraction
        ),
        "coating_cluster": int(
            coating_cluster
        ),
        "cluster_scores": cluster_scores,
        "cluster_stats": cluster_stats,
        "work_h": int(
            image_work.shape[0]
        ),
        "work_w": int(
            image_work.shape[1]
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
    }


