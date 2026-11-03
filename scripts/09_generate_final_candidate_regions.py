from pathlib import Path
import os
import json
import time
import warnings
from datetime import datetime

# ============================================================
# Windows / joblib warning处理
# 必须放在 sklearn 导入之前
# ============================================================

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# SLIC内部偶发空cluster后会自动重新初始化。
# 该warning不代表病例失败；正式结果仍由超像素数量、
# Spatial FCM收敛状态等进行检查。
warnings.filterwarnings(
    "ignore",
    message="One of the clusters is empty.*",
    category=UserWarning
)

import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from skimage.segmentation import slic


# ============================================================
# 0. 项目路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_images"
)

# ------------------------------------------------------------
# 正式候选区域输出
# ------------------------------------------------------------

FINAL_REGION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "final_candidate_regions"
)

ROI_DIR = (
    FINAL_REGION_ROOT
    / "tongue_roi"
)

COATING_MASK_DIR = (
    FINAL_REGION_ROOT
    / "candidate_coating_masks"
)

BODY_MASK_DIR = (
    FINAL_REGION_ROOT
    / "candidate_body_masks"
)

COATING_IMAGE_DIR = (
    FINAL_REGION_ROOT
    / "candidate_coating_images"
)

BODY_IMAGE_DIR = (
    FINAL_REGION_ROOT
    / "candidate_body_images"
)

OVERLAY_DIR = (
    FINAL_REGION_ROOT
    / "overlays"
)

# ------------------------------------------------------------
# 正式实验结果
# ------------------------------------------------------------

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "09_final_candidate_regions"
)

CONTACT_DIR = (
    RESULT_DIR
    / "contact_sheets"
)

for folder in [
    FINAL_REGION_ROOT,
    ROI_DIR,
    COATING_MASK_DIR,
    BODY_MASK_DIR,
    COATING_IMAGE_DIR,
    BODY_IMAGE_DIR,
    OVERLAY_DIR,
    RESULT_DIR,
    CONTACT_DIR,
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


OUTPUT_CSV = (
    RESULT_DIR
    / "candidate_region_metrics.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "candidate_region_summary.txt"
)

CONFIG_PATH = (
    RESULT_DIR
    / "candidate_region_config.json"
)

FAILED_CSV = (
    RESULT_DIR
    / "failed_cases.csv"
)

NONCONVERGED_CSV = (
    RESULT_DIR
    / "nonconverged_cases.csv"
)


# ============================================================
# 1. 冻结参数
# ============================================================

EXPECTED_N = 877

RANDOM_SEED = 42


# ------------------------------------------------------------
# 工作分辨率
# ------------------------------------------------------------

WORK_MAX_SIDE = 384


# ------------------------------------------------------------
# ROI
# ------------------------------------------------------------

ROI_THRESHOLD = 12


# ------------------------------------------------------------
# SLIC
# ------------------------------------------------------------

SLIC_N_SEGMENTS = 500

SLIC_COMPACTNESS = 7.0

SLIC_SIGMA = 1.0


# ------------------------------------------------------------
# Spatial FCM
# ------------------------------------------------------------

FCM_CLUSTERS = 2

FCM_M = 2.0

FCM_MAX_ITER = 200

FCM_TOL = 1e-5

SPATIAL_LAMBDA = 0.90


# ------------------------------------------------------------
# Membership不确定性
# ------------------------------------------------------------

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


# ============================================================
# 2. 中文路径安全读取
# ============================================================

def imread_unicode(path):

    try:

        data = np.fromfile(
            str(path),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None


# ============================================================
# 3. 中文路径安全保存PNG
# ============================================================

def imwrite_png_unicode(
    path,
    image
):

    ok, encoded = cv2.imencode(
        ".png",
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3
        ]
    )

    if not ok:

        return False

    encoded.tofile(
        str(path)
    )

    return True


# ============================================================
# 4. 最大8连通区域
# ============================================================

def largest_component(mask):

    binary = (
        mask.astype(np.uint8)
        >
        0
    ).astype(
        np.uint8
    )

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if n_labels <= 1:

        return np.zeros_like(
            binary
        )

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_label = (
        1
        +
        int(
            np.argmax(
                areas
            )
        )
    )

    return (
        labels
        ==
        largest_label
    ).astype(
        np.uint8
    )


# ============================================================
# 5. 舌体ROI
#
# 固定正式方法：
#
# max RGB > 12
# 最大8连通域
# 5×5 closing
# 再取最大连通域
# ============================================================

def extract_tongue_roi(image):

    mask = (
        image.max(
            axis=2
        )
        >
        ROI_THRESHOLD
    ).astype(
        np.uint8
    )

    mask = largest_component(
        mask
    )

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = largest_component(
        mask
    )

    return mask


# ============================================================
# 6. Resize到工作尺寸
# ============================================================

def resize_to_work(
    image,
    roi
):

    h, w = image.shape[:2]

    max_side = max(
        h,
        w
    )

    if max_side <= WORK_MAX_SIDE:

        return (
            image.copy(),
            roi.copy()
        )

    scale = (
        WORK_MAX_SIDE
        /
        max_side
    )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        )
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        )
    )

    image_work = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    roi_work = cv2.resize(
        roi,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    roi_work = (
        roi_work > 0
    ).astype(
        np.uint8
    )

    return (
        image_work,
        roi_work
    )


# ============================================================
# 7. 构建Lab + HSV特征
#
# raw:
#
# [L*, a*, b*, S]
#
# scaled:
#
# L*/100
# a*/40
# b*/50
# S
# ============================================================

def build_pixel_features(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(
        np.float32
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(
        np.float32
    )

    L = (
        lab[:, :, 0]
        *
        100.0
        /
        255.0
    )

    a = (
        lab[:, :, 1]
        -
        128.0
    )

    b = (
        lab[:, :, 2]
        -
        128.0
    )

    S = (
        hsv[:, :, 1]
        /
        255.0
    )

    raw_features = np.stack(
        [
            L,
            a,
            b,
            S
        ],
        axis=-1
    )

    scaled_features = np.stack(
        [

            np.clip(
                L / 100.0,
                0,
                1
            ),

            np.clip(
                a / 40.0,
                -2,
                2
            ),

            np.clip(
                b / 50.0,
                -2,
                2
            ),

            np.clip(
                S,
                0,
                1
            ),

        ],
        axis=-1
    )

    return (
        raw_features,
        scaled_features
    )


# ============================================================
# 8. ROI径向距离
# ============================================================

def normalized_radial_map(roi):

    ys, xs = np.where(
        roi > 0
    )

    if len(xs) == 0:

        return np.zeros(
            roi.shape,
            dtype=np.float32
        )

    cx = float(
        xs.mean()
    )

    cy = float(
        ys.mean()
    )

    area = float(
        len(xs)
    )

    equivalent_radius = np.sqrt(
        area
        /
        np.pi
    )

    equivalent_radius = max(
        equivalent_radius,
        1e-6
    )

    yy, xx = np.indices(
        roi.shape
    )

    radius = np.sqrt(
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
    )

    radius = (
        radius
        /
        equivalent_radius
    )

    return radius.astype(
        np.float32
    )


# ============================================================
# 9. SLIC
# ============================================================

def run_slic(
    image,
    roi
):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb = (
        rgb.astype(
            np.float32
        )
        /
        255.0
    )

    labels = slic(

        rgb,

        n_segments=
            SLIC_N_SEGMENTS,

        compactness=
            SLIC_COMPACTNESS,

        sigma=
            SLIC_SIGMA,

        start_label=0,

        mask=
            roi.astype(
                bool
            ),

        enforce_connectivity=True,

        channel_axis=-1,

        convert2lab=True
    )

    return labels


# ============================================================
# 10. 超像素特征
# ============================================================

def superpixel_features(
    labels,
    roi,
    raw_features,
    scaled_features,
    radial_map
):

    valid_labels = np.unique(
        labels[
            roi > 0
        ]
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

        n_pixels = int(
            mask.sum()
        )

        if n_pixels == 0:

            continue

        feature_index = len(
            features
        )

        label_to_index[
            int(label)
        ] = feature_index

        features.append(
            scaled_features[
                mask
            ].mean(
                axis=0
            )
        )

        raw_means.append(
            raw_features[
                mask
            ].mean(
                axis=0
            )
        )

        radial_means.append(
            float(
                radial_map[
                    mask
                ].mean()
            )
        )

        areas.append(
            n_pixels
        )

    X = np.asarray(
        features,
        dtype=np.float64
    )

    raw_means = np.asarray(
        raw_means,
        dtype=np.float64
    )

    radial_means = np.asarray(
        radial_means,
        dtype=np.float64
    )

    areas = np.asarray(
        areas,
        dtype=np.float64
    )

    return (
        X,
        raw_means,
        radial_means,
        areas,
        label_to_index
    )


# ============================================================
# 11. 4-neighbor超像素邻接
# ============================================================

def build_adjacency(
    labels,
    roi,
    label_to_index
):

    n = len(
        label_to_index
    )

    adjacency = [
        set()
        for _ in range(n)
    ]

    # --------------------------------------------------------
    # 水平相邻
    # --------------------------------------------------------

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

    if np.any(
        valid_h
    ):

        pairs_h = np.stack(
            [
                left[
                    valid_h
                ],
                right[
                    valid_h
                ]
            ],
            axis=1
        )

    else:

        pairs_h = np.empty(
            (0, 2),
            dtype=np.int32
        )

    # --------------------------------------------------------
    # 垂直相邻
    # --------------------------------------------------------

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

    if np.any(
        valid_v
    ):

        pairs_v = np.stack(
            [
                upper[
                    valid_v
                ],
                lower[
                    valid_v
                ]
            ],
            axis=1
        )

    else:

        pairs_v = np.empty(
            (0, 2),
            dtype=np.int32
        )

    if (
        len(pairs_h)
        +
        len(pairs_v)
        >
        0
    ):

        pairs = np.concatenate(
            [
                pairs_h,
                pairs_v
            ],
            axis=0
        )

        pairs = np.sort(
            pairs,
            axis=1
        )

        pairs = np.unique(
            pairs,
            axis=0
        )

        for label_a, label_b in pairs:

            label_a = int(
                label_a
            )

            label_b = int(
                label_b
            )

            if (
                label_a
                not in label_to_index
                or
                label_b
                not in label_to_index
            ):

                continue

            index_a = (
                label_to_index[
                    label_a
                ]
            )

            index_b = (
                label_to_index[
                    label_b
                ]
            )

            adjacency[
                index_a
            ].add(
                index_b
            )

            adjacency[
                index_b
            ].add(
                index_a
            )

    return adjacency


# ============================================================
# 12. 标准FCM membership
# ============================================================

def membership_from_centers(
    X,
    centers,
    m
):

    eps = 1e-12

    distances = np.linalg.norm(

        X[:, None, :]
        -
        centers[
            None,
            :,
            :
        ],

        axis=2
    )

    distances = np.maximum(
        distances,
        eps
    )

    exponent = (
        2.0
        /
        (m - 1.0)
    )

    ratio = (
        distances[:, :, None]
        /
        distances[:, None, :]
    )

    denominator = np.sum(
        ratio
        **
        exponent,
        axis=2
    )

    U = (
        1.0
        /
        denominator
    )

    return U


# ============================================================
# 13. 空间支持项
# ============================================================

def spatial_support(
    U_base,
    adjacency
):

    h = np.zeros_like(
        U_base
    )

    for index, neighbors in enumerate(
        adjacency
    ):

        if len(
            neighbors
        ) == 0:

            h[index] = (
                U_base[
                    index
                ]
            )

            continue

        neighbor_index = np.fromiter(
            neighbors,
            dtype=np.int64
        )

        h[index] = (
            U_base[
                neighbor_index
            ]
            .mean(
                axis=0
            )
        )

    return h


# ============================================================
# 14. Spatial FCM
# ============================================================

def spatial_fcm(
    X,
    adjacency
):

    if len(X) < 2:

        raise RuntimeError(
            "有效超像素不足2个。"
        )

    # --------------------------------------------------------
    # KMeans初始化
    # --------------------------------------------------------

    kmeans = KMeans(

        n_clusters=
            FCM_CLUSTERS,

        random_state=
            RANDOM_SEED,

        n_init=10
    )

    kmeans.fit(
        X
    )

    centers = (
        kmeans.cluster_centers_
        .astype(
            np.float64
        )
    )

    U_prev = (
        membership_from_centers(
            X,
            centers,
            FCM_M
        )
    )

    converged = False

    final_delta = np.nan

    # --------------------------------------------------------
    # Spatial FCM迭代
    # --------------------------------------------------------

    for iteration in range(
        1,
        FCM_MAX_ITER + 1
    ):

        # 标准FCM membership
        U_base = (
            membership_from_centers(
                X,
                centers,
                FCM_M
            )
        )

        # 空间邻域支持
        h = spatial_support(
            U_base,
            adjacency
        )

        h = np.maximum(
            h,
            1e-12
        )

        # 空间融合
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
                    keepdims=True
                ),
                1e-12
            )
        )

        # 聚类中心更新
        weights = (
            fused
            **
            FCM_M
        )

        centers_new = (
            weights.T
            @
            X
        )

        centers_new = (
            centers_new
            /
            np.maximum(
                weights.sum(
                    axis=0
                )[:, None],
                1e-12
            )
        )

        # 收敛误差
        delta = float(
            np.max(
                np.abs(
                    fused
                    -
                    U_prev
                )
            )
        )

        final_delta = (
            delta
        )

        centers = (
            centers_new
        )

        U_prev = (
            fused
        )

        if delta < FCM_TOL:

            converged = True

            break

    return (
        U_prev,
        centers,
        iteration,
        converged,
        final_delta
    )


# ============================================================
# 15. 候选舌苔语义映射
#
# score =
#
# +1.00 * L*/100
# -1.35 * a*/40
# +0.45 * b*/50
# +0.25 * (1-S)
# +0.35 * centrality
#
# 分数高者定义为候选舌苔区域
# ============================================================

def semantic_mapping(
    U,
    raw_means,
    radial_means,
    areas
):

    hard_labels = np.argmax(
        U,
        axis=1
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

        if not np.any(
            cluster_mask
        ):

            cluster_scores.append(
                -np.inf
            )

            cluster_stats.append(
                None
            )

            continue

        cluster_areas = (
            areas[
                cluster_mask
            ]
        )

        cluster_weights = (
            cluster_areas
            /
            cluster_areas.sum()
        )

        mean_raw = np.sum(

            raw_means[
                cluster_mask
            ]
            *
            cluster_weights[
                :,
                None
            ],

            axis=0
        )

        mean_radial = float(
            np.sum(
                radial_means[
                    cluster_mask
                ]
                *
                cluster_weights
            )
        )

        L = float(
            mean_raw[0]
        )

        a = float(
            mean_raw[1]
        )

        b = float(
            mean_raw[2]
        )

        S = float(
            mean_raw[3]
        )

        centrality = (
            1.0
            -
            np.clip(
                mean_radial
                /
                1.2,
                0,
                1
            )
        )

        score = (

            1.00
            *
            (
                L / 100.0
            )

            -

            1.35
            *
            (
                a / 40.0
            )

            +

            0.45
            *
            (
                b / 50.0
            )

            +

            0.25
            *
            (
                1.0 - S
            )

            +

            0.35
            *
            centrality
        )

        cluster_scores.append(
            float(
                score
            )
        )

        cluster_stats.append(
            {

                "L":
                    L,

                "a":
                    a,

                "b":
                    b,

                "S":
                    S,

                "radial":
                    mean_radial,

                "score":
                    float(
                        score
                    ),
            }
        )

    coating_cluster = int(
        np.argmax(
            cluster_scores
        )
    )

    return (
        coating_cluster,
        hard_labels,
        cluster_scores,
        cluster_stats
    )


# ============================================================
# 16. 超像素类别 -> 候选舌苔mask
# ============================================================

def labels_to_coating_mask(
    labels,
    roi,
    label_to_index,
    hard_labels,
    coating_cluster
):

    coating_mask = np.zeros(
        roi.shape,
        dtype=np.uint8
    )

    for (
        superpixel_label,
        feature_index
    ) in label_to_index.items():

        if (
            hard_labels[
                feature_index
            ]
            ==
            coating_cluster
        ):

            coating_mask[
                labels
                ==
                superpixel_label
            ] = 1

    coating_mask[
        roi == 0
    ] = 0

    return coating_mask


# ============================================================
# 17. mask后处理
# ============================================================

def postprocess_mask(
    coating,
    roi
):

    coating = (
        coating > 0
    ).astype(
        np.uint8
    )

    roi = (
        roi > 0
    ).astype(
        np.uint8
    )

    roi_area = int(
        roi.sum()
    )

    min_area = max(

        20,

        int(
            round(
                0.001
                *
                roi_area
            )
        )
    )

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            coating,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        coating
    )

    for label_id in range(
        1,
        n_labels
    ):

        component_area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )

        if component_area >= min_area:

            cleaned[
                labels
                ==
                label_id
            ] = 1

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    cleaned[
        roi == 0
    ] = 0

    return cleaned


# ============================================================
# 18. 单病例正式候选区域生成
# ============================================================

def segment_candidate_regions(
    image,
    roi
):

    original_h, original_w = (
        image.shape[:2]
    )

    image_work, roi_work = (
        resize_to_work(
            image,
            roi
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

    labels = (
        run_slic(
            image_work,
            roi_work
        )
    )

    (
        X,
        raw_means,
        radial_means,
        areas,
        label_to_index

    ) = superpixel_features(

        labels,
        roi_work,
        raw_features,
        scaled_features,
        radial_map
    )

    if len(X) < 2:

        raise RuntimeError(
            f"有效超像素不足2个：{len(X)}"
        )

    adjacency = (
        build_adjacency(
            labels,
            roi_work,
            label_to_index
        )
    )

    (
        U,
        centers,
        iterations,
        converged,
        final_delta

    ) = spatial_fcm(
        X,
        adjacency
    )

    (
        coating_cluster,
        hard_labels,
        cluster_scores,
        cluster_stats

    ) = semantic_mapping(

        U,
        raw_means,
        radial_means,
        areas
    )

    coating_work = (
        labels_to_coating_mask(

            labels,
            roi_work,
            label_to_index,
            hard_labels,
            coating_cluster
        )
    )

    coating = cv2.resize(

        coating_work,

        (
            original_w,
            original_h
        ),

        interpolation=
            cv2.INTER_NEAREST
    )

    coating = (
        postprocess_mask(
            coating,
            roi
        )
    )

    body = (

        (
            roi > 0
        )

        &

        (
            coating == 0
        )

    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Membership不确定区域比例
    # --------------------------------------------------------

    max_membership = np.max(
        U,
        axis=1
    )

    uncertain = (
        max_membership
        <
        UNCERTAIN_MEMBERSHIP_THRESHOLD
    )

    uncertain_fraction = float(

        areas[
            uncertain
        ].sum()

        /

        max(
            areas.sum(),
            1.0
        )
    )

    return {

        "coating":
            coating,

        "body":
            body,

        "iterations":
            int(
                iterations
            ),

        "converged":
            bool(
                converged
            ),

        "final_delta":
            float(
                final_delta
            ),

        "n_superpixels":
            int(
                len(X)
            ),

        "uncertain_fraction":
            float(
                uncertain_fraction
            ),

        "coating_cluster":
            int(
                coating_cluster
            ),

        "cluster_scores":
            cluster_scores,

        "cluster_stats":
            cluster_stats,
    }


# ============================================================
# 19. 从mask生成候选区域图像
# ============================================================

def masked_image(
    image,
    mask
):

    output = np.zeros_like(
        image
    )

    output[
        mask > 0
    ] = image[
        mask > 0
    ]

    return output


# ============================================================
# 20. Overlay
#
# 绿色：候选舌苔边界
# 蓝色：候选舌质外边界
# ============================================================

def create_overlay(
    image,
    coating,
    body
):

    output = (
        image.copy()
    )

    coating_contours, _ = (
        cv2.findContours(

            (
                coating > 0
            ).astype(
                np.uint8
            ),

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )
    )

    body_contours, _ = (
        cv2.findContours(

            (
                body > 0
            ).astype(
                np.uint8
            ),

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )
    )

    cv2.drawContours(
        output,
        body_contours,
        -1,
        (255, 0, 0),
        2
    )

    cv2.drawContours(
        output,
        coating_contours,
        -1,
        (0, 255, 0),
        2
    )

    return output


# ============================================================
# 21. Contact sheet辅助
# ============================================================

def fit_image(
    image,
    width=260,
    height=220
):

    h, w = image.shape[:2]

    scale = min(
        width / w,
        height / h
    )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        )
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        )
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    x = (
        width - new_w
    ) // 2

    y = (
        height - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# ============================================================
# 22. Contact sheet
#
# 每例：
#
# Original
# Candidate body
# Candidate coating
# Overlay
# ============================================================

def build_contact_sheet(
    sample_df,
    image_lookup,
    output_path
):

    if len(
        sample_df
    ) == 0:

        return

    THUMB_W = 260

    THUMB_H = 220

    TEXT_H = 70

    CELL_W = (
        THUMB_W * 4
    )

    CELL_H = (
        THUMB_H
        +
        TEXT_H
    )

    n_rows = len(
        sample_df
    )

    sheet = np.full(

        (
            n_rows * CELL_H,
            CELL_W,
            3
        ),

        255,

        dtype=np.uint8
    )

    for row_index, (_, row) in enumerate(
        sample_df.iterrows()
    ):

        image_id = str(
            row[
                "image_id"
            ]
        )

        original = imread_unicode(
            image_lookup[
                image_id
            ]
        )

        coating_img = imread_unicode(
            COATING_IMAGE_DIR
            /
            f"{image_id}.png"
        )

        body_img = imread_unicode(
            BODY_IMAGE_DIR
            /
            f"{image_id}.png"
        )

        overlay = imread_unicode(
            OVERLAY_DIR
            /
            f"{image_id}.png"
        )

        if any(
            x is None
            for x in [
                original,
                body_img,
                coating_img,
                overlay
            ]
        ):

            continue

        thumbs = [

            fit_image(
                original,
                THUMB_W,
                THUMB_H
            ),

            fit_image(
                body_img,
                THUMB_W,
                THUMB_H
            ),

            fit_image(
                coating_img,
                THUMB_W,
                THUMB_H
            ),

            fit_image(
                overlay,
                THUMB_W,
                THUMB_H
            ),
        ]

        y0 = (
            row_index
            *
            CELL_H
        )

        for col_index, thumb in enumerate(
            thumbs
        ):

            x0 = (
                col_index
                *
                THUMB_W
            )

            sheet[
                y0:
                y0 + THUMB_H,

                x0:
                x0 + THUMB_W
            ] = thumb

        cv2.putText(
            sheet,
            "Original",
            (
                5,
                y0 + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            "Candidate body",
            (
                THUMB_W + 5,
                y0 + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            "Candidate coating",
            (
                2 * THUMB_W + 5,
                y0 + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            "Overlay",
            (
                3 * THUMB_W + 5,
                y0 + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        text_y = (
            y0
            +
            THUMB_H
            +
            22
        )

        filename_show = image_id

        if len(
            filename_show
        ) > 65:

            filename_show = (
                filename_show[:62]
                +
                "..."
            )

        cv2.putText(
            sheet,
            filename_show,
            (
                5,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        line2 = (
            f"Coating={row['coating_ratio']:.3f}  "
            f"Body={row['body_ratio']:.3f}  "
            f"Iter={int(row['iterations'])}  "
            f"Delta={row['final_delta']:.2e}"
        )

        cv2.putText(
            sheet,
            line2,
            (
                5,
                text_y + 24
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if ok:

        encoded.tofile(
            str(
                output_path
            )
        )


# ============================================================
# 23. 扫描正式877幅原始图像
# ============================================================

image_paths = sorted(
    [

        path

        for path
        in INPUT_DIR.iterdir()

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]
)


print(
    "=" * 100
)

print(
    "Step 05 - Final Candidate Tongue-Body / Tongue-Coating Regions"
)

print(
    "=" * 100
)

print(
    f"正式输入："
    f"{len(image_paths)}"
)

print(
    "Input：Original images only"
)

print(
    "SR：NOT USED"
)

print(
    f"Spatial FCM max_iter："
    f"{FCM_MAX_ITER}"
)

print(
    f"Spatial FCM tol："
    f"{FCM_TOL}"
)


# ============================================================
# 24. 数量检查
# ============================================================

if len(
    image_paths
) != EXPECTED_N:

    raise RuntimeError(

        f"正式原始图像数量异常。"
        f"\n预期：{EXPECTED_N}"
        f"\n实际：{len(image_paths)}"
    )


# ============================================================
# 25. 检查stem重复
# ============================================================

stems = [
    path.stem
    for path in image_paths
]

stem_series = pd.Series(
    stems
)

duplicate_stems = (
    stem_series[
        stem_series.duplicated(
            keep=False
        )
    ]
    .unique()
    .tolist()
)

if duplicate_stems:

    raise RuntimeError(
        "发现重复stem：\n"
        +
        "\n".join(
            duplicate_stems
        )
    )


image_lookup = {
    path.stem:
        path

    for path
    in image_paths
}


# ============================================================
# 26. 清理旧正式输出
#
# 只清理本步骤产生的数据，
# 不修改final_images。
# ============================================================

for folder in [
    ROI_DIR,
    COATING_MASK_DIR,
    BODY_MASK_DIR,
    COATING_IMAGE_DIR,
    BODY_IMAGE_DIR,
    OVERLAY_DIR,
]:

    for old_file in folder.iterdir():

        if old_file.is_file():

            old_file.unlink()


# ============================================================
# 27. 保存配置
# ============================================================

config = {

    "created_at":
        datetime.now().isoformat(),

    "formal_dataset_n":
        EXPECTED_N,

    "input":
        "Original images only",

    "super_resolution_used":
        False,

    "work_max_side":
        WORK_MAX_SIDE,

    "roi_method":
        "max RGB > 12 + largest 8-connected component + 5x5 closing",

    "roi_threshold":
        ROI_THRESHOLD,

    "slic_n_segments":
        SLIC_N_SEGMENTS,

    "slic_compactness":
        SLIC_COMPACTNESS,

    "slic_sigma":
        SLIC_SIGMA,

    "fcm_clusters":
        FCM_CLUSTERS,

    "fcm_m":
        FCM_M,

    "fcm_max_iter":
        FCM_MAX_ITER,

    "fcm_tolerance":
        FCM_TOL,

    "spatial_lambda":
        SPATIAL_LAMBDA,

    "random_seed":
        RANDOM_SEED,

    "pixel_features":
        [
            "CIELAB L*",
            "CIELAB a*",
            "CIELAB b*",
            "HSV saturation"
        ],

    "feature_scaling":
        {
            "L":
                "L*/100",

            "a":
                "clip(a*/40,-2,2)",

            "b":
                "clip(b*/50,-2,2)",

            "S":
                "S"
        },

    "candidate_coating_semantic_score":
        (
            "1.00*(L/100)"
            " - 1.35*(a/40)"
            " + 0.45*(b/50)"
            " + 0.25*(1-S)"
            " + 0.35*(1-clip(r/1.2,0,1))"
        ),

    "important":
        (
            "Outputs are candidate tongue-body and "
            "candidate tongue-coating regions, not "
            "expert pixel-level ground-truth segmentation."
        )
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 28. 正式运行
# ============================================================

records = []

overall_start = (
    time.time()
)


for index, image_path in enumerate(
    image_paths,
    start=1
):

    case_start = (
        time.time()
    )

    image_id = (
        image_path.stem
    )

    record = {

        "index":
            index,

        "image_id":
            image_id,

        "filename":
            image_path.name,

        "input_path":
            str(
                image_path
            ),

        "success":
            False,

        "error":
            "",
    }

    try:

        # ====================================================
        # 1. 读取
        # ====================================================

        image = (
            imread_unicode(
                image_path
            )
        )

        if image is None:

            raise RuntimeError(
                "图像读取失败"
            )

        image_h, image_w = (
            image.shape[:2]
        )

        # ====================================================
        # 2. ROI
        # ====================================================

        roi = (
            extract_tongue_roi(
                image
            )
        )

        roi_area = int(
            roi.sum()
        )

        if roi_area <= 0:

            raise RuntimeError(
                "舌体ROI为空"
            )

        # ====================================================
        # 3. SLIC + Spatial FCM
        # ====================================================

        result = (
            segment_candidate_regions(
                image,
                roi
            )
        )

        coating = (
            result[
                "coating"
            ]
        )

        body = (
            result[
                "body"
            ]
        )

        # ====================================================
        # 4. 基本一致性检查
        #
        # coating + body 必须恰好构成ROI
        # ====================================================

        overlap = int(
            np.logical_and(
                coating > 0,
                body > 0
            ).sum()
        )

        union_mask = (
            (
                coating > 0
            )
            |
            (
                body > 0
            )
        ).astype(
            np.uint8
        )

        union_diff = int(
            np.logical_xor(
                union_mask > 0,
                roi > 0
            ).sum()
        )

        if overlap != 0:

            raise RuntimeError(
                f"Body / coating存在重叠：{overlap}"
            )

        if union_diff != 0:

            raise RuntimeError(
                f"Body + coating与ROI不一致：{union_diff}"
            )

        # ====================================================
        # 5. 面积
        # ====================================================

        coating_area = int(
            coating.sum()
        )

        body_area = int(
            body.sum()
        )

        coating_ratio = (
            coating_area
            /
            roi_area
        )

        body_ratio = (
            body_area
            /
            roi_area
        )

        # ====================================================
        # 6. 生成候选区域图像
        # ====================================================

        coating_image = (
            masked_image(
                image,
                coating
            )
        )

        body_image = (
            masked_image(
                image,
                body
            )
        )

        overlay = (
            create_overlay(
                image,
                coating,
                body
            )
        )

        # ====================================================
        # 7. 保存
        # ====================================================

        save_items = [

            (
                ROI_DIR
                /
                f"{image_id}.png",

                roi
                *
                255
            ),

            (
                COATING_MASK_DIR
                /
                f"{image_id}.png",

                coating
                *
                255
            ),

            (
                BODY_MASK_DIR
                /
                f"{image_id}.png",

                body
                *
                255
            ),

            (
                COATING_IMAGE_DIR
                /
                f"{image_id}.png",

                coating_image
            ),

            (
                BODY_IMAGE_DIR
                /
                f"{image_id}.png",

                body_image
            ),

            (
                OVERLAY_DIR
                /
                f"{image_id}.png",

                overlay
            ),
        ]

        for save_path, save_image in save_items:

            ok = imwrite_png_unicode(
                save_path,
                save_image
            )

            if not ok:

                raise RuntimeError(
                    f"保存失败：{save_path}"
                )

        # ====================================================
        # 8. 保存指标
        # ====================================================

        cluster_scores = (
            result[
                "cluster_scores"
            ]
        )

        cluster_stats = (
            result[
                "cluster_stats"
            ]
        )

        record.update(
            {

                "success":
                    True,

                "image_width":
                    image_w,

                "image_height":
                    image_h,

                "roi_area":
                    roi_area,

                "coating_area":
                    coating_area,

                "body_area":
                    body_area,

                "coating_ratio":
                    coating_ratio,

                "body_ratio":
                    body_ratio,

                "body_coating_overlap":
                    overlap,

                "roi_union_difference":
                    union_diff,

                "n_superpixels":
                    result[
                        "n_superpixels"
                    ],

                "iterations":
                    result[
                        "iterations"
                    ],

                "converged":
                    result[
                        "converged"
                    ],

                "final_delta":
                    result[
                        "final_delta"
                    ],

                "uncertain_fraction":
                    result[
                        "uncertain_fraction"
                    ],

                "coating_cluster":
                    result[
                        "coating_cluster"
                    ],

                "cluster0_score":
                    (
                        cluster_scores[0]
                        if
                        len(cluster_scores) > 0
                        else
                        np.nan
                    ),

                "cluster1_score":
                    (
                        cluster_scores[1]
                        if
                        len(cluster_scores) > 1
                        else
                        np.nan
                    ),

                "cluster0_stats_json":
                    json.dumps(
                        cluster_stats[0],
                        ensure_ascii=False
                    )
                    if
                    len(cluster_stats) > 0
                    else
                    "",

                "cluster1_stats_json":
                    json.dumps(
                        cluster_stats[1],
                        ensure_ascii=False
                    )
                    if
                    len(cluster_stats) > 1
                    else
                    "",

                "roi_path":
                    str(
                        ROI_DIR
                        /
                        f"{image_id}.png"
                    ),

                "coating_mask_path":
                    str(
                        COATING_MASK_DIR
                        /
                        f"{image_id}.png"
                    ),

                "body_mask_path":
                    str(
                        BODY_MASK_DIR
                        /
                        f"{image_id}.png"
                    ),

                "coating_image_path":
                    str(
                        COATING_IMAGE_DIR
                        /
                        f"{image_id}.png"
                    ),

                "body_image_path":
                    str(
                        BODY_IMAGE_DIR
                        /
                        f"{image_id}.png"
                    ),
            }
        )

    except Exception as error:

        record[
            "error"
        ] = str(
            error
        )

    record[
        "runtime_seconds"
    ] = (
        time.time()
        -
        case_start
    )

    records.append(
        record
    )

    # --------------------------------------------------------
    # 实时保存
    # --------------------------------------------------------

    pd.DataFrame(
        records
    ).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    if record[
        "success"
    ]:

        print(

            f"[{index:03d}/{EXPECTED_N}] "
            f"{image_id} "
            f"OK "
            f"{record['runtime_seconds']:.2f}s "
            f"| "
            f"Iter={record['iterations']} "
            f"| "
            f"Conv={record['converged']} "
            f"| "
            f"Coating={record['coating_ratio']:.3f}"
        )

    else:

        print(

            f"[{index:03d}/{EXPECTED_N}] "
            f"{image_id} "
            f"FAILED "
            f"| "
            f"{record['error']}"
        )


# ============================================================
# 29. 汇总
# ============================================================

df = pd.DataFrame(
    records
)

success_df = df[
    df[
        "success"
    ]
    ==
    True
].copy()

failed_df = df[
    df[
        "success"
    ]
    !=
    True
].copy()

success_n = len(
    success_df
)

failed_n = len(
    failed_df
)


# ============================================================
# 30. 收敛检查
# ============================================================

if success_n > 0:

    nonconverged_df = (
        success_df[
            success_df[
                "converged"
            ]
            !=
            True
        ]
        .copy()
    )

else:

    nonconverged_df = (
        pd.DataFrame()
    )


nonconverged_n = len(
    nonconverged_df
)


# ============================================================
# 31. 保存失败/未收敛病例
# ============================================================

failed_df.to_csv(
    FAILED_CSV,
    index=False,
    encoding="utf-8-sig"
)

nonconverged_df.to_csv(
    NONCONVERGED_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 32. 输出文件数量检查
# ============================================================

roi_count = len(
    list(
        ROI_DIR.glob(
            "*.png"
        )
    )
)

coating_mask_count = len(
    list(
        COATING_MASK_DIR.glob(
            "*.png"
        )
    )
)

body_mask_count = len(
    list(
        BODY_MASK_DIR.glob(
            "*.png"
        )
    )
)

coating_image_count = len(
    list(
        COATING_IMAGE_DIR.glob(
            "*.png"
        )
    )
)

body_image_count = len(
    list(
        BODY_IMAGE_DIR.glob(
            "*.png"
        )
    )
)


# ============================================================
# 33. 分布统计
# ============================================================

QUANTILES = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]


if success_n > 0:

    coating_quantiles = (
        success_df[
            "coating_ratio"
        ].quantile(
            QUANTILES
        )
    )

    body_quantiles = (
        success_df[
            "body_ratio"
        ].quantile(
            QUANTILES
        )
    )

    iteration_quantiles = (
        success_df[
            "iterations"
        ].quantile(
            QUANTILES
        )
    )

    delta_quantiles = (
        success_df[
            "final_delta"
        ].quantile(
            QUANTILES
        )
    )

    uncertainty_quantiles = (
        success_df[
            "uncertain_fraction"
        ].quantile(
            QUANTILES
        )
    )

else:

    coating_quantiles = (
        pd.Series(
            dtype=float
        )
    )

    body_quantiles = (
        pd.Series(
            dtype=float
        )
    )

    iteration_quantiles = (
        pd.Series(
            dtype=float
        )
    )

    delta_quantiles = (
        pd.Series(
            dtype=float
        )
    )

    uncertainty_quantiles = (
        pd.Series(
            dtype=float
        )
    )


# ============================================================
# 34. 生成随机代表病例Contact Sheet
# ============================================================

if success_n >= 24:

    random_sample = (
        success_df.sample(
            n=24,
            random_state=
                RANDOM_SEED
        )
        .sort_values(
            "image_id"
        )
        .reset_index(
            drop=True
        )
    )

else:

    random_sample = (
        success_df.copy()
    )


build_contact_sheet(

    random_sample,

    image_lookup,

    CONTACT_DIR
    /
    "01_random_24_candidate_regions.jpg"
)


# ============================================================
# 35. 再生成极端候选面积病例
# ============================================================

if success_n > 0:

    lowest_coating = (
        success_df
        .sort_values(
            "coating_ratio",
            ascending=True
        )
        .head(
            12
        )
    )

    highest_coating = (
        success_df
        .sort_values(
            "coating_ratio",
            ascending=False
        )
        .head(
            12
        )
    )

    extreme_sample = (
        pd.concat(
            [
                lowest_coating,
                highest_coating
            ],
            axis=0
        )
        .drop_duplicates(
            subset=[
                "image_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    build_contact_sheet(

        extreme_sample,

        image_lookup,

        CONTACT_DIR
        /
        "02_coating_ratio_extremes.jpg"
    )


# ============================================================
# 36. Final check
# ============================================================

all_converged = (
    success_n
    ==
    EXPECTED_N
    and
    nonconverged_n
    ==
    0
)


output_count_ok = (

    roi_count
    ==
    EXPECTED_N

    and

    coating_mask_count
    ==
    EXPECTED_N

    and

    body_mask_count
    ==
    EXPECTED_N

    and

    coating_image_count
    ==
    EXPECTED_N

    and

    body_image_count
    ==
    EXPECTED_N
)


final_pass = (

    success_n
    ==
    EXPECTED_N

    and

    failed_n
    ==
    0

    and

    all_converged

    and

    output_count_ok
)


total_runtime = (
    time.time()
    -
    overall_start
)


# ============================================================
# 37. Summary
# ============================================================

summary_lines = [

    "=" * 100,

    "Step 05 - Final Candidate Tongue-Body / Tongue-Coating Region Generation",

    "=" * 100,

    f"正式数据集：{EXPECTED_N}",

    f"成功：{success_n}",

    f"失败：{failed_n}",

    "",

    "INPUT",

    "-" * 100,

    "Original tongue images only",

    "Super-resolution: NOT USED",

    "",

    "METHOD",

    "-" * 100,

    "Tongue ROI:",
    "max RGB > 12 + largest 8-connected component + 5x5 closing",

    "",

    f"Work max side: {WORK_MAX_SIDE}",

    "",

    f"SLIC n_segments: {SLIC_N_SEGMENTS}",

    f"SLIC compactness: {SLIC_COMPACTNESS}",

    f"SLIC sigma: {SLIC_SIGMA}",

    "",

    f"Spatial FCM clusters: {FCM_CLUSTERS}",

    f"Spatial FCM m: {FCM_M}",

    f"Spatial FCM max_iter: {FCM_MAX_ITER}",

    f"Spatial FCM tolerance: {FCM_TOL}",

    f"Spatial lambda: {SPATIAL_LAMBDA}",

    "",

    "CONVERGENCE",

    "-" * 100,

    f"Converged: {success_n - nonconverged_n}",

    f"Non-converged: {nonconverged_n}",

    "",

    "Iteration quantiles:",

    iteration_quantiles.to_string(),

    "",

    "Final delta quantiles:",

    delta_quantiles.to_string(),

    "",

    "CANDIDATE COATING AREA",

    "-" * 100,

    (
        f"Mean coating ratio: "
        f"{success_df['coating_ratio'].mean():.6f}"
        if success_n > 0
        else
        "Mean coating ratio: NA"
    ),

    (
        f"Median coating ratio: "
        f"{success_df['coating_ratio'].median():.6f}"
        if success_n > 0
        else
        "Median coating ratio: NA"
    ),

    "",

    "Coating ratio quantiles:",

    coating_quantiles.to_string(),

    "",

    "CANDIDATE BODY AREA",

    "-" * 100,

    (
        f"Mean body ratio: "
        f"{success_df['body_ratio'].mean():.6f}"
        if success_n > 0
        else
        "Mean body ratio: NA"
    ),

    (
        f"Median body ratio: "
        f"{success_df['body_ratio'].median():.6f}"
        if success_n > 0
        else
        "Median body ratio: NA"
    ),

    "",

    "Body ratio quantiles:",

    body_quantiles.to_string(),

    "",

    "MEMBERSHIP UNCERTAINTY",

    "-" * 100,

    "Uncertain fraction quantiles:",

    uncertainty_quantiles.to_string(),

    "",

    "OUTPUT FILE CHECK",

    "-" * 100,

    f"ROI masks: {roi_count}",

    f"Candidate coating masks: {coating_mask_count}",

    f"Candidate body masks: {body_mask_count}",

    f"Candidate coating images: {coating_image_count}",

    f"Candidate body images: {body_image_count}",

    "",

    "RUNTIME",

    "-" * 100,

    f"Total runtime: {total_runtime:.2f} s",

    (
        f"Mean runtime per case: "
        f"{df['runtime_seconds'].mean():.3f} s"
    ),

    "",

    "IMPORTANT",

    "-" * 100,

    "1. This is the formal Original-only candidate-region generation pipeline.",

    "2. Real-ESRGAN output is not used in this formal pipeline.",

    "3. Candidate tongue-body and candidate tongue-coating regions are operational algorithmic regions.",

    "4. These masks are not expert pixel-level ground-truth segmentation.",

    "5. No case is removed according to candidate-region appearance or algorithm output.",

    "6. The formal dataset remains fixed at N=877.",

    "",

    (
        "FINAL CHECK: PASS"
        if final_pass
        else
        "FINAL CHECK: FAILED"
    ),

    "=" * 100,
]


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# 38. 控制台输出
# ============================================================

print()

print(
    "=" * 100
)

print(
    "Step 05 完成"
)

print(
    "=" * 100
)

print(
    f"成功："
    f"{success_n}/"
    f"{EXPECTED_N}"
)

print(
    f"失败："
    f"{failed_n}"
)

print(
    f"未收敛："
    f"{nonconverged_n}"
)

print()

if success_n > 0:

    print(
        f"Mean coating ratio："
        f"{success_df['coating_ratio'].mean():.4f}"
    )

    print(
        f"Median coating ratio："
        f"{success_df['coating_ratio'].median():.4f}"
    )

    print(
        f"Mean body ratio："
        f"{success_df['body_ratio'].mean():.4f}"
    )

    print(
        f"Median body ratio："
        f"{success_df['body_ratio'].median():.4f}"
    )

    print(
        f"Max final delta："
        f"{success_df['final_delta'].max():.8e}"
    )

print()

print(
    "Output files:"
)

print(
    f"ROI："
    f"{roi_count}"
)

print(
    f"Candidate coating masks："
    f"{coating_mask_count}"
)

print(
    f"Candidate body masks："
    f"{body_mask_count}"
)

print(
    f"Candidate coating images："
    f"{coating_image_count}"
)

print(
    f"Candidate body images："
    f"{body_image_count}"
)

print()

print(
    f"CSV："
    f"{OUTPUT_CSV}"
)

print(
    f"Summary："
    f"{SUMMARY_PATH}"
)

print(
    f"Contact sheets："
    f"{CONTACT_DIR}"
)

print()

if final_pass:

    print(
        "FINAL CHECK: PASS"
    )

else:

    print(
        "FINAL CHECK: FAILED"
    )

    print(
        "请先检查 failed_cases.csv "
        "和 nonconverged_cases.csv，"
        "不要进入下一步。"
    )

print(
    "=" * 100
)