from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import random


# ============================================================
# 0. 路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images"

OUTPUT_DIR = PROJECT_ROOT / "results" / "03_qc_adaptive_roi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "qc_all_images_adaptive_roi.csv"
SUMMARY_PATH = OUTPUT_DIR / "adaptive_roi_summary.txt"

SHORT_SHEET_PATH = OUTPUT_DIR / "01_short_numeric_roi_compare.jpg"
LONG_SHEET_PATH = OUTPUT_DIR / "02_long_filename_roi_compare.jpg"

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

WORK_MAX_SIDE = 384


# ============================================================
# 1. 中文路径安全读取
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
# 2. 旧版 ROI
#
# 仅用于与原来的固定阈值方式比较。
# 后续正式流程不再使用它。
# ============================================================

def extract_old_roi(image):

    foreground = (
        image.max(axis=2) > 12
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8
        )
    )

    if n_labels <= 1:

        return np.zeros(
            foreground.shape,
            dtype=np.uint8
        )

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_label = (
        1 + int(np.argmax(areas))
    )

    roi = (
        labels == largest_label
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        kernel
    )

    return roi


# ============================================================
# 3. 保留最大连通域
# ============================================================

def keep_largest_component(mask):

    mask = (
        mask > 0
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if n_labels <= 1:

        return np.zeros_like(mask)

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_label = (
        1 + int(np.argmax(areas))
    )

    return (
        labels == largest_label
    ).astype(np.uint8)


# ============================================================
# 4. 填充 ROI 内部小孔洞
# ============================================================

def fill_internal_holes(mask):

    mask = (
        mask > 0
    ).astype(np.uint8)

    inverse = (
        1 - mask
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            inverse,
            connectivity=8
        )
    )

    h, w = mask.shape

    result = mask.copy()

    for label_id in range(
        1,
        n_labels
    ):

        component = (
            labels == label_id
        )

        ys, xs = np.where(
            component
        )

        if len(xs) == 0:
            continue

        touches_border = (
            np.any(xs == 0)
            or
            np.any(xs == w - 1)
            or
            np.any(ys == 0)
            or
            np.any(ys == h - 1)
        )

        # 不接触图像边界的背景区域即内部孔洞
        if not touches_border:

            result[
                component
            ] = 1

    return result


# ============================================================
# 5. Otsu 自适应 ROI
# ============================================================

def extract_otsu_roi(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    otsu_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )

    h, w = binary.shape

    # --------------------------------------------------------
    # 建立图像最外围约2%的边界带
    # --------------------------------------------------------

    border_width = max(
        2,
        int(
            round(
                min(h, w) * 0.02
            )
        )
    )

    border_mask = np.zeros(
        (h, w),
        dtype=bool
    )

    border_mask[
        :border_width,
        :
    ] = True

    border_mask[
        -border_width:,
        :
    ] = True

    border_mask[
        :,
        :border_width
    ] = True

    border_mask[
        :,
        -border_width:
    ] = True

    border_foreground_ratio = float(
        np.mean(
            binary[
                border_mask
            ] > 0
        )
    )

    # --------------------------------------------------------
    # 如果边缘大部分被判成前景，
    # 说明Otsu极性反了
    # --------------------------------------------------------

    if border_foreground_ratio > 0.50:

        binary = (
            255 - binary
        )

    mask = (
        binary > 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 形态学处理
    # --------------------------------------------------------

    kernel_close = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )
    )

    kernel_open = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    mask = keep_largest_component(
        mask
    )

    mask = fill_internal_holes(
        mask
    )

    return (
        mask,
        float(otsu_value),
        border_foreground_ratio
    )


# ============================================================
# 6. Lab背景距离备用ROI
#
# 当Otsu产生极端异常ROI时使用
# ============================================================

def extract_background_distance_roi(
    image
):

    h, w = image.shape[:2]

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    corner_size = max(
        8,
        int(
            round(
                min(h, w) * 0.08
            )
        )
    )

    corner_size = min(
        corner_size,
        50
    )

    corners = np.concatenate(
        [
            lab[
                :corner_size,
                :corner_size
            ].reshape(-1, 3),

            lab[
                :corner_size,
                -corner_size:
            ].reshape(-1, 3),

            lab[
                -corner_size:,
                :corner_size
            ].reshape(-1, 3),

            lab[
                -corner_size:,
                -corner_size:
            ].reshape(-1, 3),
        ],
        axis=0
    )

    background_lab = np.median(
        corners,
        axis=0
    )

    distance = np.sqrt(
        np.sum(
            (
                lab
                -
                background_lab
            ) ** 2,
            axis=2
        )
    )

    corner_distance = np.sqrt(
        np.sum(
            (
                corners
                -
                background_lab
            ) ** 2,
            axis=1
        )
    )

    # 背景自然波动 + 最低安全阈值
    threshold = max(
        12.0,
        float(
            np.percentile(
                corner_distance,
                99
            )
        )
        +
        5.0
    )

    # 防止极少数污染角点把阈值推得太高
    threshold = min(
        threshold,
        45.0
    )

    mask = (
        distance > threshold
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = keep_largest_component(
        mask
    )

    mask = fill_internal_holes(
        mask
    )

    return (
        mask,
        float(threshold)
    )


# ============================================================
# 7. 最终自适应ROI
# ============================================================

def extract_adaptive_roi(image):

    (
        roi,
        otsu_value,
        border_fg_ratio
    ) = extract_otsu_roi(
        image
    )

    area_ratio = float(
        roi.mean()
    )

    method = "otsu"

    lab_threshold = np.nan

    # --------------------------------------------------------
    # Otsu ROI若明显异常，则进入Lab背景距离备用方案
    #
    # 这些范围这里只用于识别算法失败，
    # 不是临床/图像质量排除标准。
    # --------------------------------------------------------

    if (
        area_ratio < 0.15
        or
        area_ratio > 0.95
    ):

        (
            roi_lab,
            lab_threshold
        ) = extract_background_distance_roi(
            image
        )

        lab_area_ratio = float(
            roi_lab.mean()
        )

        # 只有备用结果明显更合理才替换
        if (
            0.15
            <=
            lab_area_ratio
            <=
            0.95
        ):

            roi = roi_lab

            area_ratio = (
                lab_area_ratio
            )

            method = (
                "lab_background_distance"
            )

    return {
        "roi": roi,

        "roi_method": method,

        "otsu_threshold":
            otsu_value,

        "otsu_border_foreground_ratio":
            border_fg_ratio,

        "lab_background_threshold":
            lab_threshold,
    }


# ============================================================
# 8. ROI边界信息
# ============================================================

def get_border_info(roi):

    top = bool(
        np.any(
            roi[0, :] > 0
        )
    )

    bottom = bool(
        np.any(
            roi[-1, :] > 0
        )
    )

    left = bool(
        np.any(
            roi[:, 0] > 0
        )
    )

    right = bool(
        np.any(
            roi[:, -1] > 0
        )
    )

    count = sum(
        [
            top,
            bottom,
            left,
            right
        ]
    )

    return {
        "touch_top": top,
        "touch_bottom": bottom,
        "touch_left": left,
        "touch_right": right,
        "border_touch_count":
            int(count),
    }


# ============================================================
# 9. 工作分辨率
# ============================================================

def resize_work(
    image,
    roi
):

    h, w = image.shape[:2]

    max_side = max(
        h,
        w
    )

    if max_side <= WORK_MAX_SIDE:

        return image, roi

    scale = (
        WORK_MAX_SIDE
        /
        max_side
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    image_r = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    roi_r = cv2.resize(
        roi,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    return (
        image_r,
        roi_r
    )


# ============================================================
# 10. QC指标重新计算
# ============================================================

def calculate_metrics(
    image,
    roi
):

    h, w = image.shape[:2]

    roi_area_ratio = float(
        roi.mean()
    )

    image_work, roi_work = (
        resize_work(
            image,
            roi
        )
    )

    mask = (
        roi_work > 0
    )

    if not np.any(mask):

        return {
            "roi_area_ratio":
                roi_area_ratio,

            "laplacian_variance":
                np.nan,

            "tenengrad_mean":
                np.nan,

            "mean_gray":
                np.nan,

            "bright_pixel_ratio":
                np.nan,

            "near_white_clip_ratio":
                np.nan,
        }

    gray = cv2.cvtColor(
        image_work,
        cv2.COLOR_BGR2GRAY
    )

    gray_pixels = (
        gray[mask]
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Laplacian
    # --------------------------------------------------------

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    laplacian_variance = float(
        np.var(
            lap[mask]
        )
    )

    # --------------------------------------------------------
    # Tenengrad
    # --------------------------------------------------------

    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    tenengrad = (
        gx ** 2
        +
        gy ** 2
    )

    tenengrad_mean = float(
        np.mean(
            tenengrad[mask]
        )
    )

    # --------------------------------------------------------
    # 原Bright ratio
    # 仅作辅助描述
    # --------------------------------------------------------

    bright_pixel_ratio = float(
        np.mean(
            gray_pixels > 235
        )
    )

    # --------------------------------------------------------
    # 更严格的接近白色通道饱和比例
    #
    # 三通道均接近255才计入，
    # 比单纯灰度>235更接近真正的白色剪切。
    # --------------------------------------------------------

    roi_bgr = (
        image_work[
            mask
        ]
    )

    near_white_clip_ratio = float(
        np.mean(
            np.min(
                roi_bgr,
                axis=1
            )
            >=
            245
        )
    )

    return {
        "roi_area_ratio":
            roi_area_ratio,

        "laplacian_variance":
            laplacian_variance,

        "tenengrad_mean":
            tenengrad_mean,

        "mean_gray":
            float(
                np.mean(
                    gray_pixels
                )
            ),

        "bright_pixel_ratio":
            bright_pixel_ratio,

        "near_white_clip_ratio":
            near_white_clip_ratio,
    }


# ============================================================
# 11. ROI Dice
#
# 仅用于比较旧ROI与新ROI变化，
# 不是准确率。
# ============================================================

def dice_score(
    a,
    b
):

    a = (
        a > 0
    )

    b = (
        b > 0
    )

    intersection = np.logical_and(
        a,
        b
    ).sum()

    denom = (
        a.sum()
        +
        b.sum()
    )

    if denom == 0:
        return np.nan

    return float(
        2.0
        *
        intersection
        /
        denom
    )


# ============================================================
# 12. 扫描图片
# ============================================================

image_paths = sorted(
    [
        p
        for p
        in RAW_IMAGE_DIR.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]
)

print("=" * 90)
print("Step 01-C - Adaptive ROI QC")
print("=" * 90)
print(f"图片数量：{len(image_paths)}")

records = []


for index, path in enumerate(
    image_paths,
    start=1
):

    print(
        f"[{index:4d}/"
        f"{len(image_paths):4d}] "
        f"{path.name}"
    )

    image = imread_unicode(
        path
    )

    if image is None:

        records.append(
            {
                "filename":
                    path.name,

                "absolute_path":
                    str(path),

                "read_success":
                    False,
            }
        )

        continue

    old_roi = extract_old_roi(
        image
    )

    adaptive = extract_adaptive_roi(
        image
    )

    new_roi = adaptive[
        "roi"
    ]

    old_border = get_border_info(
        old_roi
    )

    new_border = get_border_info(
        new_roi
    )

    metrics = calculate_metrics(
        image,
        new_roi
    )

    stem = path.stem

    short_numeric = bool(
        stem.isdigit()
        and
        len(stem) <= 3
    )

    record = {
        "filename":
            path.name,

        "absolute_path":
            str(path),

        "read_success":
            True,

        "short_numeric_filename":
            short_numeric,

        "roi_method":
            adaptive[
                "roi_method"
            ],

        "otsu_threshold":
            adaptive[
                "otsu_threshold"
            ],

        "otsu_border_foreground_ratio":
            adaptive[
                "otsu_border_foreground_ratio"
            ],

        "lab_background_threshold":
            adaptive[
                "lab_background_threshold"
            ],

        "old_roi_area_ratio":
            float(
                old_roi.mean()
            ),

        "new_roi_area_ratio":
            float(
                new_roi.mean()
            ),

        "old_border_touch_count":
            old_border[
                "border_touch_count"
            ],

        "new_border_touch_count":
            new_border[
                "border_touch_count"
            ],

        "old_new_roi_dice":
            dice_score(
                old_roi,
                new_roi
            ),
    }

    record.update(
        metrics
    )

    records.append(
        record
    )


df = pd.DataFrame(
    records
)

df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 数据集统计
# ============================================================

valid_df = df[
    df["read_success"] == True
].copy()

short_df = valid_df[
    valid_df[
        "short_numeric_filename"
    ]
    ==
    True
].copy()

long_df = valid_df[
    valid_df[
        "short_numeric_filename"
    ]
    ==
    False
].copy()


def q(series, value):

    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
        .quantile(value)
    )


# ============================================================
# 14. Contact sheet
# ============================================================

def draw_roi_boundary(
    image,
    roi
):

    output = image.copy()

    contours, _ = cv2.findContours(
        (
            roi.astype(np.uint8)
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        3
    )

    return output


def resize_with_padding(
    image,
    width,
    height
):

    h, w = image.shape[:2]

    scale = min(
        width / w,
        height / h
    )

    nw = max(
        1,
        int(round(w * scale))
    )

    nh = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        image,
        (nw, nh),
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
        width - nw
    ) // 2

    y = (
        height - nh
    ) // 2

    canvas[
        y:y + nh,
        x:x + nw
    ] = resized

    return canvas


def build_compare_sheet(
    sample_df,
    output_path
):

    # 每例：
    # Original | Old ROI | Adaptive ROI

    panel_w = 260
    panel_h = 220

    text_h = 75

    case_w = panel_w * 3
    case_h = panel_h + text_h

    n_cols = 1
    n_rows = len(
        sample_df
    )

    sheet = np.full(
        (
            n_rows * case_h,
            case_w,
            3
        ),
        255,
        dtype=np.uint8
    )

    for i, (_, row) in enumerate(
        sample_df.iterrows()
    ):

        image = imread_unicode(
            Path(
                row[
                    "absolute_path"
                ]
            )
        )

        old_roi = extract_old_roi(
            image
        )

        new_roi = extract_adaptive_roi(
            image
        )["roi"]

        original = resize_with_padding(
            image,
            panel_w,
            panel_h
        )

        old_overlay = resize_with_padding(
            draw_roi_boundary(
                image,
                old_roi
            ),
            panel_w,
            panel_h
        )

        new_overlay = resize_with_padding(
            draw_roi_boundary(
                image,
                new_roi
            ),
            panel_w,
            panel_h
        )

        y0 = i * case_h

        sheet[
            y0:y0 + panel_h,
            0:panel_w
        ] = original

        sheet[
            y0:y0 + panel_h,
            panel_w:panel_w * 2
        ] = old_overlay

        sheet[
            y0:y0 + panel_h,
            panel_w * 2:panel_w * 3
        ] = new_overlay

        cv2.putText(
            sheet,
            "Original",
            (
                10,
                y0 + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            "Old ROI",
            (
                panel_w + 10,
                y0 + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            "Adaptive ROI",
            (
                panel_w * 2 + 10,
                y0 + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        line1 = (
            f"#{i + 1:02d} "
            f"{row['filename']}"
        )

        line2 = (
            f"OldArea="
            f"{row['old_roi_area_ratio']:.3f} "
            f"NewArea="
            f"{row['new_roi_area_ratio']:.3f} "
            f"Dice="
            f"{row['old_new_roi_dice']:.3f}"
        )

        line3 = (
            f"OldBorder="
            f"{int(row['old_border_touch_count'])} "
            f"NewBorder="
            f"{int(row['new_border_touch_count'])}"
        )

        text_y = (
            y0
            +
            panel_h
            +
            20
        )

        cv2.putText(
            sheet,
            line1[:90],
            (8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            line2,
            (
                8,
                text_y + 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            sheet,
            line3,
            (
                8,
                text_y + 44
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    success, encoded = cv2.imencode(
        ".jpg",
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not success:

        raise RuntimeError(
            "Contact sheet保存失败"
        )

    encoded.tofile(
        str(output_path)
    )


# ============================================================
# 15. 固定抽取代表例
# ============================================================

random.seed(
    RANDOM_SEED
)

short_sample = short_df.sample(
    n=min(
        12,
        len(short_df)
    ),
    random_state=RANDOM_SEED
)

long_sample = long_df.sample(
    n=min(
        12,
        len(long_df)
    ),
    random_state=RANDOM_SEED
)

build_compare_sheet(
    short_sample,
    SHORT_SHEET_PATH
)

build_compare_sheet(
    long_sample,
    LONG_SHEET_PATH
)


# ============================================================
# 16. Summary
# ============================================================

summary = []

summary.append(
    "=" * 90
)

summary.append(
    "Step 01-C - Adaptive ROI QC Summary"
)

summary.append(
    "=" * 90
)

summary.append(
    f"总图像：{len(df)}"
)

summary.append(
    f"成功读取：{len(valid_df)}"
)

summary.append("")

summary.append(
    f"短纯数字文件：{len(short_df)}"
)

summary.append(
    f"其他文件：{len(long_df)}"
)

summary.append("")

summary.append(
    "旧ROI边界接触分布："
)

summary.append(
    valid_df[
        "old_border_touch_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

summary.append("")

summary.append(
    "新ROI边界接触分布："
)

summary.append(
    valid_df[
        "new_border_touch_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

summary.append("")

summary.append(
    "短数字文件 - 新ROI边界接触分布："
)

summary.append(
    short_df[
        "new_border_touch_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

summary.append("")

summary.append(
    "其他文件 - 新ROI边界接触分布："
)

summary.append(
    long_df[
        "new_border_touch_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

summary.append("")

summary.append(
    "-" * 90
)

summary.append(
    "Adaptive ROI area ratio"
)

for percentile in [
    0,
    0.01,
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
    0.99,
    1.0
]:

    summary.append(
        f"P{int(percentile * 100):02d}: "
        f"{q(valid_df['new_roi_area_ratio'], percentile):.6f}"
    )

summary.append("")

summary.append(
    "-" * 90
)

summary.append(
    "Adaptive ROI Laplacian variance"
)

for percentile in [
    0,
    0.01,
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
    0.99,
    1.0
]:

    summary.append(
        f"P{int(percentile * 100):02d}: "
        f"{q(valid_df['laplacian_variance'], percentile):.6f}"
    )

summary.append("")

summary.append(
    "-" * 90
)

summary.append(
    "Near-white clipping ratio"
)

for percentile in [
    0,
    0.50,
    0.90,
    0.95,
    0.99,
    1.0
]:

    summary.append(
        f"P{int(percentile * 100):02d}: "
        f"{q(valid_df['near_white_clip_ratio'], percentile):.6f}"
    )

summary.append("")

summary.append(
    "ROI方法使用数量："
)

summary.append(
    valid_df[
        "roi_method"
    ]
    .value_counts()
    .to_string()
)

summary.append("")

summary.append(
    "=" * 90
)

summary.append(
    "注意：本步骤只验证和修正ROI提取方式，没有排除任何病例。"
)

summary.append(
    "=" * 90
)


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary
        )
    )


# ============================================================
# 17. 控制台
# ============================================================

print("")
print("=" * 90)
print("Step 01-C 完成")
print("=" * 90)

print(
    f"结果CSV："
    f"{CSV_PATH}"
)

print(
    f"汇总："
    f"{SUMMARY_PATH}"
)

print("")
print("请重点查看：")

print(
    SHORT_SHEET_PATH
)

print(
    LONG_SHEET_PATH
)

print("")
print(
    "当前没有删除、移动或修改任何原始图片。"
)

print("=" * 90)