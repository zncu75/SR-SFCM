from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import hashlib
import shutil
import json
from datetime import datetime


# ============================================================
# 0. 项目路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images"

RESULT_DIR = PROJECT_ROOT / "results" / "01_qc"
REVIEW_DIR = RESULT_DIR / "flagged_review"
LOG_DIR = PROJECT_ROOT / "logs"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "qc_all_images.csv"
SUMMARY_PATH = RESULT_DIR / "qc_summary.txt"
DUPLICATE_PATH = RESULT_DIR / "duplicate_groups.csv"
CONFIG_PATH = RESULT_DIR / "qc_config.json"

LOG_PATH = LOG_DIR / "01_image_quality_control.log"


# ============================================================
# 1. 支持的图像格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# 2. 当前只用于“疑似病例复核”的参数
#
# 注意：
# 这些阈值不直接决定排除！
# 自动程序只负责 FLAG。
# 最终是否排除必须人工确认。
# ============================================================

FOREGROUND_THRESHOLD = 12

ROI_MIN_AREA_RATIO = 0.10

# 对严重疑似欠曝 / 过曝的初始技术提示
DARK_GRAY_THRESHOLD = 25
BRIGHT_GRAY_THRESHOLD = 235

MAX_DARK_PIXEL_RATIO = 0.15
MAX_BRIGHT_PIXEL_RATIO = 0.15

# 模糊不使用固定绝对阈值直接排除
# 先根据整个数据集的 Laplacian variance 分布，
# 将最低 5% 标记为 possible_blur，供人工复核。
BLUR_REVIEW_QUANTILE = 0.05

WORK_MAX_SIDE = 384


# ============================================================
# 3. 日志
# ============================================================

def log(message):
    text = str(message)

    print(text)

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(text + "\n")


# ============================================================
# 4. 中文路径安全读取
# ============================================================

def imread_unicode(path: Path):

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
# 5. 文件 MD5
#
# 用于检查完全相同的文件
# ============================================================

def calculate_file_md5(path: Path):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            md5.update(block)

    return md5.hexdigest()


# ============================================================
# 6. 像素哈希
#
# 即使图片文件编码方式不同，
# 只要解码后的像素完全一致，也可以识别出来。
# ============================================================

def calculate_pixel_hash(image):

    if image is None:
        return ""

    h = hashlib.sha256()

    h.update(
        str(image.shape).encode(
            "utf-8"
        )
    )

    h.update(
        image.tobytes()
    )

    return h.hexdigest()


# ============================================================
# 7. 舌体 ROI
#
# 与之前正式候选区域生成逻辑保持一致：
#
# 1. 黑色背景阈值
# 2. 最大8连通域
# 3. 5×5闭运算
# ============================================================

def extract_tongue_roi(image):

    foreground = (
        image.max(axis=2)
        >
        FOREGROUND_THRESHOLD
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

    component_areas = (
        stats[
            1:,
            cv2.CC_STAT_AREA
        ]
    )

    largest_label = (
        1
        +
        int(
            np.argmax(
                component_areas
            )
        )
    )

    roi = (
        labels
        ==
        largest_label
    ).astype(np.uint8)

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )
    )

    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        kernel
    )

    return roi


# ============================================================
# 8. 工作分辨率
#
# 统一到最长边≤384后评价清晰度，
# 避免不同分辨率明显影响 Laplacian variance。
# ============================================================

def resize_to_work_resolution(
    image,
    roi
):

    h, w = image.shape[:2]

    max_side = max(h, w)

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

    image_resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    roi_resized = cv2.resize(
        roi,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    return (
        image_resized,
        roi_resized
    )


# ============================================================
# 9. ROI 是否接触边界
# ============================================================

def roi_border_information(roi):

    top = bool(
        np.any(roi[0, :] > 0)
    )

    bottom = bool(
        np.any(roi[-1, :] > 0)
    )

    left = bool(
        np.any(roi[:, 0] > 0)
    )

    right = bool(
        np.any(roi[:, -1] > 0)
    )

    border_count = sum([
        top,
        bottom,
        left,
        right
    ])

    return {
        "roi_touch_top": top,
        "roi_touch_bottom": bottom,
        "roi_touch_left": left,
        "roi_touch_right": right,
        "roi_border_touch_count":
            border_count,
        "roi_touches_border":
            border_count > 0,
    }


# ============================================================
# 10. 图像质量指标
# ============================================================

def calculate_quality_metrics(image):

    height, width = image.shape[:2]

    roi = extract_tongue_roi(
        image
    )

    roi_pixels = int(
        roi.sum()
    )

    image_area = (
        height
        *
        width
    )

    if roi_pixels <= 0:

        return {
            "width": width,
            "height": height,

            "roi_valid": False,
            "roi_area_pixels": 0,
            "roi_area_ratio": 0.0,

            "roi_touch_top": False,
            "roi_touch_bottom": False,
            "roi_touch_left": False,
            "roi_touch_right": False,
            "roi_border_touch_count": 0,
            "roi_touches_border": False,

            "laplacian_variance":
                np.nan,

            "tenengrad_mean":
                np.nan,

            "mean_gray":
                np.nan,

            "median_gray":
                np.nan,

            "gray_std":
                np.nan,

            "gray_p01":
                np.nan,

            "gray_p05":
                np.nan,

            "gray_p95":
                np.nan,

            "gray_p99":
                np.nan,

            "dark_pixel_ratio":
                np.nan,

            "bright_pixel_ratio":
                np.nan,
        }

    roi_ratio = (
        roi_pixels
        /
        image_area
    )

    border_info = (
        roi_border_information(
            roi
        )
    )

    image_work, roi_work = (
        resize_to_work_resolution(
            image,
            roi
        )
    )

    gray = cv2.cvtColor(
        image_work,
        cv2.COLOR_BGR2GRAY
    )

    mask = (
        roi_work > 0
    )

    gray_pixels = (
        gray[mask]
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Laplacian variance
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    laplacian_pixels = (
        laplacian[mask]
    )

    laplacian_variance = float(
        np.var(
            laplacian_pixels
        )
    )

    # --------------------------------------------------------
    # Tenengrad
    #
    # 第二个清晰度指标，
    # 后面辅助判断真正的严重模糊。
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

    gradient_energy = (
        gx ** 2
        +
        gy ** 2
    )

    tenengrad_mean = float(
        np.mean(
            gradient_energy[mask]
        )
    )

    # --------------------------------------------------------
    # ROI内灰度统计
    # --------------------------------------------------------

    mean_gray = float(
        np.mean(
            gray_pixels
        )
    )

    median_gray = float(
        np.median(
            gray_pixels
        )
    )

    gray_std = float(
        np.std(
            gray_pixels
        )
    )

    gray_p01 = float(
        np.percentile(
            gray_pixels,
            1
        )
    )

    gray_p05 = float(
        np.percentile(
            gray_pixels,
            5
        )
    )

    gray_p95 = float(
        np.percentile(
            gray_pixels,
            95
        )
    )

    gray_p99 = float(
        np.percentile(
            gray_pixels,
            99
        )
    )

    # --------------------------------------------------------
    # 严重欠曝 / 过曝候选
    # --------------------------------------------------------

    dark_pixel_ratio = float(
        np.mean(
            gray_pixels
            <
            DARK_GRAY_THRESHOLD
        )
    )

    bright_pixel_ratio = float(
        np.mean(
            gray_pixels
            >
            BRIGHT_GRAY_THRESHOLD
        )
    )

    result = {
        "width": width,
        "height": height,

        "roi_valid": True,

        "roi_area_pixels":
            roi_pixels,

        "roi_area_ratio":
            float(roi_ratio),

        "laplacian_variance":
            laplacian_variance,

        "tenengrad_mean":
            tenengrad_mean,

        "mean_gray":
            mean_gray,

        "median_gray":
            median_gray,

        "gray_std":
            gray_std,

        "gray_p01":
            gray_p01,

        "gray_p05":
            gray_p05,

        "gray_p95":
            gray_p95,

        "gray_p99":
            gray_p99,

        "dark_pixel_ratio":
            dark_pixel_ratio,

        "bright_pixel_ratio":
            bright_pixel_ratio,
    }

    result.update(
        border_info
    )

    return result


# ============================================================
# 11. 收集所有图片
# ============================================================

image_paths = sorted([
    p
    for p
    in RAW_IMAGE_DIR.rglob("*")
    if (
        p.is_file()
        and
        p.suffix.lower()
        in IMAGE_EXTENSIONS
    )
])


# ============================================================
# 12. 开始
# ============================================================

with open(
    LOG_PATH,
    "w",
    encoding="utf-8"
) as f:
    pass


log("=" * 90)
log("Step 01 - Image Quality Control")
log("=" * 90)

log(
    f"项目根目录："
    f"{PROJECT_ROOT}"
)

log(
    f"原始图片目录："
    f"{RAW_IMAGE_DIR}"
)

log(
    f"发现图片数量："
    f"{len(image_paths)}"
)

log("=" * 90)


if len(image_paths) == 0:

    raise RuntimeError(
        f"\n没有发现图片。\n"
        f"请检查路径：\n"
        f"{RAW_IMAGE_DIR}"
    )


records = []


# ============================================================
# 13. 扫描所有图片
# ============================================================

for index, path in enumerate(
    image_paths,
    start=1
):

    relative_path = (
        path.relative_to(
            RAW_IMAGE_DIR
        )
    )

    log(
        f"[{index:4d}/"
        f"{len(image_paths):4d}] "
        f"{relative_path}"
    )

    image = imread_unicode(
        path
    )

    record = {
        "image_id":
            path.stem,

        "filename":
            path.name,

        "relative_path":
            str(relative_path),

        "absolute_path":
            str(path),

        "extension":
            path.suffix.lower(),
    }

    # --------------------------------------------------------
    # 无法读取
    # --------------------------------------------------------

    if image is None:

        record.update({
            "read_success":
                False,

            "file_md5":
                "",

            "pixel_hash":
                "",

            "width":
                np.nan,

            "height":
                np.nan,

            "roi_valid":
                False,
        })

        records.append(
            record
        )

        continue

    # --------------------------------------------------------
    # 可以读取
    # --------------------------------------------------------

    record["read_success"] = (
        True
    )

    record["file_md5"] = (
        calculate_file_md5(
            path
        )
    )

    record["pixel_hash"] = (
        calculate_pixel_hash(
            image
        )
    )

    metrics = (
        calculate_quality_metrics(
            image
        )
    )

    record.update(
        metrics
    )

    records.append(
        record
    )


df = pd.DataFrame(
    records
)


# ============================================================
# 14. 完全重复检查
# ============================================================

df[
    "exact_file_duplicate"
] = False

df[
    "exact_pixel_duplicate"
] = False


# 文件MD5重复
valid_md5 = df.loc[
    df["file_md5"].astype(str)
    .str.len() > 0,
    "file_md5"
]

duplicate_md5_values = (
    valid_md5[
        valid_md5.duplicated(
            keep=False
        )
    ]
    .unique()
)

df.loc[
    df["file_md5"].isin(
        duplicate_md5_values
    ),
    "exact_file_duplicate"
] = True


# 像素完全重复
valid_pixel_hash = df.loc[
    df["pixel_hash"].astype(str)
    .str.len() > 0,
    "pixel_hash"
]

duplicate_pixel_values = (
    valid_pixel_hash[
        valid_pixel_hash.duplicated(
            keep=False
        )
    ]
    .unique()
)

df.loc[
    df["pixel_hash"].isin(
        duplicate_pixel_values
    ),
    "exact_pixel_duplicate"
] = True


# ============================================================
# 15. 数据集模糊指标分布
#
# 注意：
# P05只是“人工复核候选”，
# 不是自动排除阈值。
# ============================================================

valid_laplacian = (
    df.loc[
        (
            df["read_success"]
            ==
            True
        )
        &
        (
            df["roi_valid"]
            ==
            True
        ),
        "laplacian_variance"
    ]
    .dropna()
)


if len(valid_laplacian) > 0:

    blur_review_threshold = float(
        valid_laplacian.quantile(
            BLUR_REVIEW_QUANTILE
        )
    )

else:

    blur_review_threshold = (
        np.nan
    )


# ============================================================
# 16. 建立 QC flags
#
# 非常重要：
# flag ≠ exclude
#
# flag 只是让我们下一步重点看这些图。
# ============================================================

def generate_qc_flags(row):

    flags = []

    # --------------------------------------------------------
    # 文件无法读取
    # --------------------------------------------------------

    if not bool(
        row.get(
            "read_success",
            False
        )
    ):

        flags.append(
            "unreadable_file"
        )

        return ";".join(
            flags
        )

    # --------------------------------------------------------
    # ROI失败
    # --------------------------------------------------------

    if not bool(
        row.get(
            "roi_valid",
            False
        )
    ):

        flags.append(
            "invalid_roi"
        )

        return ";".join(
            flags
        )

    # --------------------------------------------------------
    # ROI过小
    # --------------------------------------------------------

    roi_ratio = row.get(
        "roi_area_ratio",
        np.nan
    )

    if pd.notna(
        roi_ratio
    ):

        if (
            roi_ratio
            <
            ROI_MIN_AREA_RATIO
        ):

            flags.append(
                "very_small_roi"
            )

    # --------------------------------------------------------
    # 舌体触碰边界
    #
    # 这里只提示人工查看。
    # 接触边缘不一定等于裁切。
    # --------------------------------------------------------

    if bool(
        row.get(
            "roi_touches_border",
            False
        )
    ):

        flags.append(
            "possible_incomplete_capture"
        )

    # --------------------------------------------------------
    # 模糊候选
    # --------------------------------------------------------

    lap = row.get(
        "laplacian_variance",
        np.nan
    )

    if (
        pd.notna(lap)
        and
        pd.notna(
            blur_review_threshold
        )
    ):

        if (
            lap
            <
            blur_review_threshold
        ):

            flags.append(
                "possible_blur"
            )

    # --------------------------------------------------------
    # 欠曝
    # --------------------------------------------------------

    dark_ratio = row.get(
        "dark_pixel_ratio",
        np.nan
    )

    if pd.notna(
        dark_ratio
    ):

        if (
            dark_ratio
            >
            MAX_DARK_PIXEL_RATIO
        ):

            flags.append(
                "possible_underexposure"
            )

    # --------------------------------------------------------
    # 过曝
    # --------------------------------------------------------

    bright_ratio = row.get(
        "bright_pixel_ratio",
        np.nan
    )

    if pd.notna(
        bright_ratio
    ):

        if (
            bright_ratio
            >
            MAX_BRIGHT_PIXEL_RATIO
        ):

            flags.append(
                "possible_overexposure"
            )

    # --------------------------------------------------------
    # 重复
    # --------------------------------------------------------

    if bool(
        row.get(
            "exact_file_duplicate",
            False
        )
    ):

        flags.append(
            "exact_file_duplicate"
        )

    elif bool(
        row.get(
            "exact_pixel_duplicate",
            False
        )
    ):

        flags.append(
            "exact_pixel_duplicate"
        )

    return ";".join(
        flags
    )


df["qc_flags"] = df.apply(
    generate_qc_flags,
    axis=1
)

df[
    "flagged_for_manual_review"
] = (
    df["qc_flags"]
    .astype(str)
    .str.len()
    > 0
)


# ============================================================
# 17. 人工审核字段
#
# 下一步你只需要填写这两列。
#
# KEEP
# EXCLUDE
#
# 目前全部为空，
# 绝不让程序自动替你排除。
# ============================================================

df[
    "manual_qc_decision"
] = ""

df[
    "manual_exclusion_reason"
] = ""

df[
    "manual_qc_note"
] = ""


# ============================================================
# 18. 生成 duplicate_groups.csv
# ============================================================

duplicate_rows = df[
    (
        df["exact_file_duplicate"]
        ==
        True
    )
    |
    (
        df["exact_pixel_duplicate"]
        ==
        True
    )
].copy()


duplicate_rows = duplicate_rows[
    [
        "image_id",
        "filename",
        "relative_path",
        "file_md5",
        "pixel_hash",
        "exact_file_duplicate",
        "exact_pixel_duplicate",
    ]
]


duplicate_rows.to_csv(
    DUPLICATE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 保存主 CSV
# ============================================================

df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 清理旧 flagged_review
#
# 这里只删除上一次程序自动复制出来的复核副本，
# 不会删除 raw_images 中任何原始图片。
# ============================================================

if REVIEW_DIR.exists():

    for item in REVIEW_DIR.iterdir():

        if item.is_file():

            item.unlink()


# ============================================================
# 21. 复制需要人工复核的图片
#
# 文件名前加序号，
# 方便按 CSV 查看。
# ============================================================

flagged_df = df[
    df[
        "flagged_for_manual_review"
    ]
    ==
    True
].copy()


for review_index, (_, row) in enumerate(
    flagged_df.iterrows(),
    start=1
):

    source = Path(
        row["absolute_path"]
    )

    if not source.exists():
        continue

    target_name = (
        f"{review_index:04d}_"
        f"{row['filename']}"
    )

    target = (
        REVIEW_DIR
        /
        target_name
    )

    shutil.copy2(
        source,
        target
    )


# ============================================================
# 22. 汇总各 flag 数量
# ============================================================

flag_counts = {}

for flags in df["qc_flags"]:

    if not flags:
        continue

    for flag in flags.split(";"):

        flag_counts[flag] = (
            flag_counts.get(
                flag,
                0
            )
            +
            1
        )


# ============================================================
# 23. 保存配置
# ============================================================

config = {

    "created_at":
        datetime.now().isoformat(),

    "project_root":
        str(PROJECT_ROOT),

    "raw_image_dir":
        str(RAW_IMAGE_DIR),

    "foreground_threshold":
        FOREGROUND_THRESHOLD,

    "roi_min_area_ratio":
        ROI_MIN_AREA_RATIO,

    "dark_gray_threshold":
        DARK_GRAY_THRESHOLD,

    "bright_gray_threshold":
        BRIGHT_GRAY_THRESHOLD,

    "max_dark_pixel_ratio":
        MAX_DARK_PIXEL_RATIO,

    "max_bright_pixel_ratio":
        MAX_BRIGHT_PIXEL_RATIO,

    "blur_review_quantile":
        BLUR_REVIEW_QUANTILE,

    "blur_review_threshold_actual":
        (
            None
            if pd.isna(
                blur_review_threshold
            )
            else
            blur_review_threshold
        ),

    "work_max_side":
        WORK_MAX_SIDE,

    "important_note":
        (
            "All automatic criteria are review flags only. "
            "No image is automatically excluded."
        )
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 24. summary.txt
# ============================================================

total_images = len(df)

read_failure_count = int(
    (
        df["read_success"]
        ==
        False
    ).sum()
)

invalid_roi_count = int(
    (
        df["roi_valid"]
        ==
        False
    ).sum()
)

flagged_count = int(
    df[
        "flagged_for_manual_review"
    ].sum()
)

exact_file_duplicate_count = int(
    df[
        "exact_file_duplicate"
    ].sum()
)

exact_pixel_duplicate_count = int(
    df[
        "exact_pixel_duplicate"
    ].sum()
)


summary_lines = [

    "=" * 90,

    "Step 01 - Image Quality Control Summary",

    "=" * 90,

    f"项目根目录：{PROJECT_ROOT}",

    f"原始图片目录：{RAW_IMAGE_DIR}",

    "",

    f"发现图片总数：{total_images}",

    f"读取失败：{read_failure_count}",

    f"ROI提取失败：{invalid_roi_count}",

    "",

    (
        "Laplacian variance "
        f"P05："
        f"{blur_review_threshold:.6f}"
        if pd.notna(
            blur_review_threshold
        )
        else
        "Laplacian variance P05：NA"
    ),

    "",

    (
        "文件级完全重复标记病例数："
        f"{exact_file_duplicate_count}"
    ),

    (
        "像素级完全重复标记病例数："
        f"{exact_pixel_duplicate_count}"
    ),

    "",

    (
        "进入人工复核候选的图片数："
        f"{flagged_count}"
    ),

    "",

    "各自动 FLAG 数量：",
]


for key in sorted(
    flag_counts.keys()
):

    summary_lines.append(
        f"  {key}: "
        f"{flag_counts[key]}"
    )


summary_lines.extend([

    "",

    "重要说明：",

    "1. 自动 FLAG 只用于筛选需要人工重点复核的原始图像。",

    "2. FLAG 不等于 EXCLUDE。",

    "3. 本程序没有删除、移动或修改任何 raw_images 原始图片。",

    "4. 是否排除病例必须在人工复核后确定。",

    "5. 不允许根据候选区域生成结果、VLM结果、疾病标签或后续模型性能决定是否排除。",

    "=" * 90,
])


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# 25. 控制台最终输出
# ============================================================

log("")
log("=" * 90)

log("Step 01 完成")

log("=" * 90)

log(
    f"原始图片总数："
    f"{total_images}"
)

log(
    f"读取失败："
    f"{read_failure_count}"
)

log(
    f"ROI失败："
    f"{invalid_roi_count}"
)

if pd.notna(
    blur_review_threshold
):

    log(
        "模糊人工复核阈值 "
        "(Laplacian variance P05)："
        f"{blur_review_threshold:.6f}"
    )

log(
    f"需人工重点复核："
    f"{flagged_count}"
)

log("")
log(
    f"主CSV："
    f"{CSV_PATH}"
)

log(
    f"复核图片目录："
    f"{REVIEW_DIR}"
)

log(
    f"重复记录："
    f"{DUPLICATE_PATH}"
)

log(
    f"汇总："
    f"{SUMMARY_PATH}"
)

log(
    f"配置："
    f"{CONFIG_PATH}"
)

log("")
log(
    "注意：当前没有任何图片被自动排除。"
)

log("=" * 90)