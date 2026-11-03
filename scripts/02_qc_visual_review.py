from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 0. 固定路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images"

QC_DIR = PROJECT_ROOT / "results" / "01_qc"

INPUT_CSV = QC_DIR / "qc_all_images.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "02_qc_visual_review"

FORMAT_DIR = OUTPUT_DIR / "01_format_check"
BLUR_DIR = OUTPUT_DIR / "02_blur_check"
EXPOSURE_DIR = OUTPUT_DIR / "03_exposure_check"

FORMAT_DIR.mkdir(parents=True, exist_ok=True)
BLUR_DIR.mkdir(parents=True, exist_ok=True)
EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "qc_visual_review_summary.txt"

RANDOM_SEED = 42


# ============================================================
# 1. Contact sheet 参数
# ============================================================

THUMB_W = 320
THUMB_H = 260

TEXT_H = 90

CELL_W = THUMB_W
CELL_H = THUMB_H + TEXT_H

N_COLS = 4

FORMAT_EXAMPLE_N = 24
BLUR_REVIEW_N = 30
EXPOSURE_REVIEW_N = 30


# ============================================================
# 2. 中文路径安全读取
# ============================================================

def imread_unicode(path):

    path = Path(path)

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
# 3. 保持比例缩放到缩略图
# ============================================================

def resize_with_padding(
    image,
    target_w,
    target_h
):

    h, w = image.shape[:2]

    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (
            target_h,
            target_w,
            3
        ),
        dtype=np.uint8
    )

    y0 = (
        target_h - new_h
    ) // 2

    x0 = (
        target_w - new_w
    ) // 2

    canvas[
        y0:y0 + new_h,
        x0:x0 + new_w
    ] = resized

    return canvas


# ============================================================
# 4. 生成单个缩略图格子
# ============================================================

def make_cell(
    row,
    rank=None,
    metric_name=None,
    metric_value=None
):

    image_path = Path(
        row["absolute_path"]
    )

    image = imread_unicode(
        image_path
    )

    if image is None:

        image_canvas = np.zeros(
            (
                THUMB_H,
                THUMB_W,
                3
            ),
            dtype=np.uint8
        )

        cv2.putText(
            image_canvas,
            "READ FAILED",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        image_canvas = (
            resize_with_padding(
                image,
                THUMB_W,
                THUMB_H
            )
        )

    cell = np.full(
        (
            CELL_H,
            CELL_W,
            3
        ),
        255,
        dtype=np.uint8
    )

    cell[
        0:THUMB_H,
        0:THUMB_W
    ] = image_canvas

    filename = str(
        row["filename"]
    )

    # 文件名过长则截断显示
    if len(filename) > 36:

        filename_display = (
            filename[:33] + "..."
        )

    else:

        filename_display = filename

    line1 = filename_display

    if rank is not None:

        line1 = (
            f"#{rank:02d}  "
            + line1
        )

    lap = row.get(
        "laplacian_variance",
        np.nan
    )

    bright = row.get(
        "bright_pixel_ratio",
        np.nan
    )

    roi = row.get(
        "roi_area_ratio",
        np.nan
    )

    borders = row.get(
        "roi_border_touch_count",
        np.nan
    )

    line2 = (
        f"Lap={lap:.1f}  "
        f"Bright={bright:.3f}"
        if (
            pd.notna(lap)
            and
            pd.notna(bright)
        )
        else
        "Lap/bright=NA"
    )

    line3 = (
        f"ROI={roi:.3f}  "
        f"Borders={int(borders)}"
        if (
            pd.notna(roi)
            and
            pd.notna(borders)
        )
        else
        "ROI/borders=NA"
    )

    if (
        metric_name is not None
        and
        metric_value is not None
        and
        pd.notna(metric_value)
    ):

        line3 = (
            f"{metric_name}="
            f"{metric_value:.4f}"
            f"  ROI={roi:.3f}"
        )

    text_y0 = THUMB_H + 22

    cv2.putText(
        cell,
        line1,
        (8, text_y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        cell,
        line2,
        (8, text_y0 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        cell,
        line3,
        (8, text_y0 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    return cell


# ============================================================
# 5. 生成 Contact Sheet
# ============================================================

def build_contact_sheet(
    dataframe,
    output_path,
    metric_name=None
):

    if len(dataframe) == 0:

        print(
            f"跳过空集合："
            f"{output_path.name}"
        )

        return

    n = len(dataframe)

    n_rows = int(
        np.ceil(
            n / N_COLS
        )
    )

    sheet = np.full(
        (
            n_rows * CELL_H,
            N_COLS * CELL_W,
            3
        ),
        245,
        dtype=np.uint8
    )

    for rank, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1
    ):

        r = (
            rank - 1
        ) // N_COLS

        c = (
            rank - 1
        ) % N_COLS

        metric_value = None

        if metric_name is not None:

            metric_value = row.get(
                metric_name,
                np.nan
            )

        cell = make_cell(
            row=row,
            rank=rank,
            metric_name=metric_name,
            metric_value=metric_value
        )

        y1 = r * CELL_H
        y2 = y1 + CELL_H

        x1 = c * CELL_W
        x2 = x1 + CELL_W

        sheet[
            y1:y2,
            x1:x2
        ] = cell

    # 中文路径安全保存
    ext = output_path.suffix

    success, encoded = cv2.imencode(
        ext,
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
        if ext.lower() in {
            ".jpg",
            ".jpeg"
        }
        else []
    )

    if success:

        encoded.tofile(
            str(output_path)
        )

        print(
            f"已生成："
            f"{output_path}"
        )

    else:

        raise RuntimeError(
            f"无法保存："
            f"{output_path}"
        )


# ============================================================
# 6. 读取 Step 01 CSV
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"找不到：\n"
        f"{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


print("=" * 90)
print("Step 01-B - QC Visual Review")
print("=" * 90)
print(f"读取病例：{len(df)}")


# ============================================================
# 7. 文件名类型
#
# 重点验证我们上一步观察到的：
# 短纯数字文件名是否与四边接触高度相关。
# ============================================================

df["stem"] = (
    df["filename"]
    .astype(str)
    .map(
        lambda x:
        Path(x).stem
    )
)

df["short_numeric_filename"] = (
    df["stem"]
    .str.fullmatch(
        r"\d{1,3}"
    )
    .fillna(False)
)

df["four_border_touch"] = (
    pd.to_numeric(
        df[
            "roi_border_touch_count"
        ],
        errors="coerce"
    )
    ==
    4
)


# ============================================================
# 8. 做交叉表
# ============================================================

crosstab = pd.crosstab(
    df[
        "short_numeric_filename"
    ],
    df[
        "four_border_touch"
    ],
    margins=True
)

CROSSTAB_PATH = (
    FORMAT_DIR
    /
    "format_border_crosstab.csv"
)

crosstab.to_csv(
    CROSSTAB_PATH,
    encoding="utf-8-sig"
)


# ============================================================
# 9. Format check
# ============================================================

random.seed(
    RANDOM_SEED
)


short_df = df[
    df[
        "short_numeric_filename"
    ]
].copy()


long_df = df[
    ~df[
        "short_numeric_filename"
    ]
].copy()


four_border_df = df[
    df[
        "four_border_touch"
    ]
].copy()


def deterministic_sample(
    dataframe,
    n
):

    if len(dataframe) <= n:

        return dataframe.copy()

    return dataframe.sample(
        n=n,
        random_state=RANDOM_SEED
    )


short_examples = (
    deterministic_sample(
        short_df,
        FORMAT_EXAMPLE_N
    )
)

long_examples = (
    deterministic_sample(
        long_df,
        FORMAT_EXAMPLE_N
    )
)

border_examples = (
    deterministic_sample(
        four_border_df,
        FORMAT_EXAMPLE_N
    )
)


short_examples.to_csv(
    FORMAT_DIR
    /
    "short_numeric_examples.csv",
    index=False,
    encoding="utf-8-sig"
)

long_examples.to_csv(
    FORMAT_DIR
    /
    "long_filename_examples.csv",
    index=False,
    encoding="utf-8-sig"
)

border_examples.to_csv(
    FORMAT_DIR
    /
    "four_border_examples.csv",
    index=False,
    encoding="utf-8-sig"
)


build_contact_sheet(
    short_examples,
    FORMAT_DIR
    /
    "01_short_numeric_examples.jpg"
)

build_contact_sheet(
    long_examples,
    FORMAT_DIR
    /
    "02_long_filename_examples.jpg"
)

build_contact_sheet(
    border_examples,
    FORMAT_DIR
    /
    "03_four_border_touch_examples.jpg"
)


# ============================================================
# 10. 清晰度最低30例
# ============================================================

blur_df = (
    df[
        df[
            "laplacian_variance"
        ].notna()
    ]
    .sort_values(
        "laplacian_variance",
        ascending=True
    )
    .head(
        BLUR_REVIEW_N
    )
    .copy()
)


blur_df.to_csv(
    BLUR_DIR
    /
    "blur_bottom_30.csv",
    index=False,
    encoding="utf-8-sig"
)


build_contact_sheet(
    blur_df,
    BLUR_DIR
    /
    "blur_bottom_30.jpg",
    metric_name=
        "laplacian_variance"
)


# ============================================================
# 11. 高亮比例最高30例
# ============================================================

exposure_df = (
    df[
        df[
            "bright_pixel_ratio"
        ].notna()
    ]
    .sort_values(
        "bright_pixel_ratio",
        ascending=False
    )
    .head(
        EXPOSURE_REVIEW_N
    )
    .copy()
)


exposure_df.to_csv(
    EXPOSURE_DIR
    /
    "overexposure_top_30.csv",
    index=False,
    encoding="utf-8-sig"
)


build_contact_sheet(
    exposure_df,
    EXPOSURE_DIR
    /
    "overexposure_top_30.jpg",
    metric_name=
        "bright_pixel_ratio"
)


# ============================================================
# 12. 指标分布统计
# ============================================================

def get_percentiles(
    series,
    percentiles
):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    result = {}

    if len(s) == 0:

        return result

    for p in percentiles:

        result[p] = float(
            s.quantile(p)
        )

    return result


lap_stats = get_percentiles(
    df["laplacian_variance"],
    [
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
        1.00
    ]
)

bright_stats = get_percentiles(
    df["bright_pixel_ratio"],
    [
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
        1.00
    ]
)

roi_stats = get_percentiles(
    df["roi_area_ratio"],
    [
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
        1.00
    ]
)


# ============================================================
# 13. ROI边界接触数量分布
# ============================================================

border_distribution = (
    df[
        "roi_border_touch_count"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# ============================================================
# 14. FLAG组合
# ============================================================

flag_distribution = (
    df["qc_flags"]
    .fillna("")
    .replace(
        "",
        "NO_FLAG"
    )
    .value_counts()
)


# ============================================================
# 15. 保存 summary
# ============================================================

lines = []

lines.append(
    "=" * 90
)

lines.append(
    "Step 01-B - QC Visual Review Summary"
)

lines.append(
    "=" * 90
)

lines.append(
    f"总病例数：{len(df)}"
)

lines.append("")

lines.append(
    f"短纯数字文件名（1-3位）："
    f"{int(df['short_numeric_filename'].sum())}"
)

lines.append(
    f"其他文件名："
    f"{int((~df['short_numeric_filename']).sum())}"
)

lines.append(
    f"ROI四边均接触："
    f"{int(df['four_border_touch'].sum())}"
)

lines.append("")

lines.append(
    "文件名类型 × ROI四边接触交叉表："
)

lines.append(
    crosstab.to_string()
)

lines.append("")

lines.append(
    "-" * 90
)

lines.append(
    "Laplacian variance 分位数"
)

for p, value in lap_stats.items():

    lines.append(
        f"P{int(p * 100):02d}: "
        f"{value:.6f}"
    )

lines.append("")

lines.append(
    "-" * 90
)

lines.append(
    "Bright pixel ratio 分位数"
)

for p, value in bright_stats.items():

    lines.append(
        f"P{int(p * 100):02d}: "
        f"{value:.6f}"
    )

lines.append("")

lines.append(
    "-" * 90
)

lines.append(
    "ROI area ratio 分位数"
)

for p, value in roi_stats.items():

    lines.append(
        f"P{int(p * 100):02d}: "
        f"{value:.6f}"
    )

lines.append("")

lines.append(
    "-" * 90
)

lines.append(
    "ROI接触边界数量分布"
)

for k, v in (
    border_distribution.items()
):

    lines.append(
        f"{k}: {v}"
    )

lines.append("")

lines.append(
    "-" * 90
)

lines.append(
    "QC flag组合"
)

for k, v in (
    flag_distribution.items()
):

    lines.append(
        f"{k}: {v}"
    )

lines.append("")

lines.append(
    "=" * 90
)

lines.append(
    "注意：本步骤仅用于可视化与规则核查，不执行任何病例排除。"
)

lines.append(
    "=" * 90
)


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(lines)
    )


# ============================================================
# 16. 控制台
# ============================================================

print("")
print("=" * 90)
print("Step 01-B 完成")
print("=" * 90)

print(
    f"短纯数字文件名："
    f"{int(df['short_numeric_filename'].sum())}"
)

print(
    f"ROI四边均接触："
    f"{int(df['four_border_touch'].sum())}"
)

print("")
print("请优先查看下面4张图片：")

print(
    FORMAT_DIR
    /
    "01_short_numeric_examples.jpg"
)

print(
    FORMAT_DIR
    /
    "02_long_filename_examples.jpg"
)

print(
    BLUR_DIR
    /
    "blur_bottom_30.jpg"
)

print(
    EXPOSURE_DIR
    /
    "overexposure_top_30.jpg"
)

print("")

print(
    f"详细汇总："
    f"{SUMMARY_PATH}"
)

print("=" * 90)