from pathlib import Path
import shutil
import random

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "03_qc_adaptive_roi"
    / "qc_all_images_adaptive_roi.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "04_final_qc_review"
)

REVIEW_IMAGE_DIR = (
    OUTPUT_DIR
    / "review_images"
)

CONTACT_DIR = (
    OUTPUT_DIR
    / "contact_sheets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REVIEW_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONTACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REVIEW_CSV = (
    OUTPUT_DIR
    / "final_qc_review_cases.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_qc_review_summary.txt"
)


# ============================================================
# 1. 固定参数
#
# 注意：
# 全部只是“进入人工复核”的阈值，
# 绝对不是自动排除阈值。
# ============================================================

RANDOM_SEED = 42

# 严重模糊候选：
# Laplacian最低3% AND Tenengrad最低5%
BLUR_LAPLACIAN_QUANTILE = 0.03
BLUR_TENENGRAD_QUANTILE = 0.05

# 极端白色剪切：
# near-white clipping最高1%
CLIPPING_QUANTILE = 0.99

# ROI面积极端：
# 只为了人工确认有没有真正裁切/异常
ROI_LOW_QUANTILE = 0.01
ROI_HIGH_QUANTILE = 0.99

# 随机抽普通病例做漏检审计
RANDOM_AUDIT_N = 30


# ============================================================
# 2. Contact sheet参数
# ============================================================

THUMB_W = 320
THUMB_H = 260
TEXT_H = 105

CELL_W = THUMB_W
CELL_H = THUMB_H + TEXT_H

N_COLS = 4


# ============================================================
# 3. 中文路径读取
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
# 4. 保持比例缩放
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

    x0 = (
        target_w - new_w
    ) // 2

    y0 = (
        target_h - new_h
    ) // 2

    canvas[
        y0:y0 + new_h,
        x0:x0 + new_w
    ] = resized

    return canvas


# ============================================================
# 5. 生成单格
# ============================================================

def make_cell(
    row,
    rank
):

    path = Path(
        row["absolute_path"]
    )

    image = imread_unicode(
        path
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
            (30, 130),
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
        :THUMB_H,
        :
    ] = image_canvas

    filename = str(
        row["filename"]
    )

    if len(filename) > 34:

        filename_show = (
            filename[:31]
            +
            "..."
        )

    else:

        filename_show = filename

    lap = row.get(
        "laplacian_variance",
        np.nan
    )

    ten = row.get(
        "tenengrad_mean",
        np.nan
    )

    clip = row.get(
        "near_white_clip_ratio",
        np.nan
    )

    roi = row.get(
        "new_roi_area_ratio",
        np.nan
    )

    reasons = str(
        row.get(
            "review_reason",
            ""
        )
    )

    line1 = (
        f"#{rank:02d} "
        f"{filename_show}"
    )

    line2 = (
        f"Lap={lap:.1f}  "
        f"Ten={ten:.1f}"
    )

    line3 = (
        f"Clip={clip:.4f}  "
        f"ROI={roi:.3f}"
    )

    line4 = (
        "Reason="
        +
        reasons[:42]
    )

    y = THUMB_H + 20

    for line in [
        line1,
        line2,
        line3,
        line4
    ]:

        cv2.putText(
            cell,
            line,
            (7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        y += 22

    return cell


# ============================================================
# 6. Contact sheet
# ============================================================

def build_contact_sheet(
    dataframe,
    output_path
):

    if len(dataframe) == 0:

        return

    n_rows = int(
        np.ceil(
            len(dataframe)
            /
            N_COLS
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

        cell = make_cell(
            row,
            rank
        )

        y1 = r * CELL_H
        y2 = y1 + CELL_H

        x1 = c * CELL_W
        x2 = x1 + CELL_W

        sheet[
            y1:y2,
            x1:x2
        ] = cell

    ok, encoded = cv2.imencode(
        ".jpg",
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:

        raise RuntimeError(
            f"无法生成：{output_path}"
        )

    encoded.tofile(
        str(output_path)
    )


# ============================================================
# 7. 读取数据
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"找不到输入CSV：\n"
        f"{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


df = df[
    df["read_success"] == True
].copy()


print("=" * 90)
print("Step 01-D - Final Image Quality Review")
print("=" * 90)
print(f"有效图片：{len(df)}")


# ============================================================
# 8. 转换数值列
# ============================================================

numeric_columns = [
    "laplacian_variance",
    "tenengrad_mean",
    "near_white_clip_ratio",
    "new_roi_area_ratio",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 9. 根据931张真实分布计算阈值
# ============================================================

lap_threshold = float(
    df[
        "laplacian_variance"
    ].quantile(
        BLUR_LAPLACIAN_QUANTILE
    )
)

ten_threshold = float(
    df[
        "tenengrad_mean"
    ].quantile(
        BLUR_TENENGRAD_QUANTILE
    )
)

clip_threshold = float(
    df[
        "near_white_clip_ratio"
    ].quantile(
        CLIPPING_QUANTILE
    )
)

roi_low_threshold = float(
    df[
        "new_roi_area_ratio"
    ].quantile(
        ROI_LOW_QUANTILE
    )
)

roi_high_threshold = float(
    df[
        "new_roi_area_ratio"
    ].quantile(
        ROI_HIGH_QUANTILE
    )
)


# ============================================================
# 10. 各类候选
# ============================================================

# ------------------------------------------------------------
# A. 严重模糊候选
# 必须两个指标同时低
# ------------------------------------------------------------

df["flag_blur"] = (
    (
        df["laplacian_variance"]
        <=
        lap_threshold
    )
    &
    (
        df["tenengrad_mean"]
        <=
        ten_threshold
    )
)


# ------------------------------------------------------------
# B. 白色通道剪切极端
# ------------------------------------------------------------

df["flag_clipping"] = (
    df[
        "near_white_clip_ratio"
    ]
    >=
    clip_threshold
)


# ------------------------------------------------------------
# C. ROI面积异常低
#
# 只是检查有没有真的舌体不完整
# ------------------------------------------------------------

df["flag_roi_low"] = (
    df[
        "new_roi_area_ratio"
    ]
    <=
    roi_low_threshold
)


# ------------------------------------------------------------
# D. ROI面积异常高
#
# 只是检查极端裁剪/构图
# ------------------------------------------------------------

df["flag_roi_high"] = (
    df[
        "new_roi_area_ratio"
    ]
    >=
    roi_high_threshold
)


# ============================================================
# 11. 构建原因
# ============================================================

def build_reason(row):

    reasons = []

    if row["flag_blur"]:

        reasons.append(
            "possible_severe_blur"
        )

    if row["flag_clipping"]:

        reasons.append(
            "possible_exposure_clipping"
        )

    if row["flag_roi_low"]:

        reasons.append(
            "extreme_low_roi_area"
        )

    if row["flag_roi_high"]:

        reasons.append(
            "extreme_high_roi_area"
        )

    return ";".join(
        reasons
    )


df["review_reason"] = df.apply(
    build_reason,
    axis=1
)


df["automatic_review_candidate"] = (
    df["review_reason"]
    .astype(str)
    .str.len()
    >
    0
)


# ============================================================
# 12. 自动候选集
# ============================================================

automatic_df = df[
    df[
        "automatic_review_candidate"
    ]
].copy()


# ============================================================
# 13. 随机普通病例审计
#
# 从没有任何自动flag的人中抽30例。
# ============================================================

normal_pool = df[
    ~df[
        "automatic_review_candidate"
    ]
].copy()


random_audit_df = normal_pool.sample(
    n=min(
        RANDOM_AUDIT_N,
        len(normal_pool)
    ),
    random_state=RANDOM_SEED
).copy()


random_audit_df[
    "review_reason"
] = "random_quality_audit"


# ============================================================
# 14. 合并正式人工QC病例
# ============================================================

review_df = pd.concat(
    [
        automatic_df,
        random_audit_df
    ],
    axis=0,
    ignore_index=True
)


# 防止任何重复
review_df = review_df.drop_duplicates(
    subset=[
        "filename"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 15. 固定盲审式QC编号
# ============================================================

review_df[
    "qc_review_id"
] = [
    f"QC{i:03d}"
    for i in range(
        1,
        len(review_df) + 1
    )
]


# ============================================================
# 16. 人工填写列
#
# manual_decision 只能：
# KEEP
# EXCLUDE
#
# 排除原因建议只能使用以下几种：
#
# severe_defocus_or_motion_blur
# severe_exposure_clipping
# substantial_tongue_truncation
# severe_occlusion_or_acquisition_artifact
# unreadable_or_corrupted
# other_technical_failure
# ============================================================

review_df[
    "manual_decision"
] = ""

review_df[
    "manual_exclusion_reason"
] = ""

review_df[
    "manual_note"
] = ""


# ============================================================
# 17. 整理CSV列顺序
# ============================================================

output_columns = [
    "qc_review_id",
    "filename",
    "absolute_path",

    "review_reason",

    "laplacian_variance",
    "tenengrad_mean",
    "near_white_clip_ratio",
    "new_roi_area_ratio",

    "flag_blur",
    "flag_clipping",
    "flag_roi_low",
    "flag_roi_high",

    "manual_decision",
    "manual_exclusion_reason",
    "manual_note",
]


review_df[
    output_columns
].to_csv(
    REVIEW_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 清空旧review_images
#
# 只清除自动复制出来的副本。
# 不碰raw_images。
# ============================================================

for item in REVIEW_IMAGE_DIR.iterdir():

    if item.is_file():

        item.unlink()


# ============================================================
# 19. 复制人工复核图片
# ============================================================

for _, row in review_df.iterrows():

    src = Path(
        row["absolute_path"]
    )

    if not src.exists():

        continue

    target = (
        REVIEW_IMAGE_DIR
        /
        (
            f"{row['qc_review_id']}_"
            f"{row['filename']}"
        )
    )

    shutil.copy2(
        src,
        target
    )


# ============================================================
# 20. 分组Contact Sheet
# ============================================================

blur_df = review_df[
    review_df[
        "review_reason"
    ].str.contains(
        "possible_severe_blur",
        regex=False
    )
].copy()


clip_df = review_df[
    review_df[
        "review_reason"
    ].str.contains(
        "possible_exposure_clipping",
        regex=False
    )
].copy()


roi_df = review_df[
    (
        review_df[
            "review_reason"
        ].str.contains(
            "extreme_low_roi_area",
            regex=False
        )
    )
    |
    (
        review_df[
            "review_reason"
        ].str.contains(
            "extreme_high_roi_area",
            regex=False
        )
    )
].copy()


audit_df = review_df[
    review_df[
        "review_reason"
    ]
    ==
    "random_quality_audit"
].copy()


build_contact_sheet(
    blur_df,
    CONTACT_DIR
    /
    "01_possible_severe_blur.jpg"
)

build_contact_sheet(
    clip_df,
    CONTACT_DIR
    /
    "02_possible_exposure_clipping.jpg"
)

build_contact_sheet(
    roi_df,
    CONTACT_DIR
    /
    "03_roi_extreme_review.jpg"
)

build_contact_sheet(
    audit_df,
    CONTACT_DIR
    /
    "04_random_quality_audit.jpg"
)


# ============================================================
# 21. Summary
# ============================================================

summary = []

summary.append(
    "=" * 90
)

summary.append(
    "Step 01-D - Final Image Quality Review Summary"
)

summary.append(
    "=" * 90
)

summary.append(
    f"全数据：{len(df)}"
)

summary.append("")

summary.append(
    "人工复核候选阈值："
)

summary.append(
    f"Laplacian P03 = "
    f"{lap_threshold:.6f}"
)

summary.append(
    f"Tenengrad P05 = "
    f"{ten_threshold:.6f}"
)

summary.append(
    f"Near-white clipping P99 = "
    f"{clip_threshold:.6f}"
)

summary.append(
    f"ROI area P01 = "
    f"{roi_low_threshold:.6f}"
)

summary.append(
    f"ROI area P99 = "
    f"{roi_high_threshold:.6f}"
)

summary.append("")

summary.append(
    "-" * 90
)

summary.append(
    f"模糊双指标候选："
    f"{int(df['flag_blur'].sum())}"
)

summary.append(
    f"曝光剪切候选："
    f"{int(df['flag_clipping'].sum())}"
)

summary.append(
    f"ROI面积低端候选："
    f"{int(df['flag_roi_low'].sum())}"
)

summary.append(
    f"ROI面积高端候选："
    f"{int(df['flag_roi_high'].sum())}"
)

summary.append("")

summary.append(
    f"自动候选去重后："
    f"{len(automatic_df)}"
)

summary.append(
    f"随机普通质量审计："
    f"{len(random_audit_df)}"
)

summary.append(
    f"最终需要人工查看："
    f"{len(review_df)}"
)

summary.append("")

summary.append(
    "-" * 90
)

summary.append(
    "允许的人工排除原因："
)

summary.append(
    "1. severe_defocus_or_motion_blur"
)

summary.append(
    "2. severe_exposure_clipping"
)

summary.append(
    "3. substantial_tongue_truncation"
)

summary.append(
    "4. severe_occlusion_or_acquisition_artifact"
)

summary.append(
    "5. unreadable_or_corrupted"
)

summary.append(
    "6. other_technical_failure"
)

summary.append("")

summary.append(
    "不能作为排除原因："
)

summary.append(
    "- 厚苔、薄苔、白苔、黄苔"
)

summary.append(
    "- 裂纹、齿痕、舌色差异"
)

summary.append(
    "- 舌体形态特殊"
)

summary.append(
    "- 单纯局部高光"
)

summary.append(
    "- 后续候选区域生成效果不好"
)

summary.append(
    "- VLM评分低"
)

summary.append(
    "- 疾病类别或分类结果"
)

summary.append("")

summary.append(
    "=" * 90
)

summary.append(
    "本步骤不自动排除任何病例。"
)

summary.append(
    "请人工填写 final_qc_review_cases.csv 中的"
)

summary.append(
    "manual_decision 和 manual_exclusion_reason。"
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
# 22. 控制台
# ============================================================

print("")
print("=" * 90)

print("Step 01-D 完成")

print("=" * 90)

print(
    f"Laplacian P03："
    f"{lap_threshold:.6f}"
)

print(
    f"Tenengrad P05："
    f"{ten_threshold:.6f}"
)

print(
    f"Near-white clipping P99："
    f"{clip_threshold:.6f}"
)

print("")

print(
    f"模糊候选："
    f"{int(df['flag_blur'].sum())}"
)

print(
    f"曝光剪切候选："
    f"{int(df['flag_clipping'].sum())}"
)

print(
    f"ROI低端："
    f"{int(df['flag_roi_low'].sum())}"
)

print(
    f"ROI高端："
    f"{int(df['flag_roi_high'].sum())}"
)

print(
    f"自动候选去重后："
    f"{len(automatic_df)}"
)

print(
    f"随机审计："
    f"{len(random_audit_df)}"
)

print(
    f"最终人工复核："
    f"{len(review_df)}"
)

print("")

print(
    f"人工复核CSV："
    f"{REVIEW_CSV}"
)

print(
    f"复核图片："
    f"{REVIEW_IMAGE_DIR}"
)

print(
    f"联系图："
    f"{CONTACT_DIR}"
)

print(
    f"汇总："
    f"{SUMMARY_PATH}"
)

print("")

print(
    "当前没有删除任何原始图片。"
)

print("=" * 90)