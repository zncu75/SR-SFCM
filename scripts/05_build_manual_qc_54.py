from pathlib import Path
import shutil
import pandas as pd


# ============================================================
# 0. 路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "04_final_qc_review"
    / "final_qc_review_cases.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "05_manual_qc_54"
)

BLIND_IMAGE_DIR = (
    OUTPUT_DIR
    / "blind_review_images"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BLIND_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 给你人工填写
REVIEW_CSV = (
    OUTPUT_DIR
    / "manual_qc_54_FOR_REVIEW.csv"
)

# 内部映射，不要在审核过程中看
INTERNAL_MAPPING_CSV = (
    OUTPUT_DIR
    / "manual_qc_54_INTERNAL_mapping.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "manual_qc_54_summary.txt"
)

GUIDELINE_PATH = (
    OUTPUT_DIR
    / "manual_qc_review_guideline.txt"
)


# ============================================================
# 1. 固定随机种子
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 2. 读取 Step 01-D
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"找不到：\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


print("=" * 90)
print("Step 01-E - Build Manual QC Review Set")
print("=" * 90)

print(
    f"读取 Step 01-D 病例："
    f"{len(df)}"
)


# ============================================================
# 3. 只保留：
#
# A. 模糊候选
# B. 随机普通质量审计
#
# 不把曝光、ROI极端作为正式人工排除候选。
# ============================================================

reason = (
    df["review_reason"]
    .fillna("")
    .astype(str)
)


blur_mask = (
    reason.str.contains(
        "possible_severe_blur",
        regex=False
    )
)

audit_mask = (
    reason
    ==
    "random_quality_audit"
)


manual_df = df[
    blur_mask
    |
    audit_mask
].copy()


# ============================================================
# 4. 去重
# ============================================================

manual_df = manual_df.drop_duplicates(
    subset=[
        "filename"
    ]
).reset_index(
    drop=True
)


blur_count = int(
    manual_df[
        "review_reason"
    ]
    .astype(str)
    .str.contains(
        "possible_severe_blur",
        regex=False
    )
    .sum()
)

audit_count = int(
    (
        manual_df[
            "review_reason"
        ]
        ==
        "random_quality_audit"
    )
    .sum()
)


print(
    f"模糊候选：{blur_count}"
)

print(
    f"随机审计：{audit_count}"
)

print(
    f"合计：{len(manual_df)}"
)


# ============================================================
# 5. 随机打乱
#
# 非常重要：
# 不让人工审核者知道哪些是自动模糊候选，
# 哪些是随机普通病例。
# ============================================================

manual_df = manual_df.sample(
    frac=1.0,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


# ============================================================
# 6. 新盲化编号
# ============================================================

manual_df[
    "blind_id"
] = [
    f"QC{i:03d}"
    for i in range(
        1,
        len(manual_df) + 1
    )
]


# ============================================================
# 7. 清空旧盲评图片
#
# 只删除本程序以前复制出来的副本，
# 不影响 raw_images。
# ============================================================

if BLIND_IMAGE_DIR.exists():

    for item in BLIND_IMAGE_DIR.iterdir():

        if item.is_file():

            item.unlink()


# ============================================================
# 8. 复制原图并只保留 Blind ID
#
# 不在文件名中保留原病例ID。
# ============================================================

blind_file_names = []


for _, row in manual_df.iterrows():

    src = Path(
        row[
            "absolute_path"
        ]
    )

    if not src.exists():

        raise FileNotFoundError(
            f"原图不存在：\n{src}"
        )

    suffix = (
        src.suffix.lower()
    )

    if not suffix:

        suffix = ".png"

    blind_name = (
        f"{row['blind_id']}"
        f"{suffix}"
    )

    dst = (
        BLIND_IMAGE_DIR
        /
        blind_name
    )

    shutil.copy2(
        src,
        dst
    )

    blind_file_names.append(
        blind_name
    )


manual_df[
    "blind_filename"
] = blind_file_names


# ============================================================
# 9. 内部映射文件
#
# 这个文件保存：
# Blind ID -> 原始病例 -> 自动筛选原因
#
# 审核时不要看这个文件。
# ============================================================

internal_columns = [

    "blind_id",
    "blind_filename",

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
]


existing_internal_columns = [
    c
    for c in internal_columns
    if c in manual_df.columns
]


manual_df[
    existing_internal_columns
].to_csv(
    INTERNAL_MAPPING_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 给人工审核者填写的CSV
#
# 故意不提供：
# - 原文件名
# - Laplacian
# - Tenengrad
# - 自动筛选原因
#
# 避免评分偏倚。
# ============================================================

review_df = pd.DataFrame({

    "blind_id":
        manual_df[
            "blind_id"
        ],

    "image_file":
        manual_df[
            "blind_filename"
        ],

    "manual_decision":
        "",

    "manual_exclusion_reason":
        "",

    "manual_note":
        "",
})


review_df.to_csv(
    REVIEW_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 人工审核规则
# ============================================================

guideline = """
人工图像质量审核说明
============================================================

审核对象：
仅评价原始舌象的技术成像质量。

审核时禁止参考：
- SLIC / Spatial FCM 候选区域结果
- VLM评分
- CRC/NCRC标签
- 后续模型性能
- 自动QC指标
- 自动筛选原因


一、manual_decision

只允许填写：

KEEP
EXCLUDE


二、什么时候填写 EXCLUDE

只有存在明确技术质量缺陷，
并已经明显影响舌象信息辨识时才排除。


允许的排除原因：

1. severe_defocus_or_motion_blur

明显严重失焦或运动模糊，
导致舌面纹理、舌苔结构或主要边界难以可靠辨认。


2. severe_exposure_clipping

严重曝光异常，
导致大面积舌面颜色或纹理信息丢失。


3. substantial_tongue_truncation

舌体存在明显的大面积裁切或缺失，
影响舌体整体信息。


4. severe_occlusion_or_acquisition_artifact

存在严重遮挡、异物或明显采集伪影，
严重影响舌象分析。


5. unreadable_or_corrupted

图片损坏或无法正常读取。


6. other_technical_failure

其他明确的技术性采集失败。
必须在 manual_note 中说明。


三、以下情况不能作为排除理由

- 白苔
- 黄苔
- 厚苔
- 薄苔
- 少苔

- 舌色偏红或偏淡
- 裂纹
- 齿痕
- 舌体形态特殊

- 舌面湿润
- 局部反光
- 局部高光

- 舌体大小不同
- 伸舌角度不同
- 构图存在正常差异

- 后续算法分割效果差
- VLM评价低
- 疾病标签不同


四、推荐判断原则

如果只是：

“图片稍微模糊，但仍然可以清楚辨认主要舌面结构”

→ KEEP


只有达到：

“技术质量问题已经明显影响舌面颜色、
纹理、舌苔或边界的可靠判断”

→ EXCLUDE


五、审核要求

请按 QC001、QC002 ... 的顺序独立审核。

不要尝试根据病例特点猜测它为什么进入QC集合。

============================================================
"""


with open(
    GUIDELINE_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        guideline.strip()
    )


# ============================================================
# 12. Summary
# ============================================================

summary = f"""
==========================================================================================
Step 01-E - Manual QC Review Set Summary
==========================================================================================

输入病例数：
{len(df)}

模糊候选：
{blur_count}

随机普通质量审计：
{audit_count}

最终人工QC病例：
{len(manual_df)}

随机种子：
{RANDOM_SEED}


人工填写文件：
{REVIEW_CSV}


盲化原图目录：
{BLIND_IMAGE_DIR}


内部映射文件：
{INTERNAL_MAPPING_CSV}


人工审核说明：
{GUIDELINE_PATH}


重要：

1. 人工审核过程中不要打开 INTERNAL_mapping.csv。

2. 只根据 blind_review_images 中的原始舌象判断技术质量。

3. manual_decision 只能填写：
   KEEP
   EXCLUDE

4. EXCLUDE 必须填写真实技术排除原因。

5. 不需要为了达到某个预定样本量而排除病例。

==========================================================================================
"""


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary.strip()
    )


# ============================================================
# 13. 完成
# ============================================================

print("")
print("=" * 90)
print("Step 01-E 完成")
print("=" * 90)

print(
    f"最终人工QC病例："
    f"{len(manual_df)}"
)

print("")

print(
    "你需要填写："
)

print(
    REVIEW_CSV
)

print("")

print(
    "审核图片："
)

print(
    BLIND_IMAGE_DIR
)

print("")

print(
    "审核说明："
)

print(
    GUIDELINE_PATH
)

print("")

print(
    "审核过程中不要打开："
)

print(
    INTERNAL_MAPPING_CSV
)

print("=" * 90)
