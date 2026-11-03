from pathlib import Path
import shutil
import hashlib
import json
from datetime import datetime

import pandas as pd


# ============================================================
# 0. 路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

RAW_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_images"
)

# Step 01-D 输出
QC_REVIEW_CSV = (
    PROJECT_ROOT
    / "results"
    / "04_final_qc_review"
    / "final_qc_review_cases.csv"
)

# 最终冻结结果
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "06_final_dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_images"
)

EXCLUDED_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "excluded_qc_54"
)

FINAL_DATASET_CSV = (
    OUTPUT_DIR
    / "final_dataset.csv"
)

EXCLUDED_CSV = (
    OUTPUT_DIR
    / "excluded_54.csv"
)

ALL_STATUS_CSV = (
    OUTPUT_DIR
    / "all_931_dataset_status.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_dataset_summary.txt"
)

FREEZE_MANIFEST = (
    OUTPUT_DIR
    / "FINAL_DATASET_FREEZE_MANIFEST.json"
)


# ============================================================
# 1. 支持格式
# ============================================================

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
# 2. 文件hash
# ============================================================

def file_sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


# ============================================================
# 3. 扫描931幅原始图像
# ============================================================

raw_paths = sorted(
    [
        p
        for p in RAW_IMAGE_DIR.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]
)


print("=" * 90)
print("Step 02 - Freeze Final Dataset")
print("=" * 90)

print(
    f"原始图片数量："
    f"{len(raw_paths)}"
)


if len(raw_paths) != 931:

    raise RuntimeError(
        "\n原始图片数量不是931！"
        f"\n当前实际发现：{len(raw_paths)}"
        "\n请先检查数据目录。"
    )


# ============================================================
# 4. 读取Step 01-D
# ============================================================

if not QC_REVIEW_CSV.exists():

    raise FileNotFoundError(
        f"找不到：\n{QC_REVIEW_CSV}"
    )


qc_df = pd.read_csv(
    QC_REVIEW_CSV
)


# ============================================================
# 5. 只使用“自动QC候选”
#
# IMPORTANT:
#
# Step 01-D 中：
#
# automatic_review_candidate = True
#
# 的病例正好是54例。
#
# 不使用后面随机抽取的30例质量审计样本。
# ============================================================

if (
    "automatic_review_candidate"
    in qc_df.columns
):

    excluded_df = qc_df[
        qc_df[
            "automatic_review_candidate"
        ]
        ==
        True
    ].copy()

else:

    # 某些CSV版本没有保存这一列，
    # 则根据review_reason判断：
    #
    # 只要不是 random_quality_audit，
    # 就视为自动QC候选。
    excluded_df = qc_df[
        qc_df[
            "review_reason"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        !=
        "random_quality_audit"
    ].copy()


excluded_df = (
    excluded_df
    .drop_duplicates(
        subset=[
            "filename"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    f"自动QC排除候选："
    f"{len(excluded_df)}"
)


# ============================================================
# 6. 强制检查必须正好54例
# ============================================================

if len(excluded_df) != 54:

    print("")
    print(
        "当前提取到的自动QC候选"
        f"不是54例，而是{len(excluded_df)}例。"
    )

    print(
        "请先不要继续。"
    )

    raise RuntimeError(
        "自动QC候选数量与预期54例不一致。"
    )


# ============================================================
# 7. 建立原始文件表
# ============================================================

raw_records = []

for path in raw_paths:

    raw_records.append(
        {
            "image_id":
                path.stem,

            "filename":
                path.name,

            "absolute_path":
                str(path),

            "relative_path":
                str(
                    path.relative_to(
                        RAW_IMAGE_DIR
                    )
                ),
        }
    )


all_df = pd.DataFrame(
    raw_records
)


# ============================================================
# 8. 确保54个文件都真实存在于931中
# ============================================================

excluded_names = set(
    excluded_df[
        "filename"
    ]
    .astype(str)
)


raw_names = set(
    all_df[
        "filename"
    ]
    .astype(str)
)


missing_names = (
    excluded_names
    -
    raw_names
)


if len(missing_names) > 0:

    raise RuntimeError(
        "\n以下QC排除病例"
        "没有在raw_images中找到：\n"
        +
        "\n".join(
            sorted(
                missing_names
            )
        )
    )


# ============================================================
# 9. 标记 INCLUDE / EXCLUDE
# ============================================================

all_df[
    "dataset_status"
] = "INCLUDE"


all_df.loc[
    all_df[
        "filename"
    ]
    .isin(
        excluded_names
    ),
    "dataset_status"
] = "EXCLUDE"


included_df = (
    all_df[
        all_df[
            "dataset_status"
        ]
        ==
        "INCLUDE"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


excluded_all_df = (
    all_df[
        all_df[
            "dataset_status"
        ]
        ==
        "EXCLUDE"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# ============================================================
# 10. 检查877
# ============================================================

print(
    f"最终 INCLUDE："
    f"{len(included_df)}"
)

print(
    f"最终 EXCLUDE："
    f"{len(excluded_all_df)}"
)


if len(included_df) != 877:

    raise RuntimeError(
        "\n最终正式数据集不是877例！"
        f"\n实际：{len(included_df)}"
    )


if len(excluded_all_df) != 54:

    raise RuntimeError(
        "\n排除集不是54例！"
        f"\n实际：{len(excluded_all_df)}"
    )


# ============================================================
# 11. 把QC原因合并到excluded表
# ============================================================

reason_columns = [
    "filename",
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


available_reason_columns = [
    c
    for c in reason_columns
    if c in excluded_df.columns
]


excluded_all_df = (
    excluded_all_df
    .merge(
        excluded_df[
            available_reason_columns
        ],
        on="filename",
        how="left"
    )
)


# ============================================================
# 12. 计算SHA256
#
# 最终冻结后，每幅图像都有hash，
# 后面可以验证数据没有被偷偷替换。
# ============================================================

print("")
print(
    "正在计算最终877幅图像SHA256..."
)


included_hashes = []

for index, row in included_df.iterrows():

    path = Path(
        row[
            "absolute_path"
        ]
    )

    included_hashes.append(
        file_sha256(
            path
        )
    )

    if (
        (index + 1) % 100
        ==
        0
    ):

        print(
            f"  {index + 1}/"
            f"{len(included_df)}"
        )


included_df[
    "sha256"
] = included_hashes


# ============================================================
# 13. 清空之前生成的final_images
#
# 注意：
# 这里只删除final_images副本。
# raw_images绝对不会动。
# ============================================================

if FINAL_IMAGE_DIR.exists():

    shutil.rmtree(
        FINAL_IMAGE_DIR
    )


FINAL_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if EXCLUDED_IMAGE_DIR.exists():

    shutil.rmtree(
        EXCLUDED_IMAGE_DIR
    )


EXCLUDED_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 14. 复制877幅正式图像
# ============================================================

print("")
print(
    "复制877幅正式分析图像..."
)


for index, row in included_df.iterrows():

    src = Path(
        row[
            "absolute_path"
        ]
    )

    dst = (
        FINAL_IMAGE_DIR
        /
        row[
            "filename"
        ]
    )

    shutil.copy2(
        src,
        dst
    )

    if (
        (index + 1) % 100
        ==
        0
    ):

        print(
            f"  {index + 1}/"
            f"{len(included_df)}"
        )


# ============================================================
# 15. 复制54幅排除病例
#
# 方便以后审计。
# ============================================================

print("")
print(
    "复制54幅QC排除病例..."
)


for _, row in excluded_all_df.iterrows():

    src = Path(
        row[
            "absolute_path"
        ]
    )

    dst = (
        EXCLUDED_IMAGE_DIR
        /
        row[
            "filename"
        ]
    )

    shutil.copy2(
        src,
        dst
    )


# ============================================================
# 16. 修改final_dataset路径
#
# 后续所有实验只读取final_path。
# ============================================================

included_df[
    "final_path"
] = included_df[
    "filename"
].map(
    lambda name:
    str(
        FINAL_IMAGE_DIR
        /
        name
    )
)


# ============================================================
# 17. 保存CSV
# ============================================================

included_df.to_csv(
    FINAL_DATASET_CSV,
    index=False,
    encoding="utf-8-sig"
)


excluded_all_df.to_csv(
    EXCLUDED_CSV,
    index=False,
    encoding="utf-8-sig"
)


all_df.to_csv(
    ALL_STATUS_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. Manifest
# ============================================================

manifest = {

    "freeze_time":
        datetime.now().isoformat(),

    "project_root":
        str(PROJECT_ROOT),

    "raw_dataset_n":
        931,

    "excluded_qc_n":
        54,

    "final_dataset_n":
        877,

    "raw_image_dir":
        str(RAW_IMAGE_DIR),

    "final_image_dir":
        str(FINAL_IMAGE_DIR),

    "excluded_image_dir":
        str(EXCLUDED_IMAGE_DIR),

    "final_dataset_csv":
        str(FINAL_DATASET_CSV),

    "excluded_csv":
        str(EXCLUDED_CSV),

    "important":
        (
            "All downstream experiments "
            "must use final_dataset.csv "
            "or data/final_images only."
        )
}


with open(
    FREEZE_MANIFEST,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 19. Summary
# ============================================================

reason_counts = (
    excluded_all_df[
        "review_reason"
    ]
    .fillna(
        "NA"
    )
    .value_counts()
)


summary_lines = [

    "=" * 90,

    "FINAL DATASET FREEZE SUMMARY",

    "=" * 90,

    f"原始图像：931",

    f"QC排除：54",

    f"最终正式分析集：877",

    "",

    f"正式图像目录：",
    str(
        FINAL_IMAGE_DIR
    ),

    "",

    f"最终数据清单：",
    str(
        FINAL_DATASET_CSV
    ),

    "",

    f"排除病例清单：",
    str(
        EXCLUDED_CSV
    ),

    "",

    "54例QC标记原因分布：",

    reason_counts.to_string(),

    "",

    "-" * 90,

    "重要：",

    "1. raw_images 未被删除或修改。",

    "2. 后续正式实验只允许使用877幅final_images。",

    "3. 不允许再根据后续算法结果增删病例。",

    "4. final_dataset.csv 为后续所有正式实验的唯一病例清单。",

    "=" * 90,
]


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
# 20. 完成
# ============================================================

print("")
print("=" * 90)

print("正式数据集冻结完成")

print("=" * 90)

print(
    "原始数据：931"
)

print(
    "QC排除：54"
)

print(
    "最终分析集：877"
)

print("")

print(
    f"Final images："
    f"{FINAL_IMAGE_DIR}"
)

print(
    f"Final dataset CSV："
    f"{FINAL_DATASET_CSV}"
)

print(
    f"Excluded CSV："
    f"{EXCLUDED_CSV}"
)

print(
    f"Summary："
    f"{SUMMARY_PATH}"
)

print("")
print(
    "后续所有正式实验只读取877幅正式数据。"
)

print("=" * 90)