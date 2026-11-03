from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime

import pandas as pd


# ============================================================
# 0. 项目路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

# 877例正式冻结名单
FINAL_DATASET_CSV = (
    PROJECT_ROOT
    / "results"
    / "06_final_dataset"
    / "final_dataset.csv"
)

# 877幅正式原始舌象
FINAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_images"
)

# 输出
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "11_blind_validation_88"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 盲化图片副本
BLIND_IMAGE_DIR = (
    OUTPUT_DIR
    / "blind_original_images"
)

BLIND_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 正式盲评名单
BLIND_CSV = (
    OUTPUT_DIR
    / "blind_validation_88.csv"
)

# 内部映射表
INTERNAL_MAPPING_CSV = (
    OUTPUT_DIR
    / "blind_validation_88_INTERNAL_mapping.csv"
)

# 完整877例抽样状态
ALL_STATUS_CSV = (
    OUTPUT_DIR
    / "all_877_sampling_status.csv"
)

# 配置
CONFIG_PATH = (
    OUTPUT_DIR
    / "blind_validation_88_config.json"
)

# 摘要
SUMMARY_PATH = (
    OUTPUT_DIR
    / "blind_validation_88_summary.txt"
)


# ============================================================
# 1. 固定参数
# ============================================================

EXPECTED_TOTAL_N = 877

BLIND_N = 88

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


# ============================================================
# 2. SHA256
# ============================================================

def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 3. 检查正式数据集CSV
# ============================================================

if not FINAL_DATASET_CSV.exists():

    raise FileNotFoundError(
        f"找不到正式数据集CSV：\n"
        f"{FINAL_DATASET_CSV}"
    )


df = pd.read_csv(
    FINAL_DATASET_CSV
)


print("=" * 100)
print("Step 07 - Freeze Blind Validation Subset")
print("=" * 100)

print(
    f"读取正式数据集：{len(df)}"
)


# ============================================================
# 4. 必须严格等于877例
# ============================================================

if len(df) != EXPECTED_TOTAL_N:

    raise RuntimeError(
        f"正式数据集数量异常。\n"
        f"预期：{EXPECTED_TOTAL_N}\n"
        f"实际：{len(df)}"
    )


# ============================================================
# 5. 确定唯一病例ID
#
# 优先使用 image_id
# 如果没有，则从filename生成
# ============================================================

if "image_id" not in df.columns:

    if "filename" not in df.columns:

        raise RuntimeError(
            "final_dataset.csv 中既没有 "
            "image_id，也没有 filename。"
        )

    df["image_id"] = (
        df["filename"]
        .astype(str)
        .map(
            lambda x: Path(x).stem
        )
    )


df["image_id"] = (
    df["image_id"]
    .astype(str)
)


# ============================================================
# 6. 检查image_id唯一性
# ============================================================

duplicate_ids = (
    df[
        df["image_id"].duplicated(
            keep=False
        )
    ]["image_id"]
    .tolist()
)


if duplicate_ids:

    raise RuntimeError(
        "发现重复image_id：\n"
        +
        "\n".join(
            sorted(
                set(
                    duplicate_ids
                )
            )
        )
    )


# ============================================================
# 7. 确认原始图像路径
#
# 不依赖CSV里的旧absolute_path，
# 统一根据 final_images 重新配对。
# ============================================================

image_paths = sorted(
    [
        p
        for p in FINAL_IMAGE_DIR.iterdir()
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]
)


if len(image_paths) != EXPECTED_TOTAL_N:

    raise RuntimeError(
        f"final_images数量异常。\n"
        f"预期：{EXPECTED_TOTAL_N}\n"
        f"实际：{len(image_paths)}"
    )


image_lookup = {
    p.stem: p
    for p in image_paths
}


missing_images = sorted(
    set(
        df["image_id"]
    )
    -
    set(
        image_lookup.keys()
    )
)


if missing_images:

    raise RuntimeError(
        "以下病例在final_images中找不到：\n"
        +
        "\n".join(
            missing_images
        )
    )


# ============================================================
# 8. 建立确定性的基础排序
#
# 非常重要：
# 先按image_id排序，
# 再用固定随机种子抽样。
#
# 这样即使CSV原始行顺序变化，
# 只要877例名单不变，
# 抽出的88例仍保持一致。
# ============================================================

df = (
    df
    .sort_values(
        "image_id"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. 固定随机抽取88例
#
# 注意：
# 不使用：
# - CRC/NCRC标签
# - 候选区域面积
# - VLM结果
# - 分割效果
# - 模型预测
#
# 纯粹病例级随机抽样。
# ============================================================

selected_df = (
    df.sample(
        n=BLIND_N,
        random_state=RANDOM_SEED,
        replace=False
    )
    .copy()
)


# ============================================================
# 10. 再次随机排列88例展示顺序
#
# 与“抽到哪些病例”相同随机过程下固定。
# ============================================================

selected_df = (
    selected_df
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED + 1
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 11. 生成盲化ID
# ============================================================

selected_df["blind_id"] = [
    f"VAL{i:03d}"
    for i in range(
        1,
        BLIND_N + 1
    )
]


# ============================================================
# 12. 建立原始图像路径
# ============================================================

selected_df[
    "original_image_path"
] = selected_df[
    "image_id"
].map(
    lambda x:
    str(
        image_lookup[x]
    )
)


# ============================================================
# 13. 清空旧盲化图片副本
#
# 仅清理当前步骤产生的副本。
# 不碰final_images。
# ============================================================

for old_file in BLIND_IMAGE_DIR.iterdir():

    if old_file.is_file():

        old_file.unlink()


# ============================================================
# 14. 复制88幅原始图像
#
# 文件名只保留：
#
# VAL001.png
# VAL002.png
# ...
#
# 不泄露原始病例编号。
# ============================================================

blind_filenames = []

blind_paths = []


for _, row in selected_df.iterrows():

    image_id = row[
        "image_id"
    ]

    blind_id = row[
        "blind_id"
    ]

    src = image_lookup[
        image_id
    ]

    suffix = (
        src.suffix.lower()
    )

    if suffix not in IMAGE_EXTENSIONS:

        suffix = ".png"

    blind_filename = (
        f"{blind_id}"
        f"{suffix}"
    )

    dst = (
        BLIND_IMAGE_DIR
        /
        blind_filename
    )

    shutil.copy2(
        src,
        dst
    )

    blind_filenames.append(
        blind_filename
    )

    blind_paths.append(
        str(dst)
    )


selected_df[
    "blind_filename"
] = blind_filenames

selected_df[
    "blind_image_path"
] = blind_paths


# ============================================================
# 15. 给人工/VLM使用的正式盲评名单
#
# 故意不提供原始image_id。
# ============================================================

blind_public_df = pd.DataFrame(
    {
        "blind_id":
            selected_df[
                "blind_id"
            ],

        "image_file":
            selected_df[
                "blind_filename"
            ],

        "image_path":
            selected_df[
                "blind_image_path"
            ],
    }
)


blind_public_df.to_csv(
    BLIND_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 内部映射表
#
# 后续脚本需要：
#
# VAL001 -> 原病例ID
#
# 人工/VLM盲评过程中不要给评审者看。
# ============================================================

internal_columns = [
    "blind_id",
    "blind_filename",
    "blind_image_path",
    "image_id",
    "original_image_path",
]


# 如果原始final_dataset.csv里面还有filename等字段，
# 一并保留，方便以后恢复。
for optional_column in [
    "filename",
    "relative_path",
    "sha256",
    "final_path",
]:

    if (
        optional_column
        in selected_df.columns
        and
        optional_column
        not in internal_columns
    ):

        internal_columns.append(
            optional_column
        )


selected_df[
    internal_columns
].to_csv(
    INTERNAL_MAPPING_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 完整877例抽样状态
# ============================================================

selected_ids = set(
    selected_df[
        "image_id"
    ]
)


all_status_df = (
    df[
        [
            "image_id"
        ]
    ]
    .copy()
)


all_status_df[
    "selected_for_blind_validation"
] = (
    all_status_df[
        "image_id"
    ]
    .isin(
        selected_ids
    )
)


blind_lookup = dict(
    zip(
        selected_df[
            "image_id"
        ],
        selected_df[
            "blind_id"
        ]
    )
)


all_status_df[
    "blind_id"
] = (
    all_status_df[
        "image_id"
    ]
    .map(
        blind_lookup
    )
    .fillna("")
)


all_status_df.to_csv(
    ALL_STATUS_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 冻结哈希
#
# 对88例的：
#
# blind_id + image_id
#
# 生成SHA256。
#
# 以后可以验证名单有没有发生变化。
# ============================================================

freeze_text = "\n".join(
    [
        f"{row.blind_id}|{row.image_id}"
        for row
        in selected_df[
            [
                "blind_id",
                "image_id"
            ]
        ].itertuples(
            index=False
        )
    ]
)


selection_sha256 = (
    sha256_text(
        freeze_text
    )
)


# ============================================================
# 19. 验证数量
# ============================================================

blind_image_count = len(
    [
        p
        for p in BLIND_IMAGE_DIR.iterdir()
        if p.is_file()
    ]
)


if len(
    selected_df
) != BLIND_N:

    raise RuntimeError(
        "最终抽样病例不是88例。"
    )


if len(
    selected_ids
) != BLIND_N:

    raise RuntimeError(
        "88例中存在重复病例。"
    )


if blind_image_count != BLIND_N:

    raise RuntimeError(
        f"盲化图片数量异常："
        f"{blind_image_count}"
    )


# ============================================================
# 20. 保存配置
# ============================================================

config = {

    "created_at":
        datetime.now().isoformat(),

    "formal_dataset_n":
        EXPECTED_TOTAL_N,

    "blind_validation_n":
        BLIND_N,

    "sampling_fraction":
        BLIND_N
        /
        EXPECTED_TOTAL_N,

    "sampling_percentage":
        (
            BLIND_N
            /
            EXPECTED_TOTAL_N
            *
            100.0
        ),

    "sampling_method":
        "simple random sampling at participant/image level",

    "random_seed_selection":
        RANDOM_SEED,

    "random_seed_display_order":
        RANDOM_SEED + 1,

    "base_sort":
        "image_id ascending before random sampling",

    "uses_disease_label":
        False,

    "uses_candidate_region_result":
        False,

    "uses_model_prediction":
        False,

    "uses_vlm_score":
        False,

    "selection_sha256":
        selection_sha256,

    "important":
        (
            "The same frozen 88-case subset "
            "must be used for human review, "
            "VLM-1 review and VLM-2 review."
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
# 21. Summary
# ============================================================

sampling_percentage = (
    BLIND_N
    /
    EXPECTED_TOTAL_N
    *
    100
)


summary_lines = [

    "=" * 100,

    "Blind Validation Subset Freeze Summary",

    "=" * 100,

    f"正式完整分析集："
    f"{EXPECTED_TOTAL_N}",

    f"盲评抽样病例："
    f"{BLIND_N}",

    f"抽样比例："
    f"{sampling_percentage:.3f}%",

    "",

    "抽样方法：",
    "病例级简单随机抽样",

    f"固定随机种子："
    f"{RANDOM_SEED}",

    f"盲评展示顺序随机种子："
    f"{RANDOM_SEED + 1}",

    "",

    "抽样前处理：",
    "先按image_id升序固定877例顺序，"
    "再进行随机抽样。",

    "",

    "抽样不使用：",

    "- CRC/NCRC疾病标签",

    "- 候选舌苔面积",

    "- 候选区域视觉结果",

    "- 后续分类模型预测",

    "- VLM评分",

    "- 人工评分",

    "",

    "正式用途：",

    "1. Human blind review",

    "2. VLM-1 blind review",

    "3. VLM-2 blind review",

    "",

    "三种评价必须使用完全相同的88例。",

    "",

    f"88例冻结SHA256：",
    selection_sha256,

    "",

    f"盲评公开名单：",
    str(
        BLIND_CSV
    ),

    "",

    f"内部映射表：",
    str(
        INTERNAL_MAPPING_CSV
    ),

    "",

    f"盲化原始图像：",
    str(
        BLIND_IMAGE_DIR
    ),

    "",

    f"盲化图片数量："
    f"{blind_image_count}",

    "",

    "-" * 100,

    "IMPORTANT",

    "",

    "1. 此88例名单自本步骤起冻结。",

    "2. 后续不得根据A/B/C/D结果重新抽样。",

    "3. 后续不得根据人工或VLM评分替换病例。",

    "4. blind_validation_88_INTERNAL_mapping.csv "
    "不得提供给盲评者。",

    "5. 877例正式数据集本身没有发生任何改变。",

    "",

    "FINAL CHECK: PASS",

    "=" * 100,
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
# 22. 控制台输出
# ============================================================

print()

print(
    "=" * 100
)

print(
    "88例盲评子集冻结完成"
)

print(
    "=" * 100
)

print(
    f"正式数据集："
    f"{EXPECTED_TOTAL_N}"
)

print(
    f"随机抽样："
    f"{BLIND_N}"
)

print(
    f"抽样比例："
    f"{sampling_percentage:.3f}%"
)

print(
    f"重复病例："
    f"{BLIND_N - len(selected_ids)}"
)

print(
    f"盲化图片："
    f"{blind_image_count}"
)

print()

print(
    f"Selection SHA256："
    f"{selection_sha256}"
)

print()

print(
    f"正式盲评名单："
    f"{BLIND_CSV}"
)

print(
    f"内部映射："
    f"{INTERNAL_MAPPING_CSV}"
)

print(
    f"盲化原图："
    f"{BLIND_IMAGE_DIR}"
)

print(
    f"Summary："
    f"{SUMMARY_PATH}"
)

print()

print(
    "以后人工、VLM-1、VLM-2"
    "必须使用完全相同的这88例。"
)

print()

print(
    "FINAL CHECK: PASS"
)

print(
    "=" * 100
)