from pathlib import Path
import hashlib
import json
import random
import shutil
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 0. 项目路径
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

# Step 11：冻结的88例
BLIND_88_MAPPING = (
    PROJECT_ROOT
    / "results"
    / "11_blind_validation_88"
    / "blind_validation_88_INTERNAL_mapping.csv"
)

# 877幅正式原始舌象
FINAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_images"
)

# Step 08B：Original-D / SR-D 掩膜
GAIN_ABLATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "08B_original_vs_sr_gain_ablation"
)

ORIGINAL_D_MASK_DIR = (
    GAIN_ABLATION_DIR
    / "masks"
    / "original_coating"
)

SR_D_MASK_DIR = (
    GAIN_ABLATION_DIR
    / "masks"
    / "sr_coating_resized"
)

# Step 09：正式Original-D掩膜
# 用于确认08B中的Original-D与此前正式D完全一致，
# 以保证既往人工D评分可以继续作为正式Original-D评分。
FORMAL_D_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_candidate_regions"
    / "candidate_coating_masks"
)

# Step 14：第一次A/B/C/D人工盲评内部映射
# 这里只用于建立“既往D评分”的内部对应关系，不提供给评审者。
PRIOR_HUMAN_INTERNAL_MAPPING = (
    PROJECT_ROOT
    / "results"
    / "14_human_blind_review_88"
    / "human_review_INTERNAL_mapping.csv"
)

# 本步骤输出
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "18_sr_gain_human_blind_review_88"
)

PUBLIC_DIR = (
    OUTPUT_DIR
    / "PUBLIC_REVIEW_PACKAGE"
)

INTERNAL_DIR = (
    OUTPUT_DIR
    / "INTERNAL_DO_NOT_SHARE"
)

REVIEW_IMAGE_ROOT = (
    PUBLIC_DIR
    / "review_images"
)

PUBLIC_MANIFEST_CSV = (
    PUBLIC_DIR
    / "sr_gain_human_review_public_manifest.csv"
)

SCORING_TEMPLATE_CSV = (
    PUBLIC_DIR
    / "sr_gain_human_review_scoring_template.csv"
)

SCORING_XLSX = (
    PUBLIC_DIR
    / "sr_gain_human_review_scoring_workbook.xlsx"
)

GUIDELINE_TXT = (
    PUBLIC_DIR
    / "sr_gain_human_review_guideline.txt"
)

INTERNAL_MAPPING_CSV = (
    INTERNAL_DIR
    / "sr_gain_human_review_INTERNAL_mapping.csv"
)

ANCHOR_SELECTION_CSV = (
    INTERNAL_DIR
    / "anchor_selection_INTERNAL.csv"
)

PRIOR_D_REFERENCE_CSV = (
    INTERNAL_DIR
    / "prior_D_reference_INTERNAL.csv"
)

CONFIG_JSON = (
    INTERNAL_DIR
    / "sr_gain_human_review_config.json"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "sr_gain_human_review_build_summary.txt"
)


# ============================================================
# 1. 冻结设计参数
# ============================================================

EXPECTED_CASES = 88

# 人工只新增88个SR-D正式项目，
# 再混入20个Original-D重复锚点用于评价者时间一致性检查。
ANCHOR_N = 20

EXPECTED_SR_ITEMS = 88
EXPECTED_ANCHOR_ITEMS = 20
EXPECTED_ITEMS = (
    EXPECTED_SR_ITEMS
    +
    EXPECTED_ANCHOR_ITEMS
)

N_BATCHES = 2
ITEMS_PER_BATCH = 54

# 每个Batch：
# 44个SR-D + 10个Original-D anchor
SR_PER_BATCH = 44
ANCHOR_PER_BATCH = 10

# 固定随机种子
# anchor选择完全不使用任何既往评分、结构指标或疾病标签。
ANCHOR_SELECTION_SEED = 1801

# anchor中：
# 10例的SR放Batch1，Original anchor放Batch2；
# 另10例相反。
ANCHOR_DIRECTION_SEED = 1802

# 非anchor的68个SR病例均衡分到两个Batch，各34例。
NONANCHOR_BATCH_SEED = 1803

# 两个Batch内部顺序独立打乱。
BATCH_ORDER_SEEDS = {
    1: 1811,
    2: 1812,
}

# 中性项目ID，不出现Original / SR / Real-ESRGAN字样。
ITEM_PREFIX = "HBRX"

# 与既往正式人工盲评保持完全一致的视觉参数
PANEL_SIZE = 600
HEADER_HEIGHT = 46
GAP = 14
OVERLAY_ALPHA = 0.28

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SCORE_COLUMNS = [
    "coverage_consistency",
    "boundary_alignment",
    "oversegmentation_control",
    "undersegmentation_control",
    "fragmentation_control",
    "overall_plausibility",
]

CORE_SCORE_COLUMNS = [
    "coverage_consistency",
    "boundary_alignment",
    "oversegmentation_control",
    "undersegmentation_control",
    "fragmentation_control",
]


# ============================================================
# 2. 文件夹初始化
# ============================================================

if OUTPUT_DIR.exists():
    # 为避免旧文件与新冻结结果混在一起：
    # 如果目录已经存在，不自动删除，直接终止。
    # 用户确认后可手工改名/删除旧目录再重跑。
    raise RuntimeError(
        "输出目录已经存在，为避免覆盖既往盲评材料，脚本已停止：\n"
        f"{OUTPUT_DIR}\n\n"
        "请先确认旧目录是否需要保留；如为测试残留，请手工删除或改名后再运行。"
    )

PUBLIC_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

INTERNAL_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

REVIEW_IMAGE_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

for batch in range(
    1,
    N_BATCHES + 1,
):
    (
        REVIEW_IMAGE_ROOT
        /
        f"batch_{batch:02d}"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )


# ============================================================
# 3. 工具函数
# ============================================================

def imread_unicode(path):

    try:

        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )

    except Exception:

        return None


def imwrite_png_unicode(
    path,
    image,
):

    ok, encoded = cv2.imencode(
        ".png",
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not ok:
        return False

    encoded.tofile(
        str(path)
    )

    return True


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def sha256_text(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def fit_to_square(
    image,
    size,
):

    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise RuntimeError(
            "非法图像尺寸"
        )

    scale = min(
        size / w,
        size / h,
    )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        ),
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            size,
            size,
            3,
        ),
        dtype=np.uint8,
    )

    x = (
        size - new_w
    ) // 2

    y = (
        size - new_h
    ) // 2

    canvas[
        y:
        y + new_h,
        x:
        x + new_w,
    ] = resized

    return canvas


def create_candidate_overlay(
    image,
    coating_mask,
):

    mask = (
        coating_mask > 0
    )

    overlay = image.copy()

    # 与第一次正式人工盲评保持一致：
    # 绿色半透明候选区域
    green = np.zeros_like(
        image
    )

    green[:, :, 1] = 255

    blended = cv2.addWeighted(
        image,
        1.0 - OVERLAY_ALPHA,
        green,
        OVERLAY_ALPHA,
        0,
    )

    overlay[
        mask
    ] = blended[
        mask
    ]

    # 黄色边界
    contours, _ = cv2.findContours(
        mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (
            0,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return overlay


def build_review_image(
    original,
    coating_mask,
    item_id,
):

    if (
        original.shape[:2]
        !=
        coating_mask.shape[:2]
    ):

        coating_mask = cv2.resize(
            coating_mask,
            (
                original.shape[1],
                original.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    coating_binary = (
        coating_mask[:, :, 0]
        if coating_mask.ndim == 3
        else coating_mask
    )

    coating_binary = (
        coating_binary > 0
    ).astype(
        np.uint8
    )

    overlay = create_candidate_overlay(
        original,
        coating_binary,
    )

    left = fit_to_square(
        original,
        PANEL_SIZE,
    )

    right = fit_to_square(
        overlay,
        PANEL_SIZE,
    )

    total_width = (
        PANEL_SIZE
        * 2
        +
        GAP
    )

    total_height = (
        PANEL_SIZE
        +
        HEADER_HEIGHT
    )

    canvas = np.zeros(
        (
            total_height,
            total_width,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        HEADER_HEIGHT:
        HEADER_HEIGHT + PANEL_SIZE,
        0:
        PANEL_SIZE,
    ] = left

    canvas[
        HEADER_HEIGHT:
        HEADER_HEIGHT + PANEL_SIZE,
        PANEL_SIZE + GAP:
        PANEL_SIZE + GAP + PANEL_SIZE,
    ] = right

    # 与既往评价版式保持一致：
    # 左侧统一显示原始舌象；
    # 右侧只显示候选区域，不透露Original-D或SR-D。
    cv2.putText(
        canvas,
        "Original",
        (
            12,
            31,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Candidate coating region",
        (
            PANEL_SIZE + GAP + 12,
            31,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        item_id,
        (
            total_width - 145,
            31,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return canvas


def dice_score(
    mask_a,
    mask_b,
):

    a = (
        mask_a > 0
    )

    b = (
        mask_b > 0
    )

    intersection = np.logical_and(
        a,
        b,
    ).sum()

    denominator = (
        a.sum()
        +
        b.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0
        *
        intersection
        /
        denominator
    )


# ============================================================
# 4. 读取冻结88例
# ============================================================

if not BLIND_88_MAPPING.exists():

    raise FileNotFoundError(
        "找不到冻结88例内部映射：\n"
        f"{BLIND_88_MAPPING}"
    )


blind_df = pd.read_csv(
    BLIND_88_MAPPING,
    dtype=str,
)


required_mapping_columns = {
    "blind_id",
    "image_id",
}


missing_mapping = (
    required_mapping_columns
    -
    set(
        blind_df.columns
    )
)


if missing_mapping:

    raise RuntimeError(
        "88例内部映射缺少字段："
        +
        ",".join(
            sorted(
                missing_mapping
            )
        )
    )


blind_df["blind_id"] = (
    blind_df[
        "blind_id"
    ]
    .astype(str)
)


blind_df["image_id"] = (
    blind_df[
        "image_id"
    ]
    .astype(str)
)


blind_df = (
    blind_df
    .sort_values(
        "blind_id"
    )
    .reset_index(
        drop=True
    )
)


if len(
    blind_df
) != EXPECTED_CASES:

    raise RuntimeError(
        f"冻结病例数异常："
        f"{len(blind_df)}，"
        f"预期{EXPECTED_CASES}"
    )


if blind_df[
    "image_id"
].duplicated().any():

    raise RuntimeError(
        "冻结88例image_id存在重复。"
    )


# ============================================================
# 5. 建立图像 / mask lookup
# ============================================================

def build_lookup(
    folder,
    name,
):

    if not folder.exists():

        raise FileNotFoundError(
            f"{name}目录不存在：\n"
            f"{folder}"
        )

    paths = sorted(
        [
            p
            for p
            in folder.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    lookup = {
        p.stem:
        p
        for p
        in paths
    }

    return (
        paths,
        lookup,
    )


image_paths, image_lookup = (
    build_lookup(
        FINAL_IMAGE_DIR,
        "原始舌象",
    )
)


original_mask_paths, original_mask_lookup = (
    build_lookup(
        ORIGINAL_D_MASK_DIR,
        "08B Original-D mask",
    )
)


sr_mask_paths, sr_mask_lookup = (
    build_lookup(
        SR_D_MASK_DIR,
        "08B SR-D mask",
    )
)


formal_d_paths, formal_d_lookup = (
    build_lookup(
        FORMAL_D_MASK_DIR,
        "Step09 formal-D mask",
    )
)


frozen_ids = set(
    blind_df[
        "image_id"
    ]
)


for name, lookup in [
    (
        "原始舌象",
        image_lookup,
    ),
    (
        "08B Original-D",
        original_mask_lookup,
    ),
    (
        "08B SR-D",
        sr_mask_lookup,
    ),
    (
        "Step09 formal-D",
        formal_d_lookup,
    ),
]:

    missing_ids = sorted(
        frozen_ids
        -
        set(
            lookup.keys()
        )
    )

    if missing_ids:

        raise RuntimeError(
            f"{name}缺失冻结88例中的病例：\n"
            +
            "\n".join(
                missing_ids
            )
        )


# ============================================================
# 6. 验证08B Original-D与此前正式D完全一致
#
# 这是复用既往人工D评分的前提。
# ============================================================

formal_reproduction_rows = []


for _, row in blind_df.iterrows():

    image_id = str(
        row[
            "image_id"
        ]
    )

    original_mask_img = imread_unicode(
        original_mask_lookup[
            image_id
        ]
    )

    formal_mask_img = imread_unicode(
        formal_d_lookup[
            image_id
        ]
    )

    if (
        original_mask_img is None
        or
        formal_mask_img is None
    ):

        raise RuntimeError(
            "Formal-D复现检查mask读取失败："
            f"{image_id}"
        )

    original_mask = (
        original_mask_img[
            :,
            :,
            0
        ]
        >
        0
    ).astype(
        np.uint8
    )

    formal_mask = (
        formal_mask_img[
            :,
            :,
            0
        ]
        >
        0
    ).astype(
        np.uint8
    )

    if (
        original_mask.shape
        !=
        formal_mask.shape
    ):

        formal_mask = cv2.resize(
            formal_mask,
            (
                original_mask.shape[1],
                original_mask.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    d = dice_score(
        original_mask,
        formal_mask,
    )

    formal_reproduction_rows.append(
        {
            "image_id":
                image_id,
            "original08B_vs_formalD_dice":
                d,
        }
    )


formal_reproduction_df = pd.DataFrame(
    formal_reproduction_rows
)


formal_repro_min = float(
    formal_reproduction_df[
        "original08B_vs_formalD_dice"
    ].min()
)


formal_reproduction_df.to_csv(
    INTERNAL_DIR
    /
    "original08B_vs_formalD_reproduction_INTERNAL.csv",
    index=False,
    encoding="utf-8-sig",
)


if formal_repro_min < 0.999999:

    raise RuntimeError(
        "08B Original-D与此前正式D并非完全一致，"
        "不能直接复用第一次人工D评分。\n"
        f"冻结88例最小Dice={formal_repro_min:.9f}"
    )


# ============================================================
# 7. 读取第一次人工盲评内部映射，
#    建立每例既往D项目对应关系
# ============================================================

if not PRIOR_HUMAN_INTERNAL_MAPPING.exists():

    raise FileNotFoundError(
        "找不到第一次人工盲评内部映射：\n"
        f"{PRIOR_HUMAN_INTERNAL_MAPPING}"
    )


prior_map_df = pd.read_csv(
    PRIOR_HUMAN_INTERNAL_MAPPING,
    dtype=str,
)


required_prior_columns = {
    "item_id",
    "batch",
    "batch_order",
    "image_id",
    "method",
}


missing_prior_columns = (
    required_prior_columns
    -
    set(
        prior_map_df.columns
    )
)


if missing_prior_columns:

    raise RuntimeError(
        "第一次人工内部映射缺少字段："
        +
        ",".join(
            sorted(
                missing_prior_columns
            )
        )
    )


prior_map_df["image_id"] = (
    prior_map_df[
        "image_id"
    ]
    .astype(str)
)


prior_d_df = (
    prior_map_df[
        prior_map_df[
            "method"
        ]
        .astype(str)
        ==
        "D"
    ]
    .copy()
)


if (
    len(
        prior_d_df
    )
    !=
    EXPECTED_CASES
):

    raise RuntimeError(
        "第一次人工盲评中的D项目数量异常："
        f"{len(prior_d_df)}"
    )


if prior_d_df[
    "image_id"
].duplicated().any():

    raise RuntimeError(
        "第一次人工D项目image_id存在重复。"
    )


prior_d_lookup = {
    str(r.image_id):
    {
        "prior_D_item_id":
            str(r.item_id),
        "prior_D_batch":
            int(r.batch),
        "prior_D_batch_order":
            int(r.batch_order),
    }
    for r
    in prior_d_df[
        [
            "image_id",
            "item_id",
            "batch",
            "batch_order",
        ]
    ].itertuples(
        index=False
    )
}


missing_prior_d = sorted(
    frozen_ids
    -
    set(
        prior_d_lookup.keys()
    )
)


if missing_prior_d:

    raise RuntimeError(
        "冻结88例中存在病例没有第一次人工D项目：\n"
        +
        "\n".join(
            missing_prior_d
        )
    )


prior_d_reference_rows = []


for _, row in blind_df.iterrows():

    image_id = str(
        row[
            "image_id"
        ]
    )

    info = prior_d_lookup[
        image_id
    ]

    prior_d_reference_rows.append(
        {
            "blind_id":
                str(
                    row[
                        "blind_id"
                    ]
                ),
            "image_id":
                image_id,
            **info,
        }
    )


prior_d_reference_df = pd.DataFrame(
    prior_d_reference_rows
)


prior_d_reference_df.to_csv(
    PRIOR_D_REFERENCE_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. 随机冻结20例Original-D重复锚点
#
# 不使用：
# - 第一次评分
# - PASS/REVIEW/FAIL
# - 结构指标
# - 疾病标签
# - SR结果
# ============================================================

all_image_ids = list(
    blind_df[
        "image_id"
    ]
)


rng_anchor = random.Random(
    ANCHOR_SELECTION_SEED
)


anchor_ids = sorted(
    rng_anchor.sample(
        all_image_ids,
        ANCHOR_N,
    )
)


anchor_set = set(
    anchor_ids
)


# anchor中10例SR进Batch1，10例SR进Batch2
rng_direction = random.Random(
    ANCHOR_DIRECTION_SEED
)


anchor_shuffled = anchor_ids.copy()


rng_direction.shuffle(
    anchor_shuffled
)


anchor_sr_batch1 = set(
    anchor_shuffled[
        :ANCHOR_PER_BATCH
    ]
)


anchor_sr_batch2 = set(
    anchor_shuffled[
        ANCHOR_PER_BATCH:
    ]
)


if (
    len(
        anchor_sr_batch1
    )
    !=
    ANCHOR_PER_BATCH
    or
    len(
        anchor_sr_batch2
    )
    !=
    ANCHOR_PER_BATCH
):

    raise RuntimeError(
        "anchor批次拆分失败。"
    )


# 非anchor的68个SR病例：34/34分到两个Batch
nonanchor_ids = sorted(
    [
        x
        for x
        in all_image_ids
        if x not in anchor_set
    ]
)


if len(
    nonanchor_ids
) != (
    EXPECTED_CASES
    -
    ANCHOR_N
):

    raise RuntimeError(
        "非anchor病例数异常。"
    )


rng_nonanchor = random.Random(
    NONANCHOR_BATCH_SEED
)


nonanchor_shuffled = (
    nonanchor_ids.copy()
)


rng_nonanchor.shuffle(
    nonanchor_shuffled
)


nonanchor_batch1_n = (
    SR_PER_BATCH
    -
    ANCHOR_PER_BATCH
)


nonanchor_sr_batch1 = set(
    nonanchor_shuffled[
        :nonanchor_batch1_n
    ]
)


nonanchor_sr_batch2 = set(
    nonanchor_shuffled[
        nonanchor_batch1_n:
    ]
)


if (
    len(
        nonanchor_sr_batch1
    )
    !=
    nonanchor_batch1_n
    or
    len(
        nonanchor_sr_batch2
    )
    !=
    nonanchor_batch1_n
):

    raise RuntimeError(
        "非anchor SR批次拆分失败。"
    )


sr_batch_lookup = {}


for image_id in anchor_sr_batch1:
    sr_batch_lookup[
        image_id
    ] = 1


for image_id in anchor_sr_batch2:
    sr_batch_lookup[
        image_id
    ] = 2


for image_id in nonanchor_sr_batch1:
    sr_batch_lookup[
        image_id
    ] = 1


for image_id in nonanchor_sr_batch2:
    sr_batch_lookup[
        image_id
    ] = 2


if (
    len(
        sr_batch_lookup
    )
    !=
    EXPECTED_CASES
):

    raise RuntimeError(
        "SR病例批次分配数量异常。"
    )


# ============================================================
# 9. 构建108项内部设计
# ============================================================

blind_lookup = {
    str(r.image_id):
    str(r.blind_id)
    for r
    in blind_df[
        [
            "image_id",
            "blind_id",
        ]
    ].itertuples(
        index=False
    )
}


design_rows = []


# 88个正式SR-D
for image_id in sorted(
    all_image_ids
):

    batch = int(
        sr_batch_lookup[
            image_id
        ]
    )

    prior_info = prior_d_lookup[
        image_id
    ]

    design_rows.append(
        {
            "blind_id":
                blind_lookup[
                    image_id
                ],
            "image_id":
                image_id,
            "branch":
                "SR_D",
            "is_anchor":
                False,
            "batch":
                batch,
            "paired_counterpart_in_current_review":
                (
                    "YES"
                    if image_id in anchor_set
                    else
                    "NO"
                ),
            **prior_info,
        }
    )


# 20个Original-D重复anchor
# 为减少即时记忆效应，必须放在其SR-D项目的另一Batch。
for image_id in anchor_ids:

    sr_batch = int(
        sr_batch_lookup[
            image_id
        ]
    )

    anchor_batch = (
        2
        if sr_batch == 1
        else
        1
    )

    prior_info = prior_d_lookup[
        image_id
    ]

    design_rows.append(
        {
            "blind_id":
                blind_lookup[
                    image_id
                ],
            "image_id":
                image_id,
            "branch":
                "ORIGINAL_D_ANCHOR",
            "is_anchor":
                True,
            "batch":
                anchor_batch,
            "paired_counterpart_in_current_review":
                "YES",
            **prior_info,
        }
    )


design_df = pd.DataFrame(
    design_rows
)


if len(
    design_df
) != EXPECTED_ITEMS:

    raise RuntimeError(
        f"设计项目数异常："
        f"{len(design_df)}，"
        f"预期{EXPECTED_ITEMS}"
    )


# ============================================================
# 10. 批次平衡检查
# ============================================================

for batch in [
    1,
    2,
]:

    bdf = design_df[
        design_df[
            "batch"
        ]
        ==
        batch
    ]

    sr_n = int(
        (
            bdf[
                "branch"
            ]
            ==
            "SR_D"
        ).sum()
    )

    anchor_n = int(
        (
            bdf[
                "branch"
            ]
            ==
            "ORIGINAL_D_ANCHOR"
        ).sum()
    )

    if (
        len(
            bdf
        )
        !=
        ITEMS_PER_BATCH
        or
        sr_n
        !=
        SR_PER_BATCH
        or
        anchor_n
        !=
        ANCHOR_PER_BATCH
    ):

        raise RuntimeError(
            f"Batch {batch}不平衡："
            f"total={len(bdf)}, "
            f"SR={sr_n}, "
            f"anchor={anchor_n}"
        )


# 对20个anchor病例：
# SR与Original anchor必须处于不同Batch
for image_id in anchor_ids:

    pair_df = design_df[
        design_df[
            "image_id"
        ]
        ==
        image_id
    ]

    if len(
        pair_df
    ) != 2:

        raise RuntimeError(
            f"anchor病例项目数不是2："
            f"{image_id}"
        )

    batches = set(
        pair_df[
            "batch"
        ].astype(
            int
        )
    )

    if batches != {
        1,
        2,
    }:

        raise RuntimeError(
            f"anchor病例没有被分散到两个Batch："
            f"{image_id}"
        )


# ============================================================
# 11. 每个Batch独立随机排序
# ============================================================

ordered_parts = []


for batch in [
    1,
    2,
]:

    bdf = (
        design_df[
            design_df[
                "batch"
            ]
            ==
            batch
        ]
        .copy()
    )

    rng_batch = random.Random(
        BATCH_ORDER_SEEDS[
            batch
        ]
    )

    random_keys = [
        rng_batch.random()
        for _
        in range(
            len(
                bdf
            )
        )
    ]

    bdf[
        "_random_key"
    ] = random_keys

    bdf = (
        bdf
        .sort_values(
            "_random_key"
        )
        .drop(
            columns=[
                "_random_key"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    bdf[
        "batch_order"
    ] = np.arange(
        1,
        len(
            bdf
        )
        +
        1,
    )

    ordered_parts.append(
        bdf
    )


design_df = pd.concat(
    ordered_parts,
    ignore_index=True,
)


# 最终中性item_id
design_df[
    "global_order"
] = np.arange(
    1,
    EXPECTED_ITEMS + 1,
)


design_df[
    "item_id"
] = [
    f"{ITEM_PREFIX}{i:03d}"
    for i
    in design_df[
        "global_order"
    ]
]


# ============================================================
# 12. 生成108幅盲评图
# ============================================================

public_rows = []
internal_rows = []


print(
    "=" * 110
)

print(
    "Step 18 - Build SR Gain Human Blind Review"
)

print(
    "=" * 110
)

print(
    f"Frozen cases: {EXPECTED_CASES}"
)

print(
    f"Formal new SR-D items: {EXPECTED_SR_ITEMS}"
)

print(
    f"Original-D temporal anchors: {EXPECTED_ANCHOR_ITEMS}"
)

print(
    f"Total human-review items: {EXPECTED_ITEMS}"
)

print(
    "Design: 2 batches × 54 items; "
    "each batch = 44 SR-D + 10 blinded Original-D anchors."
)

print(
    "For every anchor case, SR-D and Original-D are separated across different batches."
)

print(
    "=" * 110
)


for index, row in design_df.iterrows():

    item_id = str(
        row[
            "item_id"
        ]
    )

    blind_id = str(
        row[
            "blind_id"
        ]
    )

    image_id = str(
        row[
            "image_id"
        ]
    )

    branch = str(
        row[
            "branch"
        ]
    )

    batch = int(
        row[
            "batch"
        ]
    )

    batch_order = int(
        row[
            "batch_order"
        ]
    )

    original_path = (
        image_lookup[
            image_id
        ]
    )

    if branch == "SR_D":

        mask_path = (
            sr_mask_lookup[
                image_id
            ]
        )

    elif branch == "ORIGINAL_D_ANCHOR":

        mask_path = (
            original_mask_lookup[
                image_id
            ]
        )

    else:

        raise RuntimeError(
            f"未知branch：{branch}"
        )

    original = imread_unicode(
        original_path
    )

    coating_mask = imread_unicode(
        mask_path
    )

    if original is None:

        raise RuntimeError(
            "原始图像读取失败："
            f"{original_path}"
        )

    if coating_mask is None:

        raise RuntimeError(
            "mask读取失败："
            f"{mask_path}"
        )

    review_image = build_review_image(
        original,
        coating_mask,
        item_id,
    )

    output_batch_dir = (
        REVIEW_IMAGE_ROOT
        /
        f"batch_{batch:02d}"
    )

    review_filename = (
        f"{batch_order:03d}_{item_id}.png"
    )

    review_path = (
        output_batch_dir
        /
        review_filename
    )

    ok = imwrite_png_unicode(
        review_path,
        review_image,
    )

    if not ok:

        raise RuntimeError(
            "盲评图片保存失败："
            f"{review_path}"
        )

    relative_image_path = str(
        Path(
            "review_images"
        )
        /
        f"batch_{batch:02d}"
        /
        review_filename
    )

    public_rows.append(
        {
            "item_id":
                item_id,
            "batch":
                batch,
            "batch_order":
                batch_order,
            "image_file":
                review_filename,
            "relative_image_path":
                relative_image_path,
        }
    )

    internal_rows.append(
        {
            "item_id":
                item_id,
            "batch":
                batch,
            "batch_order":
                batch_order,
            "blind_id":
                blind_id,
            "image_id":
                image_id,
            "branch":
                branch,
            "is_anchor":
                bool(
                    row[
                        "is_anchor"
                    ]
                ),
            "prior_D_item_id":
                str(
                    row[
                        "prior_D_item_id"
                    ]
                ),
            "prior_D_batch":
                int(
                    row[
                        "prior_D_batch"
                    ]
                ),
            "prior_D_batch_order":
                int(
                    row[
                        "prior_D_batch_order"
                    ]
                ),
            "original_path":
                str(
                    original_path
                ),
            "mask_path":
                str(
                    mask_path
                ),
            "review_image_path":
                str(
                    review_path
                ),
            "review_image_sha256":
                sha256_file(
                    review_path
                ),
        }
    )

    print(
        f"[{index + 1:03d}/{EXPECTED_ITEMS}] "
        f"{item_id} | "
        f"Batch={batch} "
        f"Order={batch_order:03d} "
        f"| OK"
    )


public_df = pd.DataFrame(
    public_rows
)


internal_df = pd.DataFrame(
    internal_rows
)


# ============================================================
# 13. 公共评分模板
# ============================================================

scoring_df = (
    public_df[
        [
            "item_id",
            "batch",
            "batch_order",
            "image_file",
            "relative_image_path",
        ]
    ]
    .copy()
)


for score_column in SCORE_COLUMNS:

    scoring_df[
        score_column
    ] = ""


scoring_df[
    "reviewer_comment"
] = ""


scoring_df = (
    scoring_df
    .sort_values(
        [
            "batch",
            "batch_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 14. 保存CSV
# ============================================================

public_df.to_csv(
    PUBLIC_MANIFEST_CSV,
    index=False,
    encoding="utf-8-sig",
)


internal_df.to_csv(
    INTERNAL_MAPPING_CSV,
    index=False,
    encoding="utf-8-sig",
)


scoring_df.to_csv(
    SCORING_TEMPLATE_CSV,
    index=False,
    encoding="utf-8-sig",
)


for batch in [
    1,
    2,
]:

    batch_df = (
        scoring_df[
            scoring_df[
                "batch"
            ]
            ==
            batch
        ]
        .copy()
    )

    batch_df.to_csv(
        PUBLIC_DIR
        /
        f"sr_gain_human_review_batch_{batch:02d}_scoring.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 15. 保存anchor内部清单
# ============================================================

anchor_output_rows = []


for image_id in anchor_ids:

    current_pair = (
        internal_df[
            internal_df[
                "image_id"
            ]
            ==
            image_id
        ]
        .sort_values(
            "branch"
        )
    )

    prior_info = prior_d_lookup[
        image_id
    ]

    sr_row = current_pair[
        current_pair[
            "branch"
        ]
        ==
        "SR_D"
    ].iloc[0]

    anchor_row = current_pair[
        current_pair[
            "branch"
        ]
        ==
        "ORIGINAL_D_ANCHOR"
    ].iloc[0]

    anchor_output_rows.append(
        {
            "blind_id":
                blind_lookup[
                    image_id
                ],
            "image_id":
                image_id,
            "current_sr_item_id":
                str(
                    sr_row[
                        "item_id"
                    ]
                ),
            "current_sr_batch":
                int(
                    sr_row[
                        "batch"
                    ]
                ),
            "current_anchor_item_id":
                str(
                    anchor_row[
                        "item_id"
                    ]
                ),
            "current_anchor_batch":
                int(
                    anchor_row[
                        "batch"
                    ]
                ),
            "prior_D_item_id":
                prior_info[
                    "prior_D_item_id"
                ],
            "prior_D_batch":
                prior_info[
                    "prior_D_batch"
                ],
            "prior_D_batch_order":
                prior_info[
                    "prior_D_batch_order"
                ],
        }
    )


anchor_selection_df = pd.DataFrame(
    anchor_output_rows
)


anchor_selection_df.to_csv(
    ANCHOR_SELECTION_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. Excel人工评分工作簿
#
# 使用openpyxl仅用于在用户本地生成评分工作簿。
# 评分图PNG原文件被嵌入Excel；
# 只改变Excel中的显示尺寸，不重新编码评分图。
# ============================================================

xlsx_created = False
xlsx_embedded_images = False


try:

    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import (
        Alignment,
        Border,
        Font,
        PatternFill,
        Side,
    )
    from openpyxl.worksheet.datavalidation import (
        DataValidation,
    )

    wb = Workbook()

    ws = wb.active
    ws.title = "Instructions"

    instructions = [
        [
            "SR gain blind review — scoring instructions"
        ],
        [
            ""
        ],
        [
            "Purpose",
            (
                "Evaluate visual-semantic plausibility of candidate tongue-coating regions. "
                "This is NOT expert pixel-level ground-truth segmentation accuracy."
            ),
        ],
        [
            "Blinding",
            (
                "All displayed items use neutral IDs. "
                "The reviewer must not access INTERNAL_DO_NOT_SHARE. "
                "Whether an item is generated from the original-input method or "
                "the image-enhanced method is intentionally hidden."
            ),
        ],
        [
            "Review order",
            (
                "Complete Batch 01 first, then Batch 02. "
                "Each batch contains 54 items. "
                "The two outputs belonging to any repeated anchor case are separated across the two batches."
            ),
        ],
        [
            "Important",
            (
                "Do not compare with previous scores. "
                "Do not try to identify repeated cases. "
                "Evaluate every item independently using the same 0–4 rules."
            ),
        ],
        [
            "Score scale",
            "0–4 integer score for each dimension.",
        ],
        [
            "4",
            "Highly plausible / no obvious visual-semantic problem.",
        ],
        [
            "3",
            "Generally plausible / only minor local deviation.",
        ],
        [
            "2",
            "Moderate mismatch / requires review.",
        ],
        [
            "1",
            "Clear substantial mismatch.",
        ],
        [
            "0",
            "Severe visual-semantic inconsistency / essentially unacceptable.",
        ],
        [
            ""
        ],
        [
            "coverage_consistency",
            (
                "How well the candidate coating region covers the visually apparent "
                "coating distribution, considering the whole tongue."
            ),
        ],
        [
            "boundary_alignment",
            (
                "How well the candidate boundary follows visually plausible transitions "
                "between coating-like and body-like appearance."
            ),
        ],
        [
            "oversegmentation_control",
            (
                "Whether obvious tongue-body or non-coating areas are avoided. "
                "Higher score = less visually apparent over-segmentation."
            ),
        ],
        [
            "undersegmentation_control",
            (
                "Whether visually apparent coating areas are retained. "
                "Higher score = less visually apparent under-segmentation."
            ),
        ],
        [
            "fragmentation_control",
            (
                "Whether the candidate coating region avoids implausible fragmentation, "
                "isolated islands and excessive local breaks."
            ),
        ],
        [
            "overall_plausibility",
            (
                "Overall visual-semantic plausibility of the candidate tongue-body / "
                "tongue-coating separation."
            ),
        ],
        [
            "Decision rule",
            (
                "PASS / REVIEW / FAIL will be derived automatically after scoring; "
                "the reviewer does not assign it manually."
            ),
        ],
    ]

    for values in instructions:

        ws.append(
            values
        )

    ws["A1"].font = Font(
        bold=True,
        size=14,
    )

    ws["A1"].fill = PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )

    ws.column_dimensions[
        "A"
    ].width = 28

    ws.column_dimensions[
        "B"
    ].width = 110

    for row_cells in ws.iter_rows():

        for cell in row_cells:

            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    thin_gray = Side(
        style="thin",
        color="D9D9D9",
    )

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="EAF2F8",
    )

    # review_image显示尺寸
    display_width_px = 760
    display_height_px = 404

    # Excel行高以points计，近似 px * 0.75
    review_row_height = 310

    for batch in [
        1,
        2,
    ]:

        bws = wb.create_sheet(
            f"Batch_{batch:02d}"
        )

        batch_df = (
            scoring_df[
                scoring_df[
                    "batch"
                ]
                ==
                batch
            ]
            .sort_values(
                "batch_order"
            )
            .reset_index(
                drop=True
            )
        )

        headers = [
            "item_id",
            "order",
            "review_image",
            "coverage_consistency",
            "boundary_alignment",
            "oversegmentation_control",
            "undersegmentation_control",
            "fragmentation_control",
            "overall_plausibility",
            "reviewer_comment",
        ]

        bws.append(
            headers
        )

        for _, r in batch_df.iterrows():

            review_path = (
                PUBLIC_DIR
                /
                str(
                    r[
                        "relative_image_path"
                    ]
                )
            )

            bws.append(
                [
                    str(
                        r[
                            "item_id"
                        ]
                    ),
                    int(
                        r[
                            "batch_order"
                        ]
                    ),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

            current_row = (
                bws.max_row
            )

            bws.row_dimensions[
                current_row
            ].height = review_row_height

            # 嵌入正式PNG，不重新编码
            xl_img = XLImage(
                str(
                    review_path
                )
            )

            xl_img.width = (
                display_width_px
            )

            xl_img.height = (
                display_height_px
            )

            xl_img.anchor = (
                f"C{current_row}"
            )

            bws.add_image(
                xl_img
            )

        bws.freeze_panes = "A2"

        bws.auto_filter.ref = (
            f"A1:J{bws.max_row}"
        )

        dv = DataValidation(
            type="whole",
            operator="between",
            formula1="0",
            formula2="4",
            allow_blank=True,
        )

        dv.error = (
            "评分必须为0、1、2、3或4。"
        )

        dv.errorTitle = (
            "无效评分"
        )

        bws.add_data_validation(
            dv
        )

        dv.add(
            f"D2:I{bws.max_row}"
        )

        for cell in bws[
            1
        ]:

            cell.font = Font(
                bold=True
            )

            cell.fill = header_fill

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

            cell.border = Border(
                bottom=thin_gray
            )

        widths = {
            "A": 14,
            "B": 9,
            "C": 108,
            "D": 22,
            "E": 20,
            "F": 24,
            "G": 25,
            "H": 23,
            "I": 22,
            "J": 42,
        }

        for col, width in widths.items():

            bws.column_dimensions[
                col
            ].width = width

        for row_cells in bws.iter_rows(
            min_row=2
        ):

            for cell in row_cells:

                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=True,
                )

        bws.sheet_view.showGridLines = False

    wb.save(
        SCORING_XLSX
    )

    xlsx_created = True
    xlsx_embedded_images = True

except ImportError:

    print()
    print(
        "WARNING: 当前环境未安装openpyxl，"
        "未生成XLSX；CSV评分模板和PNG盲评图已经完整生成。"
    )


# ============================================================
# 17. 评分说明TXT
# ============================================================

guideline = f"""
==============================================================================================================
SR Gain Human Blind Review Guideline
==============================================================================================================

研究目的
--------------------------------------------------------------------------------------------------------------
本轮评价用于检验图像增强分支所生成候选舌苔区域的视觉语义合理性，
并通过少量重复Original-D锚点评估同一人工评价者在不同时间点的评分稳定性。

本轮不是：
- 专家像素级Ground Truth评价；
- 分割准确率评价；
- 疾病诊断评价；
- 对Real-ESRGAN或Original/SR身份的判断。

==============================================================================================================
盲化设计
==============================================================================================================

冻结病例：
88例，完全沿用此前已经冻结的视觉评价病例，不重新抽样。

本轮正式新增评价：
88个SR-D候选区域项目。

重复一致性锚点：
从同一冻结88例中，按预定义随机种子独立抽取20例既往Original-D结果，
混入本轮作为隐藏重复锚点。

总人工评价项目：
88 + 20 = 108项。

分批：
2个Batch，每个Batch 54项。
每个Batch均包含：
- 44个SR-D项目；
- 10个隐藏Original-D重复锚点。

对于20个锚点病例：
其SR-D项目与Original-D重复项目被强制分在不同Batch，
降低短期记忆和直接并排比较的影响。

重要：
评审者不知道：
- 哪些项目来自SR-D；
- 哪些项目是Original-D重复锚点；
- 哪些病例曾经评过；
- 原始病例ID；
- 疾病标签；
- 结构指标；
- 既往评分；
- VLM评分。

严禁向人工评审者提供：
{INTERNAL_DIR}

==============================================================================================================
评价图
==============================================================================================================

每个项目：
左侧 = 原始舌象；
右侧 = 候选舌苔区域可视化。

所有项目左侧均统一显示原始舌象，
即使右侧候选区域由SR-D生成，也不显示SR图像本身，
从而避免单纯图像清晰度对人工评分造成提示。

绿色半透明区域 = 候选舌苔区域；
黄色边界 = 候选区域边界。

==============================================================================================================
0–4评分尺度
==============================================================================================================

4：
高度合理，无明显视觉语义问题。

3：
总体合理，仅存在轻微局部偏差。

2：
存在中等程度偏差，需要进一步复核。

1：
存在明显且较大的视觉语义偏差。

0：
严重不合理，候选区域与视觉舌象表现明显不符。

==============================================================================================================
六个评分维度
==============================================================================================================

1. coverage_consistency
候选舌苔区域对视觉上可见舌苔分布的总体覆盖一致性。

2. boundary_alignment
候选区域边界与视觉上舌苔样/舌质样外观转换位置的合理一致程度。

3. oversegmentation_control
是否避免将明显舌质样区域错误纳入候选舌苔区域。
分数越高 = 视觉上明显过分割越少。

4. undersegmentation_control
是否保留视觉上明显呈舌苔样的区域。
分数越高 = 视觉上明显欠分割越少。

5. fragmentation_control
是否避免不合理的碎片、小岛、断裂或过度局部化。
分数越高 = 候选区域空间结构越合理。

6. overall_plausibility
对候选舌质—舌苔区域划分整体视觉语义合理性的综合判断。

==============================================================================================================
派生决策规则
==============================================================================================================

PASS：
5个核心维度全部 >= 3
且 overall_plausibility >= 3

FAIL：
任一核心维度 = 0
或至少2个核心维度 <= 1
或 overall_plausibility <= 1

REVIEW：
其余情况。

人工评审者不需要填写PASS/REVIEW/FAIL。

==============================================================================================================
正式评审要求
==============================================================================================================

1. 按 Batch 01 → Batch 02 顺序完成。
2. 每个项目独立评分，不主动寻找重复病例。
3. 不查看或回忆既往评分。
4. Batch 01完成后建议休息，再进行Batch 02。
5. 本轮人工盲评应独立于后续VLM结果完成。
6. 评审者只需填写6个0–4整数评分；comment可选。
7. 本轮20个Original-D锚点的二次评分不会替代第一次正式Original-D评分；
   仅用于评价时间一致性/评分漂移。
8. 正式Original-D vs SR-D比较时，Original-D仍使用第一次冻结的正式人工评分，
   SR-D使用本轮88个SR-D项目评分。

==============================================================================================================
""".strip()


with open(
    GUIDELINE_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        guideline
    )


# ============================================================
# 18. 冻结映射SHA256
# ============================================================

freeze_text = "\n".join(
    [
        (
            f"{r.item_id}|"
            f"{r.batch}|"
            f"{r.batch_order}|"
            f"{r.blind_id}|"
            f"{r.image_id}|"
            f"{r.branch}|"
            f"{int(bool(r.is_anchor))}"
        )
        for r
        in internal_df[
            [
                "item_id",
                "batch",
                "batch_order",
                "blind_id",
                "image_id",
                "branch",
                "is_anchor",
            ]
        ].itertuples(
            index=False
        )
    ]
)


blinding_sha256 = sha256_text(
    freeze_text
)


# ============================================================
# 19. 配置JSON
# ============================================================

config = {
    "created_at":
        datetime.now().isoformat(),

    "frozen_cases":
        EXPECTED_CASES,

    "formal_new_sr_items":
        EXPECTED_SR_ITEMS,

    "temporal_original_D_anchor_items":
        EXPECTED_ANCHOR_ITEMS,

    "total_human_review_items":
        EXPECTED_ITEMS,

    "batches":
        N_BATCHES,

    "items_per_batch":
        ITEMS_PER_BATCH,

    "sr_items_per_batch":
        SR_PER_BATCH,

    "anchor_items_per_batch":
        ANCHOR_PER_BATCH,

    "anchor_selection_seed":
        ANCHOR_SELECTION_SEED,

    "anchor_direction_seed":
        ANCHOR_DIRECTION_SEED,

    "nonanchor_batch_seed":
        NONANCHOR_BATCH_SEED,

    "batch_order_seeds":
        BATCH_ORDER_SEEDS,

    "review_input":
        (
            "Original tongue image + blinded candidate coating overlay. "
            "The left panel is always the original tongue image for BOTH branches."
        ),

    "overlay_alpha":
        OVERLAY_ALPHA,

    "score_scale":
        "integer 0-4",

    "score_columns":
        SCORE_COLUMNS,

    "core_score_columns":
        CORE_SCORE_COLUMNS,

    "formal_original_D_score_policy":
        (
            "Reuse the first frozen human D scores from Step 14. "
            "Do NOT replace them with the 20 repeated anchor scores."
        ),

    "anchor_role":
        (
            "Temporal intra-rater consistency / score-drift assessment only."
        ),

    "formal_D_reproduction_min_dice_on_frozen88":
        formal_repro_min,

    "blinding_sha256":
        blinding_sha256,

    "xlsx_created":
        xlsx_created,

    "xlsx_embedded_images":
        xlsx_embedded_images,

    "derived_decision_rule": {
        "PASS":
            (
                "all five core scores >=3 "
                "AND overall_plausibility >=3"
            ),
        "FAIL":
            (
                "any core score ==0 "
                "OR at least two core scores <=1 "
                "OR overall_plausibility <=1"
            ),
        "REVIEW":
            "all remaining score patterns",
    },
}


with open(
    CONFIG_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# 20. 最终完整性检查
# ============================================================

# Public不允许出现内部字段
for forbidden_column in [
    "image_id",
    "blind_id",
    "branch",
    "is_anchor",
    "method",
    "prior_D_item_id",
]:

    if forbidden_column in (
        public_df.columns
    ):

        raise RuntimeError(
            "公共manifest泄露内部字段："
            f"{forbidden_column}"
        )

    if forbidden_column in (
        scoring_df.columns
    ):

        raise RuntimeError(
            "公共评分模板泄露内部字段："
            f"{forbidden_column}"
        )


# 108张图必须全部存在
expected_review_files = set(
    internal_df[
        "review_image_path"
    ]
)


missing_review_files = [
    p
    for p
    in expected_review_files
    if not Path(
        p
    ).exists()
]


if missing_review_files:

    raise RuntimeError(
        "存在盲评图未成功生成：\n"
        +
        "\n".join(
            missing_review_files
        )
    )


# 每个Batch 54，44 SR + 10 anchor
balance_rows = []


for batch in [
    1,
    2,
]:

    bdf = internal_df[
        internal_df[
            "batch"
        ]
        ==
        batch
    ]

    sr_n = int(
        (
            bdf[
                "branch"
            ]
            ==
            "SR_D"
        ).sum()
    )

    anchor_n = int(
        (
            bdf[
                "branch"
            ]
            ==
            "ORIGINAL_D_ANCHOR"
        ).sum()
    )

    balance_rows.append(
        {
            "batch":
                batch,
            "total":
                len(
                    bdf
                ),
            "SR_D":
                sr_n,
            "Original_D_anchor":
                anchor_n,
        }
    )


balance_df = pd.DataFrame(
    balance_rows
)


balance_df.to_csv(
    INTERNAL_DIR
    /
    "batch_balance_INTERNAL.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 21. Summary
# ============================================================

summary_lines = [
    "=" * 110,
    "SR Gain Human Blind Review Material Build Summary",
    "=" * 110,
    f"Frozen cases: {EXPECTED_CASES}",
    f"Formal new SR-D items: {EXPECTED_SR_ITEMS}",
    f"Temporal Original-D anchors: {EXPECTED_ANCHOR_ITEMS}",
    f"Total human-review items: {EXPECTED_ITEMS}",
    "",
    "FORMAL-D REPRODUCTION",
    "-" * 110,
    f"08B Original-D vs Step09 formal-D minimum Dice on frozen 88: {formal_repro_min:.9f}",
    "Result: PASS; prior formal human D scores can be reused.",
    "",
    "BLINDING DESIGN",
    "-" * 110,
    "Batches: 2",
    "Items per batch: 54",
    "Each batch: 44 SR-D + 10 hidden Original-D temporal anchors",
    "Every anchor case: its SR-D and Original-D anchor are placed in different batches.",
    f"Anchor-selection seed: {ANCHOR_SELECTION_SEED}",
    f"Anchor-direction seed: {ANCHOR_DIRECTION_SEED}",
    f"Nonanchor batch seed: {NONANCHOR_BATCH_SEED}",
    f"Batch-order seeds: {BATCH_ORDER_SEEDS}",
    "",
    "HUMAN-SCORING POLICY",
    "-" * 110,
    "Formal Original-D human scores: reuse the first frozen D scores from Step 14.",
    "Formal SR-D human scores: use the 88 SR-D items from the present review.",
    "The 20 repeated Original-D anchor scores are NOT replacements for the first D scores.",
    "They are used only for intra-rater temporal consistency / score-drift analysis.",
    "",
    "REVIEW IMAGE",
    "-" * 110,
    "Left: original tongue image for ALL items",
    "Right: blinded candidate tongue-coating overlay",
    "Green semi-transparent fill + yellow candidate boundary",
    "Original/SR identity: hidden",
    "Original case ID: hidden",
    "Disease label: hidden",
    "Prior scores: hidden",
    "Algorithmic metrics: hidden",
    "",
    "OUTPUTS",
    "-" * 110,
    f"PUBLIC package: {PUBLIC_DIR}",
    f"Review images: {REVIEW_IMAGE_ROOT}",
    f"Scoring CSV: {SCORING_TEMPLATE_CSV}",
    f"Scoring XLSX created: {xlsx_created}",
    f"Scoring XLSX: {SCORING_XLSX}",
    f"Guideline: {GUIDELINE_TXT}",
    f"INTERNAL package: {INTERNAL_DIR}",
    f"Internal mapping: {INTERNAL_MAPPING_CSV}",
    f"Anchor selection: {ANCHOR_SELECTION_CSV}",
    f"Prior D reference: {PRIOR_D_REFERENCE_CSV}",
    "",
    f"Blinding SHA256: {blinding_sha256}",
    "",
    "IMPORTANT",
    "-" * 110,
    "1. Give ONLY PUBLIC_REVIEW_PACKAGE to the human reviewer.",
    "2. NEVER give INTERNAL_DO_NOT_SHARE to the human reviewer.",
    "3. Do not expose previous scores or VLM outputs before human scoring is completed.",
    "4. Do not replace or resample any of the frozen 88 cases according to the new scores.",
    "5. These ratings measure visual-semantic plausibility, not pixel-level segmentation accuracy.",
    "",
    "FINAL CHECK: PASS",
    "=" * 110,
]


with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8",
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
    "=" * 110
)

print(
    "STEP 18 COMPLETE"
)

print(
    "=" * 110
)

print(
    balance_df.to_string(
        index=False
    )
)

print()

print(
    f"Formal-D reproduction min Dice: "
    f"{formal_repro_min:.9f}"
)

print(
    f"Blinding SHA256: "
    f"{blinding_sha256}"
)

print()

print(
    "给人工评审者的目录 ONLY："
)

print(
    PUBLIC_DIR
)

print()

print(
    "严禁给人工评审者："
)

print(
    INTERNAL_DIR
)

print()

print(
    f"XLSX created: "
    f"{xlsx_created}"
)

print()

print(
    "FINAL CHECK: PASS"
)

print(
    "=" * 110
)
