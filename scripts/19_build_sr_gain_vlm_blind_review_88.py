from pathlib import Path
import hashlib
import json
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 0. Project paths
# ============================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

BLIND_88_MAPPING = (
    PROJECT_ROOT
    / "results"
    / "11_blind_validation_88"
    / "blind_validation_88_INTERNAL_mapping.csv"
)

FINAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_images"
)

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

FORMAL_D_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "final_candidate_regions"
    / "candidate_coating_masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "19_sr_gain_vlm_blind_review_88"
)

PUBLIC_DIR = (
    OUTPUT_DIR
    / "PUBLIC_VLM_PACKAGE"
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
    / "sr_gain_vlm_public_manifest.csv"
)

INTERNAL_MAPPING_CSV = (
    INTERNAL_DIR
    / "sr_gain_vlm_INTERNAL_mapping.csv"
)

CONFIG_JSON = (
    INTERNAL_DIR
    / "sr_gain_vlm_build_config.json"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "sr_gain_vlm_build_summary.txt"
)


# ============================================================
# 1. Frozen design
# ============================================================

EXPECTED_CASES = 88
METHODS = ["ORIGINAL_D", "SR_D"]
EXPECTED_ITEMS = EXPECTED_CASES * len(METHODS)

N_BATCHES = 2
ITEMS_PER_BATCH = 88
ITEMS_PER_METHOD_PER_BATCH = 44

BATCH_ASSIGNMENT_SEED = 1901
BATCH_ORDER_SEEDS = {
    1: 1911,
    2: 1912,
}

ITEM_PREFIX = "VBR"

PANEL_SIZE = 600
HEADER_HEIGHT = 46
GAP = 14
OVERLAY_ALPHA = 0.28

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


# ============================================================
# 2. Utilities
# ============================================================

def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_png_unicode(path: Path, image) -> bool:
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fit_to_square(image, size: int):
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise RuntimeError("Invalid image size.")

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def create_candidate_overlay(image, coating_mask):
    mask = coating_mask > 0
    overlay = image.copy()

    green = np.zeros_like(image)
    green[:, :, 1] = 255

    blended = cv2.addWeighted(
        image,
        1.0 - OVERLAY_ALPHA,
        green,
        OVERLAY_ALPHA,
        0,
    )
    overlay[mask] = blended[mask]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def build_review_image(original, coating_mask, item_id: str):
    if original.shape[:2] != coating_mask.shape[:2]:
        coating_mask = cv2.resize(
            coating_mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if coating_mask.ndim == 3:
        coating_mask = coating_mask[:, :, 0]

    coating_mask = (coating_mask > 0).astype(np.uint8)
    overlay = create_candidate_overlay(original, coating_mask)

    left = fit_to_square(original, PANEL_SIZE)
    right = fit_to_square(overlay, PANEL_SIZE)

    total_width = PANEL_SIZE * 2 + GAP
    total_height = PANEL_SIZE + HEADER_HEIGHT

    canvas = np.zeros(
        (total_height, total_width, 3),
        dtype=np.uint8,
    )

    canvas[
        HEADER_HEIGHT:HEADER_HEIGHT + PANEL_SIZE,
        0:PANEL_SIZE,
    ] = left

    canvas[
        HEADER_HEIGHT:HEADER_HEIGHT + PANEL_SIZE,
        PANEL_SIZE + GAP:PANEL_SIZE + GAP + PANEL_SIZE,
    ] = right

    cv2.putText(
        canvas,
        "Original",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Candidate coating region",
        (PANEL_SIZE + GAP + 12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        item_id,
        (total_width - 145, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def dice_score(mask_a, mask_b) -> float:
    a = mask_a > 0
    b = mask_b > 0
    denominator = int(a.sum() + b.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(a, b).sum())
    return float(2.0 * intersection / denominator)


def build_lookup(folder: Path, name: str):
    if not folder.exists():
        raise FileNotFoundError(f"{name} directory not found:\n{folder}")

    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    lookup = {p.stem: p for p in paths}
    return lookup


# ============================================================
# 3. Refuse overwrite
# ============================================================

if OUTPUT_DIR.exists():
    raise RuntimeError(
        "Output directory already exists. To protect the frozen blind package, "
        "this script will not overwrite it:\n"
        f"{OUTPUT_DIR}\n"
        "If this is only a test run, rename/delete the old folder manually before rerunning."
    )

PUBLIC_DIR.mkdir(parents=True, exist_ok=False)
INTERNAL_DIR.mkdir(parents=True, exist_ok=False)
REVIEW_IMAGE_ROOT.mkdir(parents=True, exist_ok=False)

for batch in range(1, N_BATCHES + 1):
    (REVIEW_IMAGE_ROOT / f"batch_{batch:02d}").mkdir(
        parents=True,
        exist_ok=False,
    )


# ============================================================
# 4. Load frozen 88 cases
# ============================================================

if not BLIND_88_MAPPING.exists():
    raise FileNotFoundError(
        f"Frozen 88-case mapping not found:\n{BLIND_88_MAPPING}"
    )

blind_df = pd.read_csv(BLIND_88_MAPPING, dtype=str)
required_cols = {"blind_id", "image_id"}
missing = required_cols - set(blind_df.columns)
if missing:
    raise RuntimeError(
        "Frozen mapping missing columns: " + ", ".join(sorted(missing))
    )

blind_df["blind_id"] = blind_df["blind_id"].astype(str)
blind_df["image_id"] = blind_df["image_id"].astype(str)
blind_df = blind_df.sort_values("blind_id").reset_index(drop=True)

if len(blind_df) != EXPECTED_CASES:
    raise RuntimeError(
        f"Frozen case count={len(blind_df)}, expected={EXPECTED_CASES}."
    )

if blind_df["image_id"].duplicated().any():
    raise RuntimeError("Duplicate image_id detected in frozen 88 cases.")

frozen_ids = set(blind_df["image_id"])

image_lookup = build_lookup(FINAL_IMAGE_DIR, "Original image")
original_mask_lookup = build_lookup(ORIGINAL_D_MASK_DIR, "08B Original-D mask")
sr_mask_lookup = build_lookup(SR_D_MASK_DIR, "08B SR-D mask")
formal_d_lookup = build_lookup(FORMAL_D_MASK_DIR, "Step09 formal-D mask")

for name, lookup in [
    ("Original image", image_lookup),
    ("08B Original-D", original_mask_lookup),
    ("08B SR-D", sr_mask_lookup),
    ("Step09 formal-D", formal_d_lookup),
]:
    missing_ids = sorted(frozen_ids - set(lookup.keys()))
    if missing_ids:
        raise RuntimeError(
            f"{name} missing frozen cases:\n" + "\n".join(missing_ids)
        )


# ============================================================
# 5. Reconfirm Original-D == formal D on frozen 88
# ============================================================

repro_rows = []
for _, row in blind_df.iterrows():
    image_id = str(row["image_id"])

    a_img = imread_unicode(original_mask_lookup[image_id])
    b_img = imread_unicode(formal_d_lookup[image_id])

    if a_img is None or b_img is None:
        raise RuntimeError(f"Failed to read reproduction masks: {image_id}")

    a = (a_img[:, :, 0] > 0).astype(np.uint8)
    b = (b_img[:, :, 0] > 0).astype(np.uint8)

    if a.shape != b.shape:
        b = cv2.resize(
            b,
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    d = dice_score(a, b)
    repro_rows.append({"image_id": image_id, "dice": d})

repro_df = pd.DataFrame(repro_rows)
repro_min = float(repro_df["dice"].min())
repro_df.to_csv(
    INTERNAL_DIR / "original08B_vs_formalD_reproduction_INTERNAL.csv",
    index=False,
    encoding="utf-8-sig",
)

if repro_min < 0.999999:
    raise RuntimeError(
        "Original-D is not identical to the previously frozen formal D. "
        f"Minimum Dice={repro_min:.9f}"
    )


# ============================================================
# 6. Paired two-batch blinding design
#
# Each case appears exactly once per batch.
# Its Original-D and SR-D outputs are forced into different batches.
# Each batch contains 44 Original-D + 44 SR-D.
# ============================================================

rng = np.random.default_rng(BATCH_ASSIGNMENT_SEED)
offsets = np.repeat(np.arange(2, dtype=int), EXPECTED_CASES // 2)
rng.shuffle(offsets)

if len(offsets) != EXPECTED_CASES:
    raise RuntimeError("Offset count mismatch.")

design_rows = []
for case_index, row in blind_df.iterrows():
    offset = int(offsets[case_index])

    for batch in range(1, N_BATCHES + 1):
        method_index = (batch - 1 + offset) % 2
        branch = METHODS[method_index]

        design_rows.append(
            {
                "blind_id": str(row["blind_id"]),
                "image_id": str(row["image_id"]),
                "batch": batch,
                "branch": branch,
            }
        )

design_df = pd.DataFrame(design_rows)

if len(design_df) != EXPECTED_ITEMS:
    raise RuntimeError(
        f"Design item count={len(design_df)}, expected={EXPECTED_ITEMS}."
    )

for batch in range(1, N_BATCHES + 1):
    current = design_df[design_df["batch"] == batch]
    counts = current["branch"].value_counts().to_dict()

    if len(current) != ITEMS_PER_BATCH:
        raise RuntimeError(f"Batch {batch} item count is not 88.")

    if counts.get("ORIGINAL_D", 0) != ITEMS_PER_METHOD_PER_BATCH:
        raise RuntimeError(f"Batch {batch} Original-D count is not 44.")

    if counts.get("SR_D", 0) != ITEMS_PER_METHOD_PER_BATCH:
        raise RuntimeError(f"Batch {batch} SR-D count is not 44.")

# Same-case branches must be separated across the two batches.
for blind_id, g in design_df.groupby("blind_id"):
    if len(g) != 2:
        raise RuntimeError(f"Case {blind_id}: expected two VLM items.")
    if set(g["batch"].astype(int)) != {1, 2}:
        raise RuntimeError(f"Case {blind_id}: not separated across batches.")
    if set(g["branch"].astype(str)) != {"ORIGINAL_D", "SR_D"}:
        raise RuntimeError(f"Case {blind_id}: missing one branch.")


# ============================================================
# 7. Randomize order independently within each batch
# ============================================================

ordered_parts = []
for batch in range(1, N_BATCHES + 1):
    current = (
        design_df[design_df["batch"] == batch]
        .sample(frac=1.0, random_state=BATCH_ORDER_SEEDS[batch])
        .reset_index(drop=True)
    )
    current["batch_order"] = np.arange(1, len(current) + 1)
    ordered_parts.append(current)

design_df = pd.concat(ordered_parts, ignore_index=True)
design_df["global_order"] = np.arange(1, EXPECTED_ITEMS + 1)
design_df["item_id"] = [
    f"{ITEM_PREFIX}{i:03d}" for i in design_df["global_order"]
]


# ============================================================
# 8. Generate 176 blinded images
# ============================================================

public_rows = []
internal_rows = []

print("=" * 110)
print("Step 19 - Build Original-D vs SR-D VLM Blind Package")
print("=" * 110)
print(f"Frozen cases : {EXPECTED_CASES}")
print(f"VLM items    : {EXPECTED_ITEMS}")
print("Design       : 2 batches x 88; each batch 44 Original-D + 44 SR-D")
print("Pairing      : each case's Original-D and SR-D are separated across batches")
print(f"Formal-D reproduction min Dice: {repro_min:.9f}")
print("=" * 110)

for index, row in design_df.iterrows():
    item_id = str(row["item_id"])
    blind_id = str(row["blind_id"])
    image_id = str(row["image_id"])
    branch = str(row["branch"])
    batch = int(row["batch"])
    batch_order = int(row["batch_order"])

    original_path = image_lookup[image_id]
    if branch == "ORIGINAL_D":
        mask_path = original_mask_lookup[image_id]
    elif branch == "SR_D":
        mask_path = sr_mask_lookup[image_id]
    else:
        raise RuntimeError(f"Unknown branch: {branch}")

    original = imread_unicode(original_path)
    mask_img = imread_unicode(mask_path)

    if original is None:
        raise RuntimeError(f"Failed to read original image: {original_path}")
    if mask_img is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    review_image = build_review_image(original, mask_img, item_id)

    batch_dir = REVIEW_IMAGE_ROOT / f"batch_{batch:02d}"
    filename = f"{batch_order:03d}_{item_id}.png"
    review_path = batch_dir / filename

    if not imwrite_png_unicode(review_path, review_image):
        raise RuntimeError(f"Failed to write review image: {review_path}")

    relative_path = str(
        Path("review_images") / f"batch_{batch:02d}" / filename
    )

    image_sha = sha256_file(review_path)

    public_rows.append(
        {
            "item_id": item_id,
            "batch": batch,
            "batch_order": batch_order,
            "image_file": filename,
            "relative_image_path": relative_path,
        }
    )

    internal_rows.append(
        {
            "item_id": item_id,
            "batch": batch,
            "batch_order": batch_order,
            "blind_id": blind_id,
            "image_id": image_id,
            "branch": branch,
            "original_path": str(original_path),
            "mask_path": str(mask_path),
            "review_image_path": str(review_path),
            "review_image_sha256": image_sha,
        }
    )

    print(
        f"[{index + 1:03d}/{EXPECTED_ITEMS}] "
        f"{item_id} | Batch={batch} Order={batch_order:03d} | OK"
    )

public_df = pd.DataFrame(public_rows)
internal_df = pd.DataFrame(internal_rows)

public_df = public_df.sort_values(["batch", "batch_order"]).reset_index(drop=True)
internal_df = internal_df.sort_values(["batch", "batch_order"]).reset_index(drop=True)

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


# ============================================================
# 9. Freeze hash + config
# ============================================================

freeze_text = "\n".join(
    f"{r.item_id}|{r.batch}|{r.batch_order}|{r.blind_id}|{r.image_id}|{r.branch}|{r.review_image_sha256}"
    for r in internal_df.itertuples(index=False)
)

blinding_sha256 = sha256_text(freeze_text)

config = {
    "created_at": datetime.now().isoformat(),
    "frozen_cases": EXPECTED_CASES,
    "branches": METHODS,
    "expected_items": EXPECTED_ITEMS,
    "batches": N_BATCHES,
    "items_per_batch": ITEMS_PER_BATCH,
    "items_per_branch_per_batch": ITEMS_PER_METHOD_PER_BATCH,
    "batch_assignment_seed": BATCH_ASSIGNMENT_SEED,
    "batch_order_seeds": BATCH_ORDER_SEEDS,
    "review_input": (
        "Left panel is always the original tongue image; right panel is the blinded candidate coating overlay. "
        "Even SR-D items do not display the SR image itself."
    ),
    "formal_D_reproduction_min_dice_on_frozen88": repro_min,
    "blinding_sha256": blinding_sha256,
    "important": [
        "Both VLMs must use only PUBLIC_VLM_PACKAGE.",
        "The scoring script must not load INTERNAL_DO_NOT_SHARE.",
        "Original/SR branch identity is hidden during VLM scoring.",
        "VLM ratings assess visual-semantic plausibility, not pixel-level segmentation accuracy.",
    ],
}

CONFIG_JSON.write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# ============================================================
# 10. Final integrity checks
# ============================================================

if len(public_df) != EXPECTED_ITEMS or public_df["item_id"].nunique() != EXPECTED_ITEMS:
    raise RuntimeError("Public manifest item count / uniqueness check failed.")

if len(internal_df) != EXPECTED_ITEMS or internal_df["item_id"].nunique() != EXPECTED_ITEMS:
    raise RuntimeError("Internal mapping item count / uniqueness check failed.")

for forbidden in ["blind_id", "image_id", "branch", "mask_path"]:
    if forbidden in public_df.columns:
        raise RuntimeError(f"Public manifest leaked internal field: {forbidden}")

for batch in range(1, N_BATCHES + 1):
    b = internal_df[internal_df["batch"] == batch]
    if len(b) != 88:
        raise RuntimeError(f"Batch {batch}: count != 88")
    vc = b["branch"].value_counts().to_dict()
    if vc != {"ORIGINAL_D": 44, "SR_D": 44} and vc != {"SR_D": 44, "ORIGINAL_D": 44}:
        raise RuntimeError(f"Batch {batch}: branch balance failed: {vc}")

missing_files = [
    p for p in internal_df["review_image_path"]
    if not Path(p).exists()
]
if missing_files:
    raise RuntimeError("Missing generated review images.")


# ============================================================
# 11. Summary
# ============================================================

balance = pd.crosstab(internal_df["batch"], internal_df["branch"])

summary_lines = [
    "=" * 110,
    "Original-D vs SR-D VLM Blind Package Build Summary",
    "=" * 110,
    f"Frozen cases: {EXPECTED_CASES}",
    f"Total VLM items: {EXPECTED_ITEMS}",
    "Batches: 2 x 88",
    "Each batch: Original-D=44, SR-D=44",
    "Each case: Original-D and SR-D separated across the two batches",
    "",
    "BALANCE",
    "-" * 110,
    balance.to_string(),
    "",
    "FORMAL-D REPRODUCTION",
    "-" * 110,
    f"08B Original-D vs Step09 formal-D minimum Dice on frozen 88: {repro_min:.9f}",
    "",
    "BLINDING",
    "-" * 110,
    f"Blinding SHA256: {blinding_sha256}",
    f"PUBLIC package: {PUBLIC_DIR}",
    f"Internal mapping: {INTERNAL_MAPPING_CSV}",
    "",
    "IMPORTANT",
    "-" * 110,
    "1. The VLM scorer must read ONLY PUBLIC_VLM_PACKAGE.",
    "2. Do not expose INTERNAL_DO_NOT_SHARE to either VLM scoring process.",
    "3. Left panel is always the original tongue image for both branches.",
    "4. These scores are visual-semantic plausibility ratings, not pixel-level segmentation accuracy.",
    "",
    "FINAL CHECK: PASS",
    "=" * 110,
]

SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

print()
print("=" * 110)
print("STEP 19 COMPLETE")
print("=" * 110)
print(balance.to_string())
print()
print(f"Formal-D reproduction min Dice: {repro_min:.9f}")
print(f"Blinding SHA256: {blinding_sha256}")
print(f"PUBLIC VLM package: {PUBLIC_DIR}")
print(f"INTERNAL mapping: {INTERNAL_MAPPING_CSV}")
print("FINAL CHECK: PASS")
print("=" * 110)
