"""
Regenerate the complete 352-item human/VLM blind-review package after fixing FCM.

The script preserves the original frozen blind-review design:
    - the same 88 cases;
    - the same HBR001-HBR352 item IDs;
    - the same four batches and within-batch order;
    - the same method-to-item mapping.

It does NOT reuse old scores. It regenerates all 352 review images with one
common rendering procedure, embeds them into a new blank Excel workbook, and
creates separate public and internal manifests.

Required packages:
    pip install numpy opencv-python openpyxl pillow

Before running, edit PROJECT_ROOT if necessary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation


# ==================================================================================================
# 0. Paths: edit PROJECT_ROOT only when your project path is different
# ==================================================================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

INPUT_DIR = PROJECT_ROOT / "data" / "final_images"

OLD_REVIEW_ROOT = (
    PROJECT_ROOT
    / "results"
    / "14_human_blind_review_88"
)

# This file is the OLD internal mapping that fixes HBR item IDs, batches,
# order, cases and method identities. It is read only as a design manifest.
OLD_INTERNAL_MAPPING_CSV = (
    OLD_REVIEW_ROOT
    / "human_review_INTERNAL_mapping.csv"
)

OLD_MASK_ROOT = (
    PROJECT_ROOT
    / "results"
    / "12_method_comparison_A_B_C_D_full877"
    / "masks"
)

CORRECTED_MASK_ROOT = (
    PROJECT_ROOT
    / "results"
    / "12_method_comparison_A_B_C_D_full877_corrected"
    / "masks"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "14_human_blind_review_88_corrected_FCM"
)

REVIEW_IMAGE_ROOT = OUTPUT_ROOT / "review_images"

PUBLIC_MANIFEST_CSV = OUTPUT_ROOT / "human_review_public_manifest.csv"
INTERNAL_MAPPING_CSV = OUTPUT_ROOT / "human_review_INTERNAL_mapping.csv"
BLANK_REVIEW_XLSX = OUTPUT_ROOT / "human_blind_review_blank_corrected_FCM.xlsx"
AUDIT_JSON = OUTPUT_ROOT / "blind_review_generation_audit.json"
SUMMARY_TXT = OUTPUT_ROOT / "blind_review_generation_summary.txt"


# ==================================================================================================
# 1. Frozen design and rendering parameters
# ==================================================================================================

EXPECTED_ITEMS = 352
EXPECTED_CASES = 88
EXPECTED_BATCHES = 4
EXPECTED_ITEMS_PER_BATCH = 88
EXPECTED_METHODS = ("A", "B", "C", "D")

METHOD_NAMES = {
    "A": "KMeans",
    "B": "FCM",
    "C": "Spatial_FCM",
    "D": "SLIC_Spatial_FCM",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Review-image canvas, close to the previous 1214 x 646 layout.
CANVAS_WIDTH = 1214
CANVAS_HEIGHT = 646
TOP_MARGIN = 42
SIDE_MARGIN = 8
PANEL_GAP = 18
PANEL_WIDTH = (CANVAS_WIDTH - 2 * SIDE_MARGIN - PANEL_GAP) // 2
PANEL_HEIGHT = CANVAS_HEIGHT - TOP_MARGIN - 8

# Right-panel visualization. The color itself must not be interpreted as a biological cue.
OVERLAY_ALPHA = 0.32
OVERLAY_GREEN_BGR = (120, 205, 125)
BOUNDARY_YELLOW_BGR = (0, 255, 255)
BOUNDARY_THICKNESS = 3

TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 0.72
TITLE_THICKNESS = 2

SCORE_COLUMNS = (
    "coverage_consistency",
    "boundary_alignment",
    "oversegmentation_control",
    "undersegmentation_control",
    "fragmentation_control",
    "overall_plausibility",
)

SCORE_DEFINITIONS = {
    "coverage_consistency": (
        "How well the candidate coating region covers the visually apparent coating "
        "distribution, considering the whole tongue."
    ),
    "boundary_alignment": (
        "How well the candidate boundary follows visually plausible transitions "
        "between coating-like and body-like appearance."
    ),
    "oversegmentation_control": (
        "Whether obvious tongue-body or non-coating areas are avoided. "
        "Higher score means less visually apparent over-segmentation."
    ),
    "undersegmentation_control": (
        "Whether visually apparent coating areas are retained. "
        "Higher score means less visually apparent under-segmentation."
    ),
    "fragmentation_control": (
        "Whether the candidate coating region avoids implausible fragmentation, "
        "isolated islands and excessive local breaks."
    ),
    "overall_plausibility": (
        "Overall visual-semantic plausibility of the candidate tongue-body / "
        "tongue-coating separation."
    ),
}


# ==================================================================================================
# 2. General helpers
# ==================================================================================================

def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_png_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        raise RuntimeError(f"PNG encoding failed: {path}")
    encoded.tofile(str(path))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found:\n{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_by_stem(root: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = root / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Original image not found for image_id={image_id} under:\n{root}")


def normalized_batch(value: str) -> int:
    text = str(value).strip()
    if text.lower().startswith("batch_"):
        text = text.split("_", 1)[1]
    return int(text)


def fit_to_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot fit an empty image.")
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    panel[y:y + new_h, x:x + new_w] = resized
    return panel


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].copy()
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        # Preserve black background for transparent tongue PNGs.
        bgr = (bgr.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
        return bgr
    return image[:, :, :3].copy()


def load_binary_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Mask read failed:\n{path}")
    if mask_img.shape != expected_shape:
        raise RuntimeError(
            f"Mask/image shape mismatch for {path.name}: "
            f"mask={mask_img.shape}, image={expected_shape}"
        )
    unique_values = np.unique(mask_img)
    if not np.all(np.isin(unique_values, [0, 255])):
        raise RuntimeError(
            f"Mask is not binary 0/255: {path}\n"
            f"First unique values: {unique_values[:20].tolist()}"
        )
    return (mask_img > 0).astype(np.uint8)


# ==================================================================================================
# 3. Frozen mapping validation
# ==================================================================================================

def validate_and_normalize_mapping(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    required = {
        "item_id",
        "batch",
        "batch_order",
        "blind_id",
        "image_id",
        "method",
    }
    if not rows:
        raise RuntimeError("The old internal mapping is empty.")
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"Old internal mapping is missing columns: {sorted(missing)}")
    if len(rows) != EXPECTED_ITEMS:
        raise RuntimeError(f"Expected {EXPECTED_ITEMS} rows, found {len(rows)}")

    normalized: list[dict[str, object]] = []
    for r in rows:
        item_id = str(r["item_id"]).strip()
        batch = normalized_batch(r["batch"])
        batch_order = int(str(r["batch_order"]).strip())
        blind_id = str(r["blind_id"]).strip()
        image_id = str(r["image_id"]).strip()
        method = str(r["method"]).strip().upper()
        if method not in EXPECTED_METHODS:
            raise RuntimeError(f"Unexpected method {method} for {item_id}")
        normalized.append({
            "item_id": item_id,
            "batch": batch,
            "batch_order": batch_order,
            "blind_id": blind_id,
            "image_id": image_id,
            "method": method,
            "method_name": METHOD_NAMES[method],
        })

    normalized.sort(key=lambda x: (int(x["batch"]), int(x["batch_order"])))

    item_ids = [str(r["item_id"]) for r in normalized]
    if len(set(item_ids)) != EXPECTED_ITEMS:
        raise RuntimeError("item_id is not unique.")
    expected_item_ids = [f"HBR{i:03d}" for i in range(1, EXPECTED_ITEMS + 1)]
    if sorted(item_ids) != expected_item_ids:
        raise RuntimeError("item_id set is not exactly HBR001-HBR352.")

    image_ids = [str(r["image_id"]) for r in normalized]
    if len(set(image_ids)) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} unique cases, found {len(set(image_ids))}"
        )

    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_batch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for r in normalized:
        by_image[str(r["image_id"])].append(r)
        by_batch[int(r["batch"])].append(r)

    for image_id, group in by_image.items():
        methods = {str(r["method"]) for r in group}
        batches = {int(r["batch"]) for r in group}
        if len(group) != 4 or methods != set(EXPECTED_METHODS):
            raise RuntimeError(f"Case {image_id} does not contain exactly A/B/C/D once.")
        if batches != {1, 2, 3, 4}:
            raise RuntimeError(f"Case {image_id} does not appear exactly once in each batch.")

    if set(by_batch.keys()) != {1, 2, 3, 4}:
        raise RuntimeError(f"Batch set is abnormal: {sorted(by_batch.keys())}")

    for batch, group in by_batch.items():
        if len(group) != EXPECTED_ITEMS_PER_BATCH:
            raise RuntimeError(f"Batch {batch} contains {len(group)} items, expected 88.")
        orders = sorted(int(r["batch_order"]) for r in group)
        if orders != list(range(1, EXPECTED_ITEMS_PER_BATCH + 1)):
            raise RuntimeError(f"Batch {batch} order is not exactly 1-88.")
        method_counts = Counter(str(r["method"]) for r in group)
        if method_counts != Counter({"A": 22, "B": 22, "C": 22, "D": 22}):
            raise RuntimeError(f"Batch {batch} method balance is abnormal: {method_counts}")

    return normalized


# ==================================================================================================
# 4. Review-image rendering
# ==================================================================================================

def build_candidate_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    region = mask > 0
    if not np.any(region):
        raise RuntimeError("Candidate mask is empty.")

    green = np.zeros_like(out, dtype=np.uint8)
    green[:, :] = OVERLAY_GREEN_BGR
    blended = cv2.addWeighted(out, 1.0 - OVERLAY_ALPHA, green, OVERLAY_ALPHA, 0)
    out[region] = blended[region]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        out,
        contours,
        -1,
        BOUNDARY_YELLOW_BGR,
        BOUNDARY_THICKNESS,
        cv2.LINE_AA,
    )
    return out


def render_review_image(image: np.ndarray, mask: np.ndarray, item_id: str) -> np.ndarray:
    original_panel = fit_to_panel(image, PANEL_WIDTH, PANEL_HEIGHT)
    overlay_panel = fit_to_panel(
        build_candidate_overlay(image, mask),
        PANEL_WIDTH,
        PANEL_HEIGHT,
    )

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    left_x = SIDE_MARGIN
    right_x = SIDE_MARGIN + PANEL_WIDTH + PANEL_GAP
    canvas[TOP_MARGIN:TOP_MARGIN + PANEL_HEIGHT, left_x:left_x + PANEL_WIDTH] = original_panel
    canvas[TOP_MARGIN:TOP_MARGIN + PANEL_HEIGHT, right_x:right_x + PANEL_WIDTH] = overlay_panel

    white = (255, 255, 255)
    cv2.putText(
        canvas,
        "Original",
        (12, 30),
        TITLE_FONT,
        TITLE_SCALE,
        white,
        TITLE_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Candidate coating region",
        (right_x + 4, 30),
        TITLE_FONT,
        TITLE_SCALE,
        white,
        TITLE_THICKNESS,
        cv2.LINE_AA,
    )
    (text_w, _), _ = cv2.getTextSize(
        item_id,
        TITLE_FONT,
        TITLE_SCALE,
        TITLE_THICKNESS,
    )
    cv2.putText(
        canvas,
        item_id,
        (CANVAS_WIDTH - text_w - 12, 30),
        TITLE_FONT,
        TITLE_SCALE,
        white,
        TITLE_THICKNESS,
        cv2.LINE_AA,
    )
    return canvas


# ==================================================================================================
# 5. Excel workbook
# ==================================================================================================

def add_instructions_sheet(workbook: Workbook) -> None:
    ws = workbook.active
    ws.title = "Instructions"

    rows = [
        ("Human blind review - scoring instructions", ""),
        ("", ""),
        (
            "Purpose",
            "Evaluate visual-semantic plausibility of algorithmically generated candidate "
            "tongue-coating regions. This is NOT pixel-level ground-truth segmentation accuracy.",
        ),
        (
            "Blinding",
            "The reviewer must not access human_review_INTERNAL_mapping.csv. Method identity, "
            "original case ID, disease label and algorithmic metrics are intentionally hidden.",
        ),
        (
            "Review order",
            "Complete Batch 1, then Batch 2, Batch 3 and Batch 4. Each batch contains each of "
            "the 88 cases once, and the four outputs for the same case are separated across batches.",
        ),
        ("Score scale", "0-4 integer score for each dimension."),
        ("4", "Highly plausible / no obvious visual-semantic problem."),
        ("3", "Generally plausible / only minor local deviation."),
        ("2", "Moderate mismatch / requires review."),
        ("1", "Clear substantial mismatch."),
        ("0", "Severe visual-semantic inconsistency / essentially unacceptable."),
        ("", ""),
        *[(name, SCORE_DEFINITIONS[name]) for name in SCORE_COLUMNS],
        ("", ""),
        (
            "Important",
            "Do not infer disease status. Do not judge the image as CRC/NCRC. "
            "Only evaluate candidate-region plausibility.",
        ),
        (
            "Decision rule",
            "PASS/REVIEW/FAIL is derived automatically later; the reviewer does not assign it.",
        ),
        ("PASS", "All five core scores >=3 AND overall_plausibility >=3."),
        (
            "FAIL",
            "Any core score=0 OR at least two core scores<=1 OR overall_plausibility<=1.",
        ),
        ("REVIEW", "All remaining score patterns."),
        ("", ""),
        (
            "Embedded-image workbook",
            "The review PNGs are embedded directly in the four batch sheets. Scores are blank "
            "and must be entered independently for this corrected-FCM review round.",
        ),
    ]

    for r, values in enumerate(rows, start=1):
        ws.cell(r, 1, values[0])
        ws.cell(r, 2, values[1])

    ws.column_dimensions["A"].width = 31
    ws.column_dimensions["B"].width = 120
    ws.freeze_panes = "A3"
    ws.sheet_view.showGridLines = False

    title_fill = PatternFill("solid", fgColor="44546A")
    label_fill = PatternFill("solid", fgColor="D9E2F3")
    border = Border(bottom=Side(style="thin", color="B7C9E2"))

    for cell in ws[1]:
        cell.fill = title_fill
        cell.font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
    ws.row_dimensions[1].height = 28

    for row in range(3, len(rows) + 1):
        ws.cell(row, 1).font = Font(name="Arial", size=10, bold=True)
        ws.cell(row, 2).font = Font(name="Arial", size=10)
        ws.cell(row, 1).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row, 2).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row, 1).border = border
        ws.cell(row, 2).border = border
        if ws.cell(row, 1).value not in (None, ""):
            ws.cell(row, 1).fill = label_fill


def add_batch_sheet(
    workbook: Workbook,
    batch: int,
    batch_rows: list[dict[str, object]],
) -> None:
    ws = workbook.create_sheet(f"Batch_{batch:02d}")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"

    headers = [
        "item_id",
        "order",
        "Review image",
        *SCORE_COLUMNS,
        "reviewer_comment",
    ]

    for col, header in enumerate(headers, start=1):
        cell = ws.cell(1, col, header)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="44546A")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(
            left=Side(style="thin", color="FFFFFF"),
            right=Side(style="thin", color="FFFFFF"),
            top=Side(style="thin", color="FFFFFF"),
            bottom=Side(style="thin", color="FFFFFF"),
        )

    for index, score_name in enumerate(SCORE_COLUMNS, start=4):
        ws.cell(1, index).comment = Comment(
            SCORE_DEFINITIONS[score_name] + " Score must be an integer from 0 to 4.",
            "Study team",
        )

    ws.row_dimensions[1].height = 32
    ws.column_dimensions["A"].width = 13
    ws.column_dimensions["B"].width = 9
    ws.column_dimensions["C"].width = 96.43
    for col in "DEFGHI":
        ws.column_dimensions[col].width = 21
    ws.column_dimensions["J"].width = 38

    thin_gray = Side(style="thin", color="D9D9D9")
    score_fill = PatternFill("solid", fgColor="FFF2CC")
    invalid_fill = PatternFill("solid", fgColor="F4CCCC")

    for excel_row, item in enumerate(batch_rows, start=2):
        item_id = str(item["item_id"])
        batch_order = int(item["batch_order"])
        image_path = Path(str(item["review_image_path"]))
        relative_path = Path(str(item["relative_image_path"]))

        ws.cell(excel_row, 1, item_id)
        ws.cell(excel_row, 2, batch_order)
        ws.cell(excel_row, 3, "")
        ws.cell(excel_row, 3).hyperlink = str(relative_path)
        ws.cell(excel_row, 10, "")

        ws.row_dimensions[excel_row].height = 280.5

        for col in range(1, 11):
            cell = ws.cell(excel_row, col)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(
                horizontal="center" if col != 10 else "left",
                vertical="center",
                wrap_text=True,
            )
            cell.border = Border(
                left=thin_gray,
                right=thin_gray,
                top=thin_gray,
                bottom=thin_gray,
            )
        for col in range(4, 10):
            ws.cell(excel_row, col).fill = score_fill

        xl_image = XLImage(str(image_path))
        xl_image.width = 680
        xl_image.height = 362
        ws.add_image(xl_image, f"C{excel_row}")

    score_range = f"D2:I{1 + len(batch_rows)}"
    validation = DataValidation(
        type="whole",
        operator="between",
        formula1="0",
        formula2="4",
        allow_blank=True,
    )
    validation.promptTitle = "Score required"
    validation.prompt = "Enter an integer from 0 to 4."
    validation.errorTitle = "Invalid score"
    validation.error = "Only integers from 0 to 4 are allowed."
    validation.showInputMessage = True
    validation.showErrorMessage = True
    ws.add_data_validation(validation)
    validation.add(score_range)

    # This highlights values outside the intended range if pasted into Excel.
    ws.conditional_formatting.add(
        score_range,
        CellIsRule(operator="lessThan", formula=["0"], fill=invalid_fill),
    )
    ws.conditional_formatting.add(
        score_range,
        CellIsRule(operator="greaterThan", formula=["4"], fill=invalid_fill),
    )

    ws.auto_filter.ref = f"A1:J{1 + len(batch_rows)}"


def build_blank_review_workbook(generated_rows: list[dict[str, object]]) -> None:
    workbook = Workbook()
    add_instructions_sheet(workbook)

    for batch in range(1, EXPECTED_BATCHES + 1):
        batch_rows = [r for r in generated_rows if int(r["batch"]) == batch]
        batch_rows.sort(key=lambda r: int(r["batch_order"]))
        add_batch_sheet(workbook, batch, batch_rows)

    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    workbook.save(BLANK_REVIEW_XLSX)


def validate_saved_workbook() -> dict[str, object]:
    workbook = load_workbook(BLANK_REVIEW_XLSX, read_only=False, data_only=False)
    expected_sheets = ["Instructions"] + [f"Batch_{i:02d}" for i in range(1, 5)]
    if workbook.sheetnames != expected_sheets:
        raise RuntimeError(f"Workbook sheet order is abnormal: {workbook.sheetnames}")

    total_items = 0
    total_images = 0
    nonblank_scores = 0
    workbook_item_ids: list[str] = []

    for batch in range(1, 5):
        ws = workbook[f"Batch_{batch:02d}"]
        if ws.max_row != 89 or ws.max_column != 10:
            raise RuntimeError(
                f"Batch_{batch:02d} shape abnormal: {ws.max_row} x {ws.max_column}"
            )
        total_items += ws.max_row - 1
        total_images += len(ws._images)
        workbook_item_ids.extend(str(ws.cell(r, 1).value) for r in range(2, 90))
        for row in ws.iter_rows(min_row=2, max_row=89, min_col=4, max_col=9):
            nonblank_scores += sum(cell.value not in (None, "") for cell in row)

    if total_items != EXPECTED_ITEMS:
        raise RuntimeError(f"Workbook item count abnormal: {total_items}")
    if total_images != EXPECTED_ITEMS:
        raise RuntimeError(f"Workbook embedded-image count abnormal: {total_images}")
    if len(set(workbook_item_ids)) != EXPECTED_ITEMS:
        raise RuntimeError("Workbook item IDs are not unique.")
    if nonblank_scores != 0:
        raise RuntimeError("New blind-review workbook unexpectedly contains old/nonblank scores.")

    workbook.close()
    return {
        "sheet_names": expected_sheets,
        "workbook_items": total_items,
        "embedded_images": total_images,
        "nonblank_scores": nonblank_scores,
    }


# ==================================================================================================
# 6. Main generation
# ==================================================================================================

def main() -> None:
    start_time = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REVIEW_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for batch in range(1, EXPECTED_BATCHES + 1):
        (REVIEW_IMAGE_ROOT / f"batch_{batch:02d}").mkdir(parents=True, exist_ok=True)

    old_rows = read_csv_rows(OLD_INTERNAL_MAPPING_CSV)
    frozen_rows = validate_and_normalize_mapping(old_rows)

    print("=" * 110)
    print("Corrected-FCM human/VLM blind-review package")
    print("=" * 110)
    print(f"Frozen mapping : {OLD_INTERNAL_MAPPING_CSV}")
    print(f"Corrected masks: {CORRECTED_MASK_ROOT}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print(f"Items          : {EXPECTED_ITEMS}")
    print(f"Cases          : {EXPECTED_CASES}")
    print("=" * 110)

    generated_rows: list[dict[str, object]] = []
    old_unchanged_comparisons = {"A": 0, "C": 0, "D": 0}
    old_unchanged_checked = {"A": 0, "C": 0, "D": 0}
    selected_ab_exact_same = 0

    for global_index, row in enumerate(frozen_rows, start=1):
        item_id = str(row["item_id"])
        batch = int(row["batch"])
        batch_order = int(row["batch_order"])
        image_id = str(row["image_id"])
        method = str(row["method"])

        original_path = resolve_image_by_stem(INPUT_DIR, image_id)
        mask_path = CORRECTED_MASK_ROOT / method / f"{image_id}.png"

        original = imread_unicode(original_path, cv2.IMREAD_UNCHANGED)
        if original is None:
            raise FileNotFoundError(f"Original image read failed:\n{original_path}")
        original_bgr = ensure_bgr(original)
        mask = load_binary_mask(mask_path, original_bgr.shape[:2])

        relative_path = (
            Path("review_images")
            / f"batch_{batch:02d}"
            / f"{batch_order:03d}_{item_id}.png"
        )
        review_path = OUTPUT_ROOT / relative_path
        review_image = render_review_image(original_bgr, mask, item_id)
        imwrite_png_unicode(review_path, review_image)

        if not review_path.exists() or review_path.stat().st_size <= 0:
            raise RuntimeError(f"Review image was not saved correctly: {review_path}")

        # Optional deterministic audit: corrected A/C/D should match the old masks.
        if method in old_unchanged_checked:
            old_path = OLD_MASK_ROOT / method / f"{image_id}.png"
            if old_path.exists():
                old_unchanged_checked[method] += 1
                if sha256_file(old_path) == sha256_file(mask_path):
                    old_unchanged_comparisons[method] += 1

        # For the 88 selected cases, verify that corrected B is not still a copy of A.
        if method == "B":
            a_path = CORRECTED_MASK_ROOT / "A" / f"{image_id}.png"
            a_mask = load_binary_mask(a_path, original_bgr.shape[:2])
            if np.array_equal(a_mask, mask):
                selected_ab_exact_same += 1

        generated_rows.append({
            **row,
            "original_path": str(original_path),
            "mask_path": str(mask_path),
            "relative_image_path": str(relative_path),
            "review_image_path": str(review_path),
            "review_image_sha256": sha256_file(review_path),
        })

        print(
            f"[{global_index:03d}/{EXPECTED_ITEMS}] {item_id} | "
            f"Batch {batch} order {batch_order:02d} | generated"
        )

    if len(generated_rows) != EXPECTED_ITEMS:
        raise RuntimeError(f"Generated row count abnormal: {len(generated_rows)}")

    review_files = sorted(REVIEW_IMAGE_ROOT.glob("batch_*/*.png"))
    if len(review_files) != EXPECTED_ITEMS:
        raise RuntimeError(
            f"Expected {EXPECTED_ITEMS} review images, found {len(review_files)}"
        )

    # The old faulty run had all 88 selected A/B masks equal. Catch that exact failure.
    if selected_ab_exact_same == EXPECTED_CASES:
        raise RuntimeError(
            "All 88 selected corrected B masks are still identical to A. "
            "The script may be reading the old mask directory."
        )

    public_rows = [
        {
            "item_id": r["item_id"],
            "batch": r["batch"],
            "batch_order": r["batch_order"],
            "relative_image_path": r["relative_image_path"],
        }
        for r in generated_rows
    ]
    write_csv_rows(
        PUBLIC_MANIFEST_CSV,
        public_rows,
        ["item_id", "batch", "batch_order", "relative_image_path"],
    )

    write_csv_rows(
        INTERNAL_MAPPING_CSV,
        generated_rows,
        [
            "item_id",
            "batch",
            "batch_order",
            "blind_id",
            "image_id",
            "method",
            "method_name",
            "original_path",
            "mask_path",
            "relative_image_path",
            "review_image_path",
            "review_image_sha256",
        ],
    )

    print("\nEmbedding 352 images into the blank Excel workbook. This may take several minutes...")
    build_blank_review_workbook(generated_rows)
    workbook_audit = validate_saved_workbook()

    # Confirm public manifest contains no method/case-identifying columns.
    public_headers = set(public_rows[0].keys())
    forbidden_public_headers = {
        "image_id", "blind_id", "method", "method_name", "mask_path", "original_path"
    }
    public_blinding_ok = not bool(public_headers & forbidden_public_headers)
    if not public_blinding_ok:
        raise RuntimeError("Public manifest leaks method or case identity.")

    total_runtime = time.time() - start_time
    audit = {
        "created_at": datetime.now().isoformat(),
        "source_mapping": str(OLD_INTERNAL_MAPPING_CSV),
        "source_mapping_sha256": sha256_file(OLD_INTERNAL_MAPPING_CSV),
        "corrected_mask_root": str(CORRECTED_MASK_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "items": len(generated_rows),
        "unique_cases": len({str(r["image_id"]) for r in generated_rows}),
        "batches": EXPECTED_BATCHES,
        "items_per_batch": EXPECTED_ITEMS_PER_BATCH,
        "methods_per_batch": {"A": 22, "B": 22, "C": 22, "D": 22},
        "review_images": len(review_files),
        "selected_A_B_exact_same_masks": selected_ab_exact_same,
        "old_vs_corrected_A_C_D_checked": old_unchanged_checked,
        "old_vs_corrected_A_C_D_identical": old_unchanged_comparisons,
        "public_manifest_blinding_ok": public_blinding_ok,
        "workbook": workbook_audit,
        "workbook_sha256": sha256_file(BLANK_REVIEW_XLSX),
        "runtime_seconds": total_runtime,
        "final_check": "PASS",
    }
    with AUDIT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    lines = [
        "=" * 110,
        "Corrected-FCM Human/VLM Blind-Review Package",
        "=" * 110,
        f"Items: {len(generated_rows)}",
        f"Unique cases: {audit['unique_cases']}",
        f"Batches: {EXPECTED_BATCHES} x {EXPECTED_ITEMS_PER_BATCH}",
        f"Review images: {len(review_files)}",
        f"Selected corrected A/B exact-same masks: {selected_ab_exact_same}/{EXPECTED_CASES}",
        f"Public manifest blinding check: {public_blinding_ok}",
        f"Workbook embedded images: {workbook_audit['embedded_images']}",
        f"Workbook nonblank score cells: {workbook_audit['nonblank_scores']}",
        "",
        f"Public manifest: {PUBLIC_MANIFEST_CSV}",
        f"Internal mapping: {INTERNAL_MAPPING_CSV}",
        f"Blank scoring workbook: {BLANK_REVIEW_XLSX}",
        f"Audit JSON: {AUDIT_JSON}",
        f"Runtime: {total_runtime:.2f} seconds",
        "",
        "FINAL CHECK: PASS",
        "=" * 110,
    ]
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("Important: keep human_review_INTERNAL_mapping.csv away from human and VLM evaluators.")


if __name__ == "__main__":
    main()

