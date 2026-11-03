from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests


# ==================================================================================================
# 0. Project paths
# ==================================================================================================

import os
PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))

REVIEW_ROOT = (
    PROJECT_ROOT
    / "results"
    / "19_sr_gain_vlm_blind_review_88"
    / "PUBLIC_VLM_PACKAGE"
)

PUBLIC_MANIFEST = (
    REVIEW_ROOT
    / "sr_gain_vlm_public_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "20_dual_vlm_sr_gain_blind_review"
)

EXPECTED_ITEMS = 176


# ==================================================================================================
# 1. Fixed VLM scoring prompt
#
# IMPORTANT:
# - Same prompt for Qwen and Doubao.
# - Same frozen 352 blind-review images.
# - The model must not infer disease or method identity.
# - Scores are visual-semantic plausibility ratings, NOT pixel-level GT accuracy.
# ==================================================================================================

SCORING_PROMPT = r"""
You are independently evaluating the visual-semantic plausibility of an algorithmically generated
candidate tongue-coating region.

The image has TWO PANELS:
- LEFT: the original tongue image.
- RIGHT: the same tongue image with the candidate tongue-coating region highlighted by a
  semi-transparent green overlay and a yellow boundary.

IMPORTANT:
1. Judge tongue color/texture primarily from the LEFT original image.
2. Use the RIGHT panel only to identify the proposed candidate region and its boundary.
3. Do NOT treat the green overlay color itself as a biological or diagnostic cue.
4. Do NOT infer disease status.
5. Do NOT guess which algorithm generated the candidate region.
6. Do NOT compare this image with any other method or case.
7. This is NOT expert pixel-level ground-truth segmentation accuracy. It is a blinded
   visual-semantic plausibility assessment.

Score EACH of the following six dimensions using an INTEGER from 0 to 4.

GENERAL SCALE
4 = highly plausible / no obvious visual-semantic problem
3 = generally plausible / only minor local deviation
2 = moderate mismatch / requires review
1 = clear substantial mismatch
0 = severe visual-semantic inconsistency / essentially unacceptable

DIMENSION 1 — coverage_consistency
Judge whether the candidate coating region broadly covers the visually apparent tongue-coating
distribution in the original image.
4: visually apparent coating distribution is covered very well overall.
3: coverage is generally appropriate with only minor local omission or excess.
2: moderate mismatch in coverage; a noticeable portion is omitted or unnecessarily included.
1: large mismatch; substantial visually apparent coating is missed or the proposed coverage is
   largely inconsistent with the original.
0: the proposed coverage is severely inconsistent with the visually apparent coating distribution.

DIMENSION 2 — boundary_alignment
Judge whether the yellow candidate boundary broadly follows visually plausible transitions between
coating-like and tongue-body-like appearance. Exact pixel-level agreement is NOT required.
4: boundary location and overall course are highly plausible.
3: boundary is generally plausible with only small local displacement.
2: moderate boundary mismatch in several locations.
1: large portions of the boundary are clearly misplaced.
0: boundary is fundamentally inconsistent with the visible coating/body appearance.

DIMENSION 3 — oversegmentation_control
Judge whether the candidate coating region avoids including areas that are visually more consistent
with tongue body rather than coating.
Higher score = less visually apparent over-segmentation.
4: essentially no obvious inappropriate inclusion.
3: only small/local inappropriate inclusion.
2: moderate amount of visually inappropriate inclusion.
1: substantial tongue-body-like area is included.
0: very extensive inappropriate inclusion; the candidate region is largely over-segmented.

DIMENSION 4 — undersegmentation_control
Judge whether visually apparent coating-like regions are retained by the candidate region.
Higher score = less visually apparent under-segmentation.
4: essentially all obvious coating-like regions are retained.
3: only small/local omission.
2: moderate omission of visually apparent coating.
1: substantial obvious coating-like area is omitted.
0: most visually apparent coating-like area is missed.

DIMENSION 5 — fragmentation_control
Judge spatial coherence: avoid implausible islands, holes, excessive local breaks, jagged
fragmentation, or highly scattered pieces.
Higher score = more spatially coherent and less fragmented.
4: highly coherent; essentially no problematic fragmentation.
3: generally coherent with only minor holes/islands/local breaks.
2: noticeable fragmentation, but the main structure remains interpretable.
1: strong fragmentation with many implausible holes/islands/breaks.
0: extremely fragmented or spatially incoherent candidate region.

DIMENSION 6 — overall_plausibility
Considering the image as a whole, judge whether the proposed candidate tongue-body/tongue-coating
partition is visually and semantically plausible.
4: highly plausible overall.
3: generally plausible overall with minor limitations.
2: moderate overall concern; requires review.
1: clearly implausible overall.
0: severely implausible overall.

Return ONLY one JSON object with EXACTLY these keys:
{
  "coverage_consistency": <0-4 integer>,
  "boundary_alignment": <0-4 integer>,
  "oversegmentation_control": <0-4 integer>,
  "undersegmentation_control": <0-4 integer>,
  "fragmentation_control": <0-4 integer>,
  "overall_plausibility": <0-4 integer>,
  "brief_reason": "<one short sentence explaining the main visual reason>"
}

Do not output markdown, code fences, PASS/FAIL labels, disease labels, algorithm names, or any
additional keys.
""".strip()


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


# ==================================================================================================
# 2. Provider configuration
# ==================================================================================================

PROVIDERS = {
    "qwen": {
        # Official Model Studio OpenAI-compatible endpoint.
        "base_url": os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "api_key": os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        "model": os.getenv("QWEN_MODEL", "qwen3-vl-plus"),
    },
    "doubao": {
        # Official Volcano Ark ChatCompletions endpoint.
        "base_url": os.getenv(
            "DOUBAO_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        ),
        "api_key": os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY"),
        # Volcano Ark requires an available model / endpoint ID from your Ark console.
        "model": os.getenv("DOUBAO_MODEL_ID", ""),
    },
}


# ==================================================================================================
# 3. HTTP / parsing parameters
# ==================================================================================================

REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 4
RETRY_BASE_SECONDS = 3.0

TEMPERATURE = 0.0
MAX_TOKENS = 500


# ==================================================================================================
# 4. Utilities
# ==================================================================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_to_data_url(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    suffix = path.suffix.lower()

    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image format: {path}")

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}", sha256_bytes(raw)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Remove accidental code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Robust fallback: take the first outer JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse a JSON object from model output.")


def validate_scores(obj: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(SCORE_COLUMNS + ["brief_reason"])
    extra = set(obj.keys()) - allowed
    missing = set(SCORE_COLUMNS) - set(obj.keys())

    if extra:
        raise ValueError(f"Unexpected JSON keys: {sorted(extra)}")
    if missing:
        raise ValueError(f"Missing score keys: {sorted(missing)}")

    clean: Dict[str, Any] = {}

    for col in SCORE_COLUMNS:
        value = obj[col]

        # Reject booleans explicitly because bool is a subclass of int in Python.
        if isinstance(value, bool):
            raise ValueError(f"{col} is boolean, not integer.")

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        if isinstance(value, str) and re.fullmatch(r"[0-4]", value.strip()):
            value = int(value.strip())

        if not isinstance(value, int):
            raise ValueError(f"{col} is not an integer: {value!r}")

        if not 0 <= value <= 4:
            raise ValueError(f"{col} out of range 0-4: {value}")

        clean[col] = int(value)

    reason = str(obj.get("brief_reason", "")).strip()
    clean["brief_reason"] = reason[:500]

    return clean


def derive_decision(scores: Dict[str, Any]) -> str:
    core = [int(scores[c]) for c in CORE_SCORE_COLUMNS]
    overall = int(scores["overall_plausibility"])

    if min(core) >= 3 and overall >= 3:
        return "PASS"

    if (
        min(core) == 0
        or sum(v <= 1 for v in core) >= 2
        or overall <= 1
    ):
        return "FAIL"

    return "REVIEW"


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ==================================================================================================
# 5. API call
# ==================================================================================================

def call_vlm(
    provider: str,
    image_path: Path,
) -> Dict[str, Any]:

    cfg = PROVIDERS[provider]

    api_key = cfg["api_key"]
    model = cfg["model"]
    base_url = str(cfg["base_url"]).rstrip("/")

    if not api_key:
        raise RuntimeError(
            f"{provider}: API key is missing.\n"
            f"Qwen: set QWEN_API_KEY or DASHSCOPE_API_KEY.\n"
            f"Doubao: set DOUBAO_API_KEY or ARK_API_KEY."
        )

    if not model:
        raise RuntimeError(
            "Doubao model ID is missing. Set environment variable "
            "DOUBAO_MODEL_ID to the multimodal model / endpoint ID available "
            "in your Volcano Ark console."
        )

    data_url, image_sha = image_to_data_url(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        },
                    },
                    {
                        "type": "text",
                        "text": SCORING_PROMPT,
                    },
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{base_url}/chat/completions"

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            elapsed = time.perf_counter() - t0

            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )

            raw_response = response.json()

            content = (
                raw_response["choices"][0]["message"]["content"]
            )

            # Some providers may represent content as a list.
            if isinstance(content, list):
                pieces = []
                for part in content:
                    if isinstance(part, dict):
                        if "text" in part:
                            pieces.append(str(part["text"]))
                        elif part.get("type") == "text":
                            pieces.append(str(part.get("content", "")))
                    else:
                        pieces.append(str(part))
                content = "\n".join(pieces)

            parsed = validate_scores(
                extract_json_object(str(content))
            )

            return {
                "scores": parsed,
                "raw_text": str(content),
                "raw_response": raw_response,
                "latency_seconds": elapsed,
                "image_sha256": image_sha,
                "model": model,
                "base_url": base_url,
            }

        except Exception as exc:
            last_error = exc

            if attempt >= MAX_RETRIES:
                break

            sleep_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))

            print(
                f"  retry {attempt}/{MAX_RETRIES - 1} after error: "
                f"{type(exc).__name__}: {exc}"
            )

            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


# ==================================================================================================
# 6. Manifest loading
# ==================================================================================================

def load_manifest() -> list[dict[str, str]]:
    if not PUBLIC_MANIFEST.exists():
        raise FileNotFoundError(
            f"Public manifest not found:\n{PUBLIC_MANIFEST}"
        )

    with PUBLIC_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_ITEMS:
        raise RuntimeError(
            f"Expected {EXPECTED_ITEMS} manifest rows, got {len(rows)}."
        )

    item_ids = [str(r["item_id"]) for r in rows]

    if len(set(item_ids)) != EXPECTED_ITEMS:
        raise RuntimeError("item_id is not unique in public manifest.")

    # Preserve frozen public-manifest order.
    return rows


# ==================================================================================================
# 7. Resume-safe output
# ==================================================================================================

def load_existing_completed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    completed = {
        str(r["item_id"])
        for r in rows
        if str(r.get("status", "")).upper() == "OK"
    }

    return completed


def append_csv_row(
    csv_path: Path,
    row: Dict[str, Any],
    fieldnames: list[str],
) -> None:
    exists = csv_path.exists()

    with csv_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ==================================================================================================
# 8. Main provider run
# ==================================================================================================

def run_provider(
    provider: str,
    start_from: int = 1,
    limit: Optional[int] = None,
) -> None:

    provider = provider.lower()

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}"
        )

    manifest = load_manifest()

    cfg = PROVIDERS[provider]

    provider_dir = (
        OUTPUT_ROOT
        / provider
    )

    provider_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_csv = (
        provider_dir
        / f"{provider}_vlm_scores.csv"
    )

    raw_jsonl = (
        provider_dir
        / f"{provider}_raw_responses.jsonl"
    )

    errors_csv = (
        provider_dir
        / f"{provider}_errors.csv"
    )

    config_json = (
        provider_dir
        / f"{provider}_run_config.json"
    )

    prompt_txt = (
        provider_dir
        / "frozen_vlm_scoring_prompt.txt"
    )

    prompt_txt.write_text(
        SCORING_PROMPT,
        encoding="utf-8",
    )

    config = {
        "provider": provider,
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "prompt_sha256": sha256_text(SCORING_PROMPT),
        "expected_items": EXPECTED_ITEMS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "manifest": str(PUBLIC_MANIFEST),
        "review_root": str(REVIEW_ROOT),
        "created_at": utc_timestamp(),
        "important": [
            "The same frozen 176 Original-D/SR-D blinded images are used for both VLMs.",
            "The same frozen scoring prompt is used for both VLMs.",
            "Internal Original-D/SR-D mapping is not loaded by this script.",
            "VLM ratings assess visual-semantic plausibility, not pixel-level segmentation accuracy.",
        ],
    }

    config_json.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = load_existing_completed(results_csv)

    rows_to_run = manifest[
        max(0, start_from - 1):
    ]

    if limit is not None:
        rows_to_run = rows_to_run[:limit]

    fieldnames = [
        "item_id",
        "batch",
        "batch_order",
        "relative_image_path",
        "provider",
        "model",
        *SCORE_COLUMNS,
        "decision",
        "brief_reason",
        "image_sha256",
        "prompt_sha256",
        "latency_seconds",
        "timestamp_utc",
        "status",
    ]

    print("=" * 110)
    print("Step 20 - Original-D vs SR-D Dual VLM Blind Review")
    print("=" * 110)
    print(f"Provider      : {provider}")
    print(f"Model         : {cfg['model']}")
    print(f"Base URL      : {cfg['base_url']}")
    print(f"Frozen items  : {EXPECTED_ITEMS}")
    print(f"Already done  : {len(completed)}")
    print(f"Prompt SHA256 : {sha256_text(SCORING_PROMPT)}")
    print("=" * 110)

    success = 0
    skipped = 0
    failed = 0

    for idx, r in enumerate(rows_to_run, start=1):
        item_id = str(r["item_id"])

        if item_id in completed:
            skipped += 1
            continue

        relative_path = Path(
            str(r["relative_image_path"])
        )

        image_path = (
            REVIEW_ROOT
            / relative_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Review image not found:\n{image_path}"
            )

        print(
            f"[{idx:03d}/{len(rows_to_run):03d}] "
            f"{item_id} | {provider} ...",
            end=" ",
            flush=True,
        )

        try:
            result = call_vlm(
                provider=provider,
                image_path=image_path,
            )

            scores = result["scores"]

            decision = derive_decision(scores)

            output_row = {
                "item_id": item_id,
                "batch": r["batch"],
                "batch_order": r["batch_order"],
                "relative_image_path": str(relative_path),
                "provider": provider,
                "model": result["model"],
                **{
                    col: scores[col]
                    for col in SCORE_COLUMNS
                },
                "decision": decision,
                "brief_reason": scores.get("brief_reason", ""),
                "image_sha256": result["image_sha256"],
                "prompt_sha256": sha256_text(SCORING_PROMPT),
                "latency_seconds": f"{result['latency_seconds']:.4f}",
                "timestamp_utc": utc_timestamp(),
                "status": "OK",
            }

            append_csv_row(
                results_csv,
                output_row,
                fieldnames,
            )

            raw_record = {
                "item_id": item_id,
                "provider": provider,
                "model": result["model"],
                "image_sha256": result["image_sha256"],
                "prompt_sha256": sha256_text(SCORING_PROMPT),
                "raw_text": result["raw_text"],
                "raw_response": result["raw_response"],
                "timestamp_utc": utc_timestamp(),
            }

            with raw_jsonl.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        raw_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            completed.add(item_id)
            success += 1

            print(
                f"OK | overall={scores['overall_plausibility']} "
                f"| {decision}"
            )

        except Exception as exc:
            failed += 1

            error_row = {
                "item_id": item_id,
                "batch": r["batch"],
                "batch_order": r["batch_order"],
                "relative_image_path": str(relative_path),
                "provider": provider,
                "model": cfg["model"],
                "timestamp_utc": utc_timestamp(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

            append_csv_row(
                errors_csv,
                error_row,
                [
                    "item_id",
                    "batch",
                    "batch_order",
                    "relative_image_path",
                    "provider",
                    "model",
                    "timestamp_utc",
                    "error_type",
                    "error_message",
                ],
            )

            print(
                f"FAILED | {type(exc).__name__}: {exc}"
            )

    final_completed = load_existing_completed(results_csv)

    print()
    print("=" * 110)
    print("Run summary")
    print("=" * 110)
    print(f"Provider       : {provider}")
    print(f"Success now    : {success}")
    print(f"Skipped/resume : {skipped}")
    print(f"Failed now     : {failed}")
    print(f"Total complete : {len(final_completed)}/{EXPECTED_ITEMS}")
    print(f"Scores CSV     : {results_csv}")
    print(f"Raw JSONL      : {raw_jsonl}")
    print(f"Errors CSV     : {errors_csv}")
    print()

    if len(final_completed) == EXPECTED_ITEMS:
        print("FINAL CHECK: PASS")
    else:
        print(
            "FINAL CHECK: INCOMPLETE "
            f"({len(final_completed)}/{EXPECTED_ITEMS})"
        )

    print("=" * 110)


# ==================================================================================================
# 9. CLI
# ==================================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run blinded 0-4 visual-semantic scoring on the 176 frozen Original-D/SR-D comparison images using Qwen and/or Doubao."
        )
    )

    parser.add_argument(
        "--provider",
        choices=[
            "qwen",
            "doubao",
            "all",
        ],
        required=True,
        help="VLM provider to run.",
    )

    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help=(
            "1-based starting row in the frozen public manifest. "
            "Normally leave as 1; resume is automatic."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of manifest rows for a small API test. "
            "Example: --limit 3"
        ),
    )

    args = parser.parse_args()

    if args.start_from < 1:
        raise ValueError("--start-from must be >= 1.")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.provider == "all":
        run_provider(
            "qwen",
            start_from=args.start_from,
            limit=args.limit,
        )
        run_provider(
            "doubao",
            start_from=args.start_from,
            limit=args.limit,
        )
    else:
        run_provider(
            args.provider,
            start_from=args.start_from,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
