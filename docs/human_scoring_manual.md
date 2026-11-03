# Human blind-review scoring manual

## Purpose and source

Evaluate the **visual-semantic plausibility** of a candidate tongue-coating
region. These ordinal ratings are not expert pixel-level ground-truth
segmentation accuracy and are not a disease diagnosis.

This standalone manual consolidates the workbook instructions in
[`add_instructions_sheet`](../scripts/15_generate_blind_review_corrected_fcm.py)
and the supplied [frozen VLM scoring principles](16_frozen_vlm_scoring_prompt.txt).
The dimension-specific anchors below reproduce the existing English
`SCORING_PROMPT` in [step 16](../scripts/16_dual_vlm_blind_review.py).
No scoring dimensions, numeric thresholds, or decision rules are changed.
This consolidation does not establish that historical human reviewers
received this expanded, dimension-by-dimension wording; document the actual
materials used for each review round when reporting study methods.

## Materials, blinding, and review procedure

1. Work only from the blinded review images and blank scoring workbook.
   Do not access internal mappings, method identities, original case IDs,
   disease labels, algorithmic metrics, or other reviewers' scores.
2. Review Batch 1, then Batch 2, Batch 3, and Batch 4, following the item
   order within each batch. In the corrected four-method design, each batch
   contains 88 items; each case appears once per batch, with its four method
   outputs separated across batches. Do not attempt to reconstruct identities.
3. Judge tongue color and texture from the **left original image**. Use the
   **right image** to locate the proposed candidate region and yellow boundary.
   Do not interpret the green overlay as a biological feature.
4. Judge each item independently. Do not infer CRC/NCRC or any other disease,
   guess its algorithm, or compare it with another case or method output.
5. Enter one integer from 0 to 4 in each of the six named score columns.
   A higher score always indicates a more plausible candidate region or
   better control of the stated error. Score oversegmentation and
   undersegmentation separately; they may coexist.
6. Use `reviewer_comment` for a brief visual reason or an image-display problem.
   A missing/unreadable image is not automatically a score of zero: leave
   affected entries unscored and resolve it through the review coordinator.
   Missing entries must not be imputed as zero or assigned PASS/REVIEW/FAIL.
7. Reviewers enter the six scores, not the derived decision. Preserve the item
   IDs and batch order when returning the completed workbook.

The batch counts above describe the four-method corrected-FCM package.
For SR-gain packages, retain their generated item IDs and batch order rather
than assuming the same number of items; the same six dimensions and decision
rules apply.

## General 0-4 scale

| Score | General meaning |
| --- | --- |
| 4 | Highly plausible; no obvious visual-semantic problem. |
| 3 | Generally plausible; only minor local deviation. |
| 2 | Moderate mismatch; requires review. |
| 1 | Clear substantial mismatch. |
| 0 | Severe visual-semantic inconsistency; essentially unacceptable. |

These are qualitative visual anchors. No pixel-distance or area-percentage
cutoffs are introduced, and exact pixel-level boundary agreement is not required.

## Dimension-specific anchors

### `coverage_consistency`

Judge whether the candidate coating region broadly covers the visually apparent tongue-coating distribution in the original image.

| Score | Anchor |
| --- | --- |
| 4 | visually apparent coating distribution is covered very well overall. |
| 3 | coverage is generally appropriate with only minor local omission or excess. |
| 2 | moderate mismatch in coverage; a noticeable portion is omitted or unnecessarily included. |
| 1 | large mismatch; substantial visually apparent coating is missed or the proposed coverage is largely inconsistent with the original. |
| 0 | the proposed coverage is severely inconsistent with the visually apparent coating distribution. |

### `boundary_alignment`

Judge whether the yellow candidate boundary broadly follows visually plausible transitions between coating-like and tongue-body-like appearance. Exact pixel-level agreement is NOT required.

| Score | Anchor |
| --- | --- |
| 4 | boundary location and overall course are highly plausible. |
| 3 | boundary is generally plausible with only small local displacement. |
| 2 | moderate boundary mismatch in several locations. |
| 1 | large portions of the boundary are clearly misplaced. |
| 0 | boundary is fundamentally inconsistent with the visible coating/body appearance. |

### `oversegmentation_control`

Judge whether the candidate coating region avoids including areas that are visually more consistent with tongue body rather than coating. Higher score = less visually apparent over-segmentation.

| Score | Anchor |
| --- | --- |
| 4 | essentially no obvious inappropriate inclusion. |
| 3 | only small/local inappropriate inclusion. |
| 2 | moderate amount of visually inappropriate inclusion. |
| 1 | substantial tongue-body-like area is included. |
| 0 | very extensive inappropriate inclusion; the candidate region is largely over-segmented. |

### `undersegmentation_control`

Judge whether visually apparent coating-like regions are retained by the candidate region. Higher score = less visually apparent under-segmentation.

| Score | Anchor |
| --- | --- |
| 4 | essentially all obvious coating-like regions are retained. |
| 3 | only small/local omission. |
| 2 | moderate omission of visually apparent coating. |
| 1 | substantial obvious coating-like area is omitted. |
| 0 | most visually apparent coating-like area is missed. |

### `fragmentation_control`

Judge spatial coherence: avoid implausible islands, holes, excessive local breaks, jagged fragmentation, or highly scattered pieces. Higher score = more spatially coherent and less fragmented.

| Score | Anchor |
| --- | --- |
| 4 | highly coherent; essentially no problematic fragmentation. |
| 3 | generally coherent with only minor holes/islands/local breaks. |
| 2 | noticeable fragmentation, but the main structure remains interpretable. |
| 1 | strong fragmentation with many implausible holes/islands/breaks. |
| 0 | extremely fragmented or spatially incoherent candidate region. |

### `overall_plausibility`

Considering the image as a whole, judge whether the proposed candidate tongue-body/tongue-coating partition is visually and semantically plausible.

| Score | Anchor |
| --- | --- |
| 4 | highly plausible overall. |
| 3 | generally plausible overall with minor limitations. |
| 2 | moderate overall concern; requires review. |
| 1 | clearly implausible overall. |
| 0 | severely implausible overall. |

## Automatically derived PASS / REVIEW / FAIL

The **five core scores** are `coverage_consistency`, `boundary_alignment`,
`oversegmentation_control`, `undersegmentation_control`, and
`fragmentation_control`. `overall_plausibility` is the separate sixth score.
Apply the following only after all six scores are valid integers in 0-4:

| Decision | Unchanged rule |
| --- | --- |
| PASS | Every core score is at least 3 **and** overall plausibility is at least 3. |
| FAIL | Any core score is 0; **or** at least two core scores are at most 1; **or** overall plausibility is at most 1. |
| REVIEW | Every other complete, valid score pattern. |

The classification is derived in post-processing, matching `derive_decision`
in [step 16](../scripts/16_dual_vlm_blind_review.py), not selected by the
human reviewer or returned by the VLM.

Examples (five core scores; then overall plausibility):

| Scores | Derived decision | Reason |
| --- | --- | --- |
| 3, 3, 3, 3, 3; 3 | PASS | All six scores meet the PASS thresholds. |
| 2, 3, 3, 3, 3; 3 | REVIEW | A moderate core deviation prevents PASS; no FAIL condition. |
| 1, 3, 3, 3, 3; 3 | REVIEW | One low core score alone does not meet the FAIL rule. |
| 1, 1, 3, 3, 3; 3 | FAIL | Two core scores are at most 1. |
| 0, 4, 4, 4, 4; 4 | FAIL | One core score is 0. |
| 4, 4, 4, 4, 4; 1 | FAIL | Overall plausibility is at most 1. |

## Human and VLM output formats

Human reviewers use the six integer workbook columns and optional
`reviewer_comment`. The VLM uses the same six score keys plus `brief_reason`
as a JSON object, as specified in the supplied frozen principles and the
embedded step-16 prompt. The VLM must not output a disease label, algorithm
identity, or a self-assigned PASS/REVIEW/FAIL label. The original supplied
text is preserved in this repository without modification.
