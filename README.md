# SR-SFCM

Code-only release of the supplied tongue-image quality-control,
Real-ESRGAN x2 super-resolution, SLIC + spatial fuzzy C-means
candidate-region pipeline, and functional blind-review tools.

The inference kernels come from `12_method_compare_corrected.py`.
The corrected standard FCM implementation updates cluster centers before
recomputing memberships and checking convergence. The lightweight runner
uses extracted inference kernels; the optional cohort workflow also retains
the corrected full-dataset mask-generation and implementation-audit script.

## Installation

Use a Python virtual environment. Python 3.10 or 3.11 is suggested for the
optional legacy BasicSR dependency; the segmentation-only code was checked
locally on Python 3.14.5. Dependency ranges are compatibility constraints,
not a claim that every combination was tested.

```bash
python -m venv .venv
# Activate the virtual environment using your operating system's command.
python -m pip install -r requirements.txt
```

For super-resolution, install a compatible PyTorch/torchvision pair for your
CPU or CUDA platform, then install the optional dependencies:

```bash
python -m pip install -r requirements-sr.txt
```

Obtain `RealESRGAN_x2plus.pth` from the upstream
[Real-ESRGAN project](https://github.com/xinntao/Real-ESRGAN).
The weights are external dependencies and are not included here.
The SR runner records the weight SHA-256 in its run manifest.

## Execution order

### 01–06: original cohort preparation (optional)

Place background-separated input images in `data/raw_images/` and run
the following scripts from the repository root in numerical order:

```bash
python scripts/01_image_quality_control.py
python scripts/02_qc_visual_review.py
python scripts/03_adaptive_roi_qc.py
python scripts/04_final_image_quality_review.py
python scripts/05_build_manual_qc_54.py
python scripts/06_freeze_final_dataset_877.py
```

These are the original cohort-specific preparation scripts, with only the
project root made portable. They retain assumptions about 931 original
images, 54 exclusion candidates, and 877 retained images. They generate QC
review material and cohort manifests; they are not a general-purpose
automatic clinical inclusion rule. Step 06 uses the automatic candidate
flags from step 04; it does **not** read the completed manual review sheet
from step 05. Review and reconcile those exclusions before freezing a
research cohort. These scripts can rebuild their generated outputs.

The default root is this repository. Set `SR_SFCM_ROOT` to use a different
data/output root. For an already curated image directory, skip steps 01–06.

### 07: super-resolution

```bash
python scripts/07_run_realesrgan_x2.py --input data/final_images --output outputs/sr_x2 --weights weights/RealESRGAN_x2plus.pth
```

CUDA with half precision is the default, matching the supplied SR script.
Pass `--device cpu` for full-precision CPU inference. CPU inference is an
additional convenience path and has not been validated for paper results.
SR uses RRDBNet (64 features, 23 blocks, growth channels 32), scale 2,
tile 256, tile padding 20, and pre-padding 0.

### 08: candidate-region inference

```bash
python scripts/08_generate_candidate_regions_corrected.py --input outputs/sr_x2 --output outputs/candidate_regions
```

The default method is SLIC + spatial FCM. The optional `--method` values
`kmeans`, `fcm`, and `spatial-fcm` expose the corrected functional kernels
individually; no method-comparison experiment or statistical testing runs.
To process original-resolution images directly, supply that directory as
`--input`. The original step 09 processed `data/final_images`; SR and
original inputs are therefore explicit choices in this release. Selecting
SR inputs is not a claim that the original study used that path for every
experiment.

Both steps accept any nonempty image count and require unique file stems
across subdirectories. Both require an empty output directory to prevent
silent replacement. Any failed case produces a nonzero exit status and
an error entry in `run.json`.

Segmentation writes binary ROI/coating/body masks (PNG values 0/255), masked
candidate images, boundary overlays, and an operational run manifest.
Masks retain the input image dimensions. Coating and body partition the
extracted ROI. Convergence status is reported even when the maximum
iteration count is reached; a saved mask does not imply convergence.

## Fixed segmentation parameters

| Parameter | Value |
| --- | --- |
| Seed | 42 |
| Maximum working side | 384 pixels |
| Foreground threshold | maximum BGR channel > 12 |
| SLIC segments / compactness / sigma | 500 / 7.0 / 1.0 |
| Clusters / fuzzifier | 2 / 2.0 |
| Iteration limit / tolerance | 200 / 1e-5 |
| Spatial weight | 0.90 |
| Uncertain-membership threshold | 0.60 |

Inputs should already have a dark background outside the tongue; ROI
extraction thresholds the foreground and keeps its largest component.
Cluster semantics use the original lightness, chromaticity, saturation,
and radial-location scoring rule. These are candidate regions, not
ground-truth annotations or a diagnostic classifier.

## Release scope and provenance

The root-level and `scripts/` source copies were byte-identical. This
release includes one copy of steps 01–06, a portable SR runner adapted
from step 07, and inference functions extracted verbatim from the corrected
step 12. The original buggy step 10/12 implementations are excluded.
The SR wrapper retains network and tiling settings, adds explicit input,
output and weight arguments, removes the fixed 877-image check, and avoids
deleting previous outputs. No full-cohort numerical equivalence is claimed.

Excluded: inferential statistical analysis, ablation analysis, parameter/resolution sensitivity,
failure-case analysis, standalone paper tables/figures,
duplicate scripts, private image data, generated results, logs, model
weights, virtual environments, and IDE settings. No new scientific results
or paper citation metadata have been invented.

## Optional original-cohort and blind-review workflow

Install `requirements-review.txt` for Excel review packages and VLM clients.
These cohort-specific scripts retain their original 877-image/88-case checks.
Their roots also respect `SR_SFCM_ROOT`. Run them only after reviewing QC.

1. Run `09_generate_final_candidate_regions.py` for the original formal-D
   outputs. Its input is original `data/final_images`, not SR images.
2. Run `11_build_blind_validation_88.py` to freeze the 88-case sample.
3. Run `12_method_compare_corrected.py` to generate corrected A/B/C/D
   masks, structural diagnostics, and implementation audits. Its basic
   descriptive summaries are retained; separate statistical testing and
   paper plotting scripts are excluded. The old step 10/12 is not needed.
4. Run `14_build_blind_design.py` to construct the original balanced item
   mapping. Only the design construction/checks are retained from old step
   14; the old renderer and workbook step 15 are replaced by the corrected
   renderer below. An existing mapping is preserved rather than overwritten.
5. Run `15_generate_blind_review_corrected_fcm.py` to render and embed all
   352 corrected review images into a blank human-scoring workbook. Its
   comparison with old A/C/D masks is optional; old masks are not required.
6. If desired, run `16_dual_vlm_blind_review.py --provider qwen` (or
   `doubao`/`all`) for external VLM scoring. API credentials are read only
   from environment variables, never stored in this repository. Model and
   endpoint defaults are those of the supplied scripts. Running these
   clients sends review images to the selected provider and may incur fees;
   no provider calls are made during release verification.

For the SR-gain blind-review tools, first prepare their input masks without
running the excluded ablation statistics:

```bash
python scripts/08_prepare_sr_gain_masks.py --original data/final_images --sr outputs/sr_x2 --output results/08B_original_vs_sr_gain_ablation
python scripts/08b_check_convergence.py results/08B_original_vs_sr_gain_ablation/run.json
python scripts/18_build_sr_gain_human_blind_review_88.py
python scripts/19_build_sr_gain_vlm_blind_review_88.py
```

Step 08 uses the original step-08B segmentation kernels, original ROI
scaled to the SR dimensions, and original-dimension mask outputs. No
ablation test statistics are included. Steps 18/19 also require steps
09/11/14 above; step 18's references to prior human-D scores are only
mapping references, not evidence that any human scoring has been completed.
`20_dual_vlm_sr_gain_blind_review.py --provider qwen` optionally scores the
resulting paired review package. Use the same environment configuration as
step 16. Step 08b is adapted to the new JSON operational manifests rather
than the excluded sensitivity-analysis CSV.

The entire original-cohort/review workflow, Excel rendering, GPU SR, and
external VLM services have not been executed as part of release validation.
Preserve the directory structure and run the commands from the repository
root. Numbering gaps reflect excluded analyses and superseded scripts.

## Verification

```bash
python -m unittest discover -s tests -v
```

Checks cover genuine FCM updates, membership normalization, deterministic
inference, ROI partitioning, and CLI output/error handling with synthetic
images. Full original-cohort QC and GPU SR require the external data and
runtime and are not established by these checks.
