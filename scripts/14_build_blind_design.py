"""Build only the original balanced blind design; step 15 renders corrected masks."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("SR_SFCM_ROOT", Path(__file__).resolve().parents[1]))
METHODS = ["A", "B", "C", "D"]
METHOD_NAMES = {"A":"KMeans", "B":"FCM", "C":"Spatial_FCM", "D":"SLIC_Spatial_FCM"}
EXPECTED_CASES = 88
EXPECTED_ITEMS = 352
BATCH_ASSIGNMENT_SEED = 44
BATCH_ORDER_SEEDS = {1:101, 2:102, 3:103, 4:104}
blind_df = pd.read_csv(PROJECT_ROOT / "results/11_blind_validation_88/blind_validation_88_INTERNAL_mapping.csv", dtype=str)
blind_df = blind_df.sort_values("blind_id").reset_index(drop=True)
if len(blind_df) != 88 or blind_df.image_id.nunique() != 88 or blind_df.blind_id.nunique() != 88:
    raise ValueError("Expected 88 distinct frozen cases and blind IDs")
rng = np.random.default_rng(
    BATCH_ASSIGNMENT_SEED
)

offsets = np.repeat(
    np.arange(
        4,
        dtype=int
    ),
    EXPECTED_CASES // 4
)

rng.shuffle(
    offsets
)

if len(offsets) != EXPECTED_CASES:

    raise RuntimeError(
        "offset数量异常。"
    )

design_rows = []

for case_index, row in blind_df.iterrows():

    offset = int(
        offsets[
            case_index
        ]
    )

    for batch in range(
        1,
        5
    ):

        method_index = (
            (batch - 1 + offset)
            %
            4
        )

        method = (
            METHODS[
                method_index
            ]
        )

        design_rows.append(
            {
                "blind_id":
                    row[
                        "blind_id"
                    ],

                "image_id":
                    row[
                        "image_id"
                    ],

                "batch":
                    batch,

                "method":
                    method,

                "method_name":
                    METHOD_NAMES[
                        method
                    ],
            }
        )

design_df = pd.DataFrame(
    design_rows
)

ordered_batches = []

for batch in range(
    1,
    5
):

    current = (
        design_df[
            design_df[
                "batch"
            ]
            ==
            batch
        ]
        .copy()
    )

    current = (
        current
        .sample(
            frac=1.0,
            random_state=
                BATCH_ORDER_SEEDS[
                    batch
                ]
        )
        .reset_index(
            drop=True
        )
    )

    current[
        "batch_order"
    ] = np.arange(
        1,
        len(current) + 1
    )

    ordered_batches.append(
        current
    )

design_df = pd.concat(
    ordered_batches,
    axis=0,
    ignore_index=True
)

design_df[
    "item_id"
] = [
    f"HBR{i:03d}"
    for i in range(
        1,
        EXPECTED_ITEMS + 1
    )
]

if len(
    design_df
) != EXPECTED_ITEMS:

    raise RuntimeError(
        f"盲评项目数量异常："
        f"{len(design_df)}"
    )

case_method_check = (
    design_df
    .groupby(
        "blind_id"
    )["method"]
    .agg(
        lambda x:
        ",".join(
            sorted(
                x
            )
        )
    )
)

if not (
    case_method_check
    ==
    "A,B,C,D"
).all():

    raise RuntimeError(
        "存在病例未完整分配A/B/C/D四种方法。"
    )

batch_counts = (
    design_df[
        "batch"
    ]
    .value_counts()
    .sort_index()
)

if not (
    batch_counts
    ==
    EXPECTED_CASES
).all():

    raise RuntimeError(
        "4个batch数量不平衡。"
    )

batch_method_table = pd.crosstab(
    design_df[
        "batch"
    ],
    design_df[
        "method"
    ]
).reindex(
    index=[
        1,
        2,
        3,
        4
    ],
    columns=METHODS,
    fill_value=0
)

expected_per_method_batch = (
    EXPECTED_CASES
    //
    4
)

if not (
    batch_method_table
    ==
    expected_per_method_batch
).all().all():

    raise RuntimeError(
        "每个batch内A/B/C/D未达到22/22/22/22平衡。"
    )

output = PROJECT_ROOT / "results/14_human_blind_review_88/human_review_INTERNAL_mapping.csv"
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError(f"Refusing to replace a frozen design: {output}")
design_df.to_csv(output, index=False, encoding="utf-8-sig")
print(f"Saved {len(design_df)} blinded design items to {output}")
