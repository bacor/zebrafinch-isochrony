"""Rows, colors and names shared by the Figure 2 notebooks (fig2*.ipynb).

The layout helpers themselves live in `src/viz/rows.py`.
"""

import seaborn as sns

from constants import RHYTHM_TYPES
from viz.palette import white_to_color_cmap
from viz.rows import flatten_sections

# Rows of Figure 2, as (datasets pooled in the row, rhythm type), grouped by
# which label types the datasets have.
SECTIONS = [
    {
        "title": "RT & BWPS",
        "rows": [(["rt", "bwps-clean"], "syll_ioi")],
    },
    {
        "title": "NS & BVS",
        "rows": [
            (["ns", "bvs"], "syll_ioi"),
            (["ns", "bvs"], "syll_dur"),
            (["ns", "bvs"], "gap_dur"),
        ],
    },
    {
        "title": "BVS only",
        "rows": [
            (["bvs"], "elem_ioi"),
            (["bvs"], "elem_dur"),
        ],
    },
]


def build_row_specs(sections=SECTIONS, wrap=False):
    """`viz.rows.flatten_sections` of ``sections``, labeling each row by its rhythm type.

    ``wrap`` puts the label on two lines ("Syllable\\nIOIs").
    """

    def row_label(row):
        label = RHYTHM_TYPES[row[1]]["label"]
        return label.replace(" ", "\n") if wrap else label

    return flatten_sections(sections, row_label)


# Colors
PALETTE = sns.color_palette("deep")
ANNOT_COLOR = PALETTE[0]
PRED_COLOR = PALETTE[1]
SHUFFLE_COLOR = ".6"
SAMPLE_COLOR = PALETTE[4]
SERIES_COLORS = {
    "annot": ANNOT_COLOR,
    "pred": PRED_COLOR,
    "shuffle": SHUFFLE_COLOR,
}

# White-to-color colormaps, for shading within a label type's own color.
ANNOT_CMAP = white_to_color_cmap(ANNOT_COLOR, "annot")
PRED_CMAP = white_to_color_cmap(PRED_COLOR, "pred")

# Legend name of each label type.
LABEL_TYPE_NAMES = {
    "annot": "Annotated labels",
    "pred": "Predicted labels",
    "shuffle": "Shuffled labels",
}
