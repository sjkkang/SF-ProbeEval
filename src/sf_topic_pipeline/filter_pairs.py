
"""
filter_pairs.py
---------------
Quality-filter `infilled_pairs_with_hybrid.jsonl` (Step 4-A output) and
produce a cleaned file `clean_pairs.jsonl` for downstream sampling / training.

Filtering rules
---------------
A. Residual mask tokens       : drop records whose `infilled` string still
                                contains "<mask>" or "<extra_id_X>".
B. Infill log-prob threshold  : drop if `infill_score` is None OR < TH_LP.
C. Hybrid similarity threshold: drop if `hybrid_score` < TH_HYB.
D. Length bounds              : keep only `min_len ≤ length ≤ max_len`.
E. Deduplication              : for duplicate `pair_id`, keep the row with
                                highest `hybrid_score`.

All thresholds are configurable via CLI flags.

Usage
-----
python filter_pairs.py \
    --input  output/step4_infill/infilled_pairs_with_hybrid.jsonl \
    --output output/step4_infill/clean_pairs.jsonl \
    --th_lp   -3.0   \
    --th_hyb   0.3   \
    --min_len  4     \
    --max_len  180
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

LOGFMT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT)
logger = logging.getLogger(__name__)

# Regex to catch any residual mask tokens (<mask>, <extra_id_X>, broken masks, etc.)
MASK_RE = re.compile(r"<mask>|<extra_id_\d+>|<\w*mask\w*>|<\w*sk\w*>", flags=re.IGNORECASE)


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    logger.info("Loading: %s", path)
    try:
        df = pd.read_json(path, lines=True)
        logger.info("Loaded %d records", len(df))
        return df
    except Exception as e:
        logger.error("Failed to load file %s: %s", path, e)
        raise


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to JSON Lines."""
    logger.info("Saving cleaned file → %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def apply_filters(
    df: pd.DataFrame,
    th_lp: float,
    th_hyb: float,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    """Apply the five filtering rules to the DataFrame."""
    n_orig = len(df)

    # A. residual <mask> tokens
    df = df[~df["infilled"].str.contains(MASK_RE)]

    # B. infill_score threshold
    df = df[df["infill_score"].fillna(-9.9) >= th_lp]

    # C. hybrid_score threshold
    df = df[df["hybrid_score"] >= th_hyb]

    # D. length bounds
    df = df[(df["length"] >= min_len) & (df["length"] <= max_len)]

    # E. deduplicate by pair_id, keep highest hybrid_score
    before_dedup = len(df)
    df = (
        df.sort_values("hybrid_score", ascending=False)
        .drop_duplicates(subset="pair_id", keep="first")
    )

    logger.info(
        "Filtering summary: original=%d  after_filters=%d  after_dedup=%d",
        n_orig,
        before_dedup,
        len(df),
    )
    return df.reset_index(drop=True)


def describe_histogram(df: pd.DataFrame, bins: List[float] | None = None) -> None:
    """Print a simple bucket count of hybrid_score for quick sanity check."""
    if bins is None:
        bins = [0.0, 0.3, 0.7, 1.01]
    labels = ["low", "mid", "high"]
    counts = pd.cut(df["hybrid_score"], bins=bins, labels=labels).value_counts()
    logger.info("Hybrid-score bucket counts:\n%s", counts.sort_index().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File with hybrid_score field")
    parser.add_argument("--output", required=True, help="Destination JSONL")
    parser.add_argument("--th_lp", type=float, default=-3.0, help="Min infill_score")
    parser.add_argument("--th_hyb", type=float, default=0.3, help="Min hybrid_score")
    parser.add_argument("--min_len", type=int, default=4, help="Minimum sentence length")
    parser.add_argument("--max_len", type=int, default=180, help="Maximum sentence length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_jsonl(Path(args.input))
    df_clean = apply_filters(
        df,
        th_lp=args.th_lp,
        th_hyb=args.th_hyb,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    describe_histogram(df_clean)
    save_jsonl(df_clean, Path(args.output))


if __name__ == "__main__":
    main()