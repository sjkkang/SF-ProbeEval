# step3_pair_generation.py  
"""
Sentence‑level pair generation with quality filtering.

Input
-----
• topic_masked_paragraphs.jsonl   (step2 output, line‑wise dict)
• topic_keywords.json (optional)  {topic_id: [kw1, …]}

Output
------
• masked_pairs.jsonl   — clean pairs (mask_ratio & length meta)
• pairs_stats.json     — global statistics after filtering

Filtering Rules (default)
-------------------------
✓ min_tokens ≥ 4  AND  at least one VERB  
✓ 0.05 ≤ mask_ratio ≤ 0.4  
✓ not heading/footer regex  (e.g. CHAPTER I, THE END)  
✓ sentence not entirely uppercase (length ≤ 6)  
✓ duplicate (original, masked) pairs removed
CLI flags let you change thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

import spacy

from common import get_nlp

# ───────────────────────────────────────────
# Logger
# ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────
# Regex patterns
# ───────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(
    r"^(CHAPTER|BOOK|PART|SECTION|THE END|END)\b[\s\W\d]*$", re.IGNORECASE
)

# ───────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────
def fast_sent_split(text: str) -> List[str]:
    return _SENT_SPLIT_RE.split(text.strip())


def calc_mask_ratio(masked: str) -> float:
    tok = masked.split()
    return 0.0 if not tok else tok.count("<mask>") / len(tok)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def is_heading(sent: str) -> bool:
    return bool(_HEADING_RE.match(sent.strip()))


def is_all_caps_short(sent: str, max_len: int = 6) -> bool:
    t = sent.strip()
    return len(t.split()) <= max_len and t.isupper()


def has_verb(doc: spacy.tokens.Doc) -> bool:
    return any(t.pos_ == "VERB" for t in doc)


# ───────────────────────────────────────────
# Core
# ───────────────────────────────────────────
def run_pair_generation(
    masked_jsonl: Path,
    out_pairs: Path,
    out_stats: Path,
    topic_kw_path: Path | None,
    *,
    min_tokens: int = 4,
    mask_min: float = 0.05,
    mask_max: float = 0.40,
) -> None:
    # load variants
    variants = list(iter_jsonl(masked_jsonl))
    logger.info("Loaded %d masked variants.", len(variants))

    # topic‑specific vocab (optional)
    topic_vocab: dict[int, set[str]] = defaultdict(set)
    if topic_kw_path and topic_kw_path.exists():
        topic_vocab = {
            int(k): set(map(str.lower, v))
            for k, v in json.load(open(topic_kw_path, encoding="utf-8")).items()
        }
        logger.info("Loaded topic keywords for %d topics.", len(topic_vocab))

    nlp = get_nlp()
    pairs, stats = [], Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for var in variants:
        # Handle different data formats
        if "paragraph_text" in var:
            o_para, m_para = var["paragraph_text"], var["masked_text"]
        elif "original_text" in var:
            o_para, m_para = var["original_text"], var["masked_text"]
        else:
            stats["no_text_field"] += 1
            continue
            
        tid = int(var.get("topic_id", -1))

        for o_raw, m_raw in zip(fast_sent_split(o_para), fast_sent_split(m_para)):
            if "<mask>" not in m_raw:
                continue

            doc_o = nlp(o_raw)
            doc_m = nlp(m_raw)
            o_sent, m_sent = doc_o.text.strip(), doc_m.text.strip()

            # ───── filtering ─────
            if len(doc_o) < min_tokens:
                continue
            if not has_verb(doc_o):
                continue
            if is_heading(o_sent) or is_all_caps_short(o_sent):
                continue

            r = calc_mask_ratio(m_sent)
            if r < mask_min or r > mask_max:
                continue

            if topic_vocab and tid in topic_vocab:
                if not any(tok.lower_ in topic_vocab[tid] for tok in doc_o):
                    continue

            key = (o_sent, m_sent)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            # ───── add pair ─────
            pairs.append(
                {
                    "pair_id": str(uuid.uuid4()),
                    "topic_id": tid,
                    "original": o_sent,
                    "masked": m_sent,
                    "mask_ratio": round(r, 3),
                    "length": len(doc_o),
                }
            )
            stats["pairs"] += 1
            stats["tokens"] += len(doc_o)
            stats["masks"] += m_sent.split().count("<mask>")

    # save outputs
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pairs, "w", encoding="utf-8") as fh:
        for p in pairs:
            json.dump(p, fh, ensure_ascii=False)
            fh.write("\n")
    logger.info("Saved %d clean pairs → %s", len(pairs), out_pairs)

    stats_dict = {
        "total_pairs": stats["pairs"],
        "avg_len": round(stats["tokens"] / max(stats["pairs"], 1), 2),
        "avg_mask_ratio": round(stats["masks"] / max(stats["tokens"], 1), 3),
        "unique_pairs_removed": len(seen_pairs) - stats["pairs"],
    }
    out_stats.write_text(json.dumps(stats_dict, indent=2))
    logger.info("Stats → %s  %s", out_stats, stats_dict)


# ───────────────────────────────────────────
# CLI
# ───────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--masked_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--topic_keywords", default=None)

    # filter hyper‑parameters
    p.add_argument("--min_tokens", type=int, default=4)
    p.add_argument("--mask_min", type=float, default=0.05)
    p.add_argument("--mask_max", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Auto-detect improved topics if available
    masked_jsonl_path = Path(args.masked_jsonl)
    if not masked_jsonl_path.exists():
        # Check if we should use improved topics
        improved_path = masked_jsonl_path.parent.parent / "improved_topics" / "topic_masked_paragraphs.jsonl"
        if improved_path.exists():
            logger.info(f"Using improved topic masked data from {improved_path}")
            masked_jsonl_path = improved_path
    
    # Auto-detect topic keywords path
    topic_kw_path = None
    if args.topic_keywords:
        topic_kw_path = Path(args.topic_keywords)
    else:
        # Try to find topic keywords in the same directory
        potential_kw_path = masked_jsonl_path.parent / "topic_keywords.json"
        if potential_kw_path.exists():
            topic_kw_path = potential_kw_path
            logger.info(f"Auto-detected topic keywords at {topic_kw_path}")

    run_pair_generation(
        masked_jsonl=masked_jsonl_path,
        out_pairs=Path(args.output_dir) / "masked_pairs.jsonl",
        out_stats=Path(args.output_dir) / "pairs_stats.json",
        topic_kw_path=topic_kw_path,
        min_tokens=args.min_tokens,
        mask_min=args.mask_min,
        mask_max=args.mask_max,
    )


if __name__ == "__main__":
    main()