
"""
Topic‑word masking generator.

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from spacy.matcher import PhraseMatcher

from .common import get_nlp, PreprocessConfig, preprocess_text

# --------------------------------------------------------------------------- #
# Config & logger
# --------------------------------------------------------------------------- #
MAX_VARIANTS_PER_PARA = 6
TARGET_MASK_RATIO = 0.25  # guideline; not enforced strictly

LOGFMT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_phrase_matcher(nlp: "spacy.Language", keywords: List[str]) -> PhraseMatcher:
    """Create a PhraseMatcher with LOWER attribute for the given keywords."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(k) for k in keywords]
    matcher.add("TOPIC_KW", patterns)
    return matcher


def non_overlapping_spans(
    spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """Greedy longest‑first non‑overlap filter."""
    spans = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    chosen, occupied = [], []
    for s in spans:
        if all(not (s[0] < o1 < s[1] or s[0] < o2 < s[1]) for o1, o2 in occupied):
            chosen.append(s)
            occupied.append((s[0], s[1]))
    return sorted(chosen, key=lambda s: s[0])


def apply_mask(text: str, spans: List[Tuple[int, int]]) -> str:
    """Replace char‑index spans with <mask>; assumes spans non‑overlapping & sorted."""
    masked = text
    for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
        masked = masked[:start] + "<mask>" + masked[end:]
    # collapse duplicates
    return masked.replace("<mask> <mask>", "<mask>")


# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paragraph_json", required=True, 
                   help="processed_stories.json (will be treated as paragraph data)")
    p.add_argument("--model_dir", required=True, 
                   help="Directory containing topic_model.pkl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["extract", "match", "mask", "all"], default="all", 
                   help="Stage to run: extract (keywords), match (find keywords), mask (generate variants), all")
    return p.parse_args()


def stage_extract_keywords(model_dir: Path) -> None:
    """Stage 1: Extract topic keywords and save them."""
    logger.info("Stage 1: Extracting topic keywords...")
    
    # Check if we already have extended keywords (50 per topic)
    keywords_file = model_dir / "topic_keywords.json"
    if keywords_file.exists():
        with open(keywords_file, "r", encoding="utf-8") as f:
            existing_keywords = json.load(f)
        
        # Check if we have extended keywords (more than 10 per topic)
        first_topic_keywords = list(existing_keywords.values())[0] if existing_keywords else []
        if len(first_topic_keywords) >= 30:  # Already have extended keywords
            logger.info(f"Using existing extended keywords: {len(existing_keywords)} topics with {len(first_topic_keywords)} keywords each")
            return
    
    # Load BERTopic model for fallback
    topic_model: BERTopic = pickle.load(open(model_dir / "topic_model.pkl", "rb"))
    
    # Build keyword lookup - try to get more keywords
    topic_keywords = {}
    for t_id in topic_model.get_topics():
        if t_id != -1:
            topic_words = topic_model.get_topic(t_id)
            # Try to get up to 50 keywords, fall back to available
            keywords = [w for w, _ in topic_words[:50]] if topic_words else []
            topic_keywords[t_id] = keywords
    
    # Save keywords
    with open(keywords_file, "w", encoding="utf-8") as f:
        json.dump(topic_keywords, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Keywords extracted for {len(topic_keywords)} topics → {keywords_file}")


def stage_match_keywords(paragraphs: List[dict], model_dir: Path) -> None:
    """Stage 2: Find keyword matches in paragraphs and save spans."""
    logger.info("Stage 2: Matching keywords in paragraphs...")
    
    # Load topic assignments and keywords
    topics, _ = pickle.load(open(model_dir / "topic_assignments.pkl", "rb"))
    topic_keywords = json.load(open(model_dir / "topic_keywords.json", encoding="utf-8"))
    
    # Convert string keys back to int
    topic_keywords = {int(k): v for k, v in topic_keywords.items()}
    
    # Handle mismatch between paragraphs and topics (due to filtering in step1)
    if len(paragraphs) != len(topics):
        logger.warning(f"Mismatch: {len(paragraphs)} paragraphs vs {len(topics)} topics")
        
        # Filter paragraphs to match the ones used in step1
        from common import PreprocessConfig, preprocess_text
        cfg = PreprocessConfig(
            lowercase=True, 
            remove_stopwords=True, 
            lemmatize=True,
            min_word_length=3,
            remove_numbers=True,
            normalize_hyphens=True,
            min_alpha_ratio=0.7,
            remove_single_chars=True,
            normalize_contractions=True
        )
        
        filtered_paragraphs = []
        for para in paragraphs:
            cleaned_text = preprocess_text(para["paragraph_text"], cfg)
            if len(cleaned_text.split()) >= 10:  # Same filter as step1
                filtered_paragraphs.append(para)
        
        paragraphs = filtered_paragraphs
        logger.info(f"Filtered to {len(paragraphs)} paragraphs matching step1 processing")
        
        # Check if lengths match now
        if len(paragraphs) != len(topics):
            logger.error(f"Still mismatched: {len(paragraphs)} vs {len(topics)}")
            # Truncate to minimum length
            min_len = min(len(paragraphs), len(topics))
            paragraphs = paragraphs[:min_len]
            topics = topics[:min_len]
            logger.info(f"Truncated both to {min_len} items")
    
    # Convert topics to integer IDs if they are probability arrays
    if isinstance(topics[0], np.ndarray):
        topics = [np.argmax(t) if isinstance(t, np.ndarray) else t for t in topics]
        logger.info("Converted probability arrays to topic IDs")
    
    nlp = get_nlp()
    matcher_cache: dict[int, PhraseMatcher] = {}
    
    paragraph_spans = []
    
    for idx, para in enumerate(paragraphs):
        text = para["paragraph_text"]
        t_id = topics[idx]
        
        # Convert to int safely
        try:
            topic_id_int = int(t_id)
        except Exception as e:
            logger.warning(f"Error converting t_id at index {idx}: {t_id}, error: {e}")
            topic_id_int = -1
        
        spans_data = {
            "paragraph_id": para.get("paragraph_id", idx),
            "topic_id": topic_id_int,
            "spans": []
        }
        
        if topic_id_int != -1 and topic_id_int in topic_keywords:
            # Build matcher for this topic
            if topic_id_int not in matcher_cache:
                matcher_cache[topic_id_int] = build_phrase_matcher(nlp, topic_keywords[topic_id_int])
            matcher = matcher_cache[topic_id_int]
            
            doc = nlp(text)
            matches = matcher(doc)
            
            spans = [
                (doc[start].idx, doc[end - 1].idx + len(doc[end - 1]), doc[start:end].text)
                for _, start, end in matches
            ]
            spans = non_overlapping_spans(spans)
            spans_data["spans"] = spans
        
        paragraph_spans.append(spans_data)
    
    # Save spans
    spans_file = model_dir / "paragraph_spans.json"
    with open(spans_file, "w", encoding="utf-8") as f:
        json.dump(paragraph_spans, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Keyword spans saved → {spans_file}")


def stage_generate_masks(paragraphs: List[dict], model_dir: Path) -> None:
    """Stage 3: Generate masked variants using saved spans."""
    logger.info("Stage 3: Generating masked variants...")
    
    # Apply same filtering as in stage_match_keywords
    from common import PreprocessConfig, preprocess_text
    cfg = PreprocessConfig(
        lowercase=True, 
        remove_stopwords=True, 
        lemmatize=True,
        min_word_length=3,
        remove_numbers=True,
        normalize_hyphens=True,
        min_alpha_ratio=0.7,
        remove_single_chars=True,
        normalize_contractions=True
    )
    
    filtered_paragraphs = []
    for para in paragraphs:
        cleaned_text = preprocess_text(para["paragraph_text"], cfg)
        if len(cleaned_text.split()) >= 10:
            filtered_paragraphs.append(para)
    
    paragraphs = filtered_paragraphs
    logger.info(f"Using {len(paragraphs)} filtered paragraphs for masking")
    
    # Load spans
    paragraph_spans = json.load(open(model_dir / "paragraph_spans.json", encoding="utf-8"))
    
    # Ensure spans match filtered paragraphs
    if len(paragraphs) != len(paragraph_spans):
        logger.warning(f"Mismatch: {len(paragraphs)} paragraphs vs {len(paragraph_spans)} spans")
        min_len = min(len(paragraphs), len(paragraph_spans))
        paragraphs = paragraphs[:min_len]
        paragraph_spans = paragraph_spans[:min_len]
        logger.info(f"Truncated both to {min_len} items")
    
    masked_rows, report_counts = [], Counter()
    
    for idx, para in enumerate(paragraphs):
        text = para["paragraph_text"]
        spans_data = paragraph_spans[idx]
        spans = spans_data["spans"]
        t_id = spans_data["topic_id"]
        
        if not spans:
            # No keywords found, keep original
            variant = {
                **para,
                "variant_id": f"{uuid.uuid4()}",
                "topic_id": t_id,
                "masked_tokens": [],
                "masked_text": text,
                "offsets": [],
            }
            masked_rows.append(variant)
            continue
        
        # Dynamic mask count
        nlp = get_nlp()
        doc = nlp(text)
        n_tok = len(doc)
        max_mask = max(1, round(math.sqrt(n_tok)))
        max_mask = min(max_mask, len(spans))
        
        # Generate variants
        variants_created = 0
        while variants_created < MAX_VARIANTS_PER_PARA:
            k = random.randint(1, max_mask)
            sel = random.sample(spans, k=k)
            sel_sorted = sorted(sel, key=lambda s: s[0])
            masked_text = apply_mask(text, [(s, e) for s, e, _ in sel_sorted])
            
            variant = {
                **para,
                "variant_id": f"{uuid.uuid4()}",
                "topic_id": t_id,
                "masked_tokens": [kw for *_, kw in sel_sorted],
                "masked_text": masked_text,
                "offsets": [(s, e) for s, e, _ in sel_sorted],
            }
            masked_rows.append(variant)
            variants_created += 1
            report_counts["variants"] += 1
        
        report_counts["paragraphs"] += 1
        report_counts["tokens_masked_total"] += sum(
            len(v["masked_tokens"]) for v in masked_rows[-variants_created:]
        )
    
    # Save outputs
    out_csv = model_dir / "topic_masked_paragraphs.csv.gz"
    out_jsonl = model_dir / "topic_masked_paragraphs.jsonl"
    pd.DataFrame(masked_rows).to_csv(out_csv, compression="gzip", index=False)
    
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in masked_rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")
    
    # Quality report
    avg_masks = report_counts["tokens_masked_total"] / max(report_counts["variants"], 1)
    report = {
        "paragraphs_processed": report_counts["paragraphs"],
        "variants_generated": report_counts["variants"],
        "avg_masks_per_variant": round(avg_masks, 2),
    }
    (model_dir / "masking_report.json").write_text(json.dumps(report, indent=2))
    
    logger.info(
        "Masking done: %d variants over %d paragraphs  |  avg masked tokens %.2f",
        report_counts["variants"],
        report_counts["paragraphs"],
        avg_masks,
    )
    logger.info("Saved CSV → %s", out_csv)
    logger.info("Saved JSONL → %s", out_jsonl)
    logger.info("Report → %s", model_dir / "masking_report.json")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    
    model_dir = Path(args.model_dir)
    
    if args.stage in ["extract", "all"]:
        stage_extract_keywords(model_dir)
    
    if args.stage in ["match", "mask", "all"]:
        # Load paragraph data
        paragraphs = json.load(open(args.paragraph_json, encoding="utf-8"))
        logger.info(f"Loaded {len(paragraphs)} paragraphs")
        
        if args.stage in ["match", "all"]:
            stage_match_keywords(paragraphs, model_dir)
        
        if args.stage in ["mask", "all"]:
            stage_generate_masks(paragraphs, model_dir)