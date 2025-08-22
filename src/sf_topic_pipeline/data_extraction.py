# step0_json_creation.py  (v2 ‑ 2025‑07‑07)

"""
Stage 0: read raw Amazing‑Stories HTML files → extract story‑level metadata →
paragraph segmentation → save full corpus JSON **and** train/dev/test splits.

Key improvements over v1
------------------------
1.  logging instead of print
2.  optional --skip_errors flag
3.  SHA‑256 hash per raw file for provenance
4.  crude duplicate filtering (exact content hash)
5.  simple time‑based train/dev/test split:
        • newest 20 % ⇒ test
        • next 15 %  ⇒ dev
        • rest       ⇒ train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from .common import get_sorted_file_entries, process_texts_df
from transformers import AutoTokenizer
import nltk
import nltk.data

LOGFMT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT)
logger = logging.getLogger(__name__)

# Download punkt tokenizer for sentence splitting
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # New version requirement
    # Initialize tokenizers for paragraph segmentation
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    logger.info("Tokenizers initialized successfully")
except Exception as e:
    logger.error("Failed to initialize tokenizers: %s", e)
    raise

def segment_paragraphs(story_entries: List[dict], max_tokens: int = 400) -> List[dict]:
    """
    Segment each story's text into non-overlapping chunks of up to max_tokens BERT tokens,
    preserving sentence boundaries and metadata.
    
    Parameters
    ----------
    story_entries : List[dict]
        List of story dictionaries with 'text' key containing full story text
    max_tokens : int, default=400
        Maximum number of BERT tokens per paragraph chunk
        
    Returns
    -------
    List[dict]
        List of paragraph dictionaries with 'paragraph_text' and 'paragraph_index' keys
    """
    paragraphs = []
    logger.info("Starting paragraph segmentation for %d stories", len(story_entries))
    
    for entry_idx, entry in enumerate(story_entries):
        if "full_text" not in entry:
            logger.warning("Story entry %d missing 'full_text' key, skipping", entry_idx)
            continue
            
        try:
            sentences = sentence_tokenizer.tokenize(entry["full_text"])
        except Exception as e:
            logger.error("Failed to tokenize story entry %d: %s", entry_idx, e)
            continue
        current_chunk = []
        current_len = 0
        para_idx = 0
        
        for sentence in sentences:
            try:
                count = len(tokenizer.tokenize(sentence))
            except Exception as e:
                logger.warning("Failed to tokenize sentence in story %d: %s", entry_idx, e)
                continue
                
            if current_chunk and current_len + count > max_tokens:
                # finalize current chunk
                para_text = " ".join(current_chunk).strip()
                para_entry = {k: entry[k] for k in entry if k != "full_text"}
                para_entry.update({"paragraph_index": para_idx, "paragraph_text": para_text})
                paragraphs.append(para_entry)
                para_idx += 1
                current_chunk = [sentence]
                current_len = count
            else:
                current_chunk.append(sentence)
                current_len += count
                
        # add last chunk
        if current_chunk:
            para_text = " ".join(current_chunk).strip()
            para_entry = {k: entry[k] for k in entry if k != "full_text"}
            para_entry.update({"paragraph_index": para_idx, "paragraph_text": para_text})
            paragraphs.append(para_entry)
            
    logger.info("Paragraph segmentation complete: %d paragraphs created", len(paragraphs))
    return paragraphs


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _sha256(path: str | Path) -> str:
    """Return SHA‑256 of a file (binary read, 64 kB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_all_files(base_dir: str, skip_errors: bool = False) -> List[dict]:
    """
    Scan *base_dir* for files that match Amazing‑Stories pattern and read them.

    Returns
    -------
    list[dict]
        Each dict contains volume/issue/year/month/text/raw_hash.
    """
    file_entries = get_sorted_file_entries(base_dir)
    logger.info("Found %d files under %s", len(file_entries), base_dir)

    data: List[dict] = []
    seen_hashes: set[str] = set()  # duplicate detection

    for y, m, vol, iss, fname in tqdm(file_entries, desc="Reading files"):
        path = Path(base_dir) / fname
        try:
            raw_text = path.read_text(encoding="utf‑8")
            file_hash = _sha256(path)

            if file_hash in seen_hashes:
                logger.warning("Duplicate file skipped: %s", fname)
                continue
            seen_hashes.add(file_hash)

            data.append(
                {
                    "volume_number": vol,
                    "issue_number": iss,
                    "publication_year": y,
                    "publication_month": m,
                    "raw_hash": file_hash,
                    "text": raw_text,
                }
            )
        except Exception as exc:
            msg = f"Failed to read {path}: {exc}"
            if skip_errors:
                logger.error(msg)
                continue
            raise RuntimeError(msg) from exc

    logger.info("Loaded %d unique files", len(data))
    return data


def split_time_based(entries: List[dict]) -> tuple[List[dict], List[dict], List[dict]]:
    """
    Split paragraph entries into train/dev/test by publication date percentile.
    Newest 20 % → test, next 15 % → dev, remainder → train.
    """
    sorted_e = sorted(
        entries,
        key=lambda e: (e["publication_year"], e["publication_month"]),
    )
    n = len(sorted_e)
    test_cut = int(n * 0.8)
    dev_cut = int(n * 0.65)

    train, dev, test = sorted_e[:dev_cut], sorted_e[dev_cut:test_cut], sorted_e[test_cut:]
    logger.info("Split: train=%d  dev=%d  test=%d", len(train), len(dev), len(test))
    return train, dev, test


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="Folder with raw HTML/text files")
    p.add_argument("--output_dir", required=True, help="Where to dump JSON outputs")
    p.add_argument("--skip_errors", action="store_true", help="Skip unreadable files")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load raw
    raw_file_records = read_all_files(str(base_dir), skip_errors=args.skip_errors)

    # 2. story‑level extraction
    story_entries = process_texts_df(raw_file_records)

    # 3. save story-level corpus (full text per story)
    story_corpus_path = out_dir / "processed_stories_by_story.json"
    story_corpus_path.write_text(json.dumps(story_entries, ensure_ascii=False, indent=2))
    logger.info("Story-level corpus saved → %s  (stories=%d)", story_corpus_path, len(story_entries))

    # 4. paragraph segmentation
    paragraphs = segment_paragraphs(story_entries)
    
    # Validate paragraph structure
    if paragraphs:
        sample_para = paragraphs[0]
        required_keys = ["paragraph_text", "paragraph_index"]
        missing_keys = [k for k in required_keys if k not in sample_para]
        if missing_keys:
            logger.error("Missing required keys in paragraph data: %s", missing_keys)
            raise ValueError(f"Invalid paragraph structure: missing {missing_keys}")
        logger.info("Paragraph structure validation passed")
    else:
        logger.warning("No paragraphs generated - check input data")

    # 5. save paragraph-level corpus
    para_corpus_path = out_dir / "processed_stories_by_paragraph.json"
    para_corpus_path.write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2))
    logger.info("Paragraph-level corpus saved → %s  (paragraphs=%d)", para_corpus_path, len(paragraphs))

    # 6. split + save subsets (paragraph-based)
    train, dev, test = split_time_based(paragraphs)
    (out_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2))
    (out_dir / "dev.json").write_text(json.dumps(dev, ensure_ascii=False, indent=2))
    (out_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2))

    logger.info("step0 complete; outputs in %s", out_dir)
    
    # Print summary statistics
    logger.info("="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info("Raw files processed: %d", len(raw_file_records))
    logger.info("Stories extracted: %d", len(story_entries))
    logger.info("Paragraphs created: %d", len(paragraphs))
    logger.info("Train/Dev/Test split: %d/%d/%d", len(train), len(dev), len(test))
    
    # Calculate average tokens per paragraph
    if paragraphs:
        avg_tokens = sum(len(tokenizer.tokenize(p["paragraph_text"])) for p in paragraphs[:100]) / min(100, len(paragraphs))
        logger.info("Average tokens per paragraph (sample): %.1f", avg_tokens)
    
    logger.info("="*50)


if __name__ == "__main__":
    main()