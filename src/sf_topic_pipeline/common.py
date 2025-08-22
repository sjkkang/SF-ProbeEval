"""
common.py (v2, 2025‑07‑07)

Utility functions shared by every stage of the pipeline.
The module is I/O‑free on purpose: no file reading/writing happens here.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import nltk
import spacy
from nltk.corpus import stopwords as _nltk_sw
from nltk.stem import WordNetLemmatizer

# ────────────────────────────────────────────────────────────────────────────
# Public symbols
# ────────────────────────────────────────────────────────────────────────────
__all__ = [
    "PreprocessConfig",
    "get_nlp",
    "get_lemmatizer",
    "build_stopwords",
    "preprocess_text",
    "extract_story_metadata",
    "process_texts_df",
    "segment_paragraphs",
    "get_sorted_file_entries",
]

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Lazy singletons
# ────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_nlp() -> "spacy.Language":
    """Load spaCy once and memo‑ize the result."""
    logger.debug("Loading spaCy model (en_core_web_sm)")
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    """Return a memo‑ized WordNetLemmatizer instance."""
    return WordNetLemmatizer()


# ────────────────────────────────────────────────────────────────────────────
# Pre‑processing configuration
# ────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PreprocessConfig:
    """Config switches for text preprocessing."""
    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    allowed_pos: set[str] | None = field(default_factory=lambda: None)  # e.g. {"NOUN", "ADJ"}
    min_word_length: int = 2  # Minimum word length to keep
    remove_numbers: bool = False  # Remove standalone numbers
    remove_punctuation: bool = True  # Remove punctuation
    remove_short_sentences: bool = False  # Remove very short sentences
    
    # Enhanced options for better text cleaning
    normalize_hyphens: bool = True  # Convert hyphenated words like "to-morrow" to "tomorrow"
    min_alpha_ratio: float = 0.5  # Minimum ratio of alphabetic characters in a word
    remove_single_chars: bool = True  # Remove single character words (except 'a', 'i')
    normalize_contractions: bool = True  # Normalize contractions like "won't" to "will not"

    def __post_init__(self) -> None:
        if self.allowed_pos is not None:
            self.allowed_pos = {p.upper() for p in self.allowed_pos}


# ────────────────────────────────────────────────────────────────────────────
# Stopword handling
# ────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def build_stopwords(extra_file: str | os.PathLike | None = None) -> set[str]:
    """
    Combine NLTK stopwords with domain‑specific terms (Amazing Stories jargon).

    Parameters
    ----------
    extra_file : path or None
        Optional newline‑separated list with additional stopwords.

    Returns
    -------
    Set[str]
        The merged stopword vocabulary (already lower‑cased).
    """
    # Ensure required corpora are available
    for corp in ("stopwords", "wordnet"):
        try:
            nltk.data.find(f"corpora/{corp}")
        except LookupError:
            nltk.download(corp, quiet=True)

    base_sw = set(_nltk_sw.words("english"))
    domain_sw = {"amazing", "stories", "volume", "issue"}

    if extra_file is not None and Path(extra_file).is_file():
        with open(extra_file, encoding="utf‑8") as fh:
            domain_sw |= {w.strip().lower() for w in fh if w.strip()}

    merged = {w.lower() for w in (base_sw | domain_sw)}
    logger.info("Stopword list built – size=%d", len(merged))
    return merged


# ────────────────────────────────────────────────────────────────────────────
# Enhanced text preprocessing utilities
# ────────────────────────────────────────────────────────────────────────────
def normalize_text_enhanced(text: str, cfg: PreprocessConfig) -> str:
    """Apply enhanced text normalization before spaCy processing."""
    if not text or not text.strip():
        return ""
    
    # Start with the original text
    normalized = text
    
    # 1. Normalize contractions
    if cfg.normalize_contractions:
        contraction_map = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'ll": " will", "'re": " are", "'ve": " have", "'d": " would",
            "'m": " am", "'s": " is"  # This handles possessives too, but that's ok
        }
        for contraction, expansion in contraction_map.items():
            normalized = re.sub(rf"\b\w*{re.escape(contraction)}\b", 
                              lambda m: m.group().replace(contraction, expansion), 
                              normalized, flags=re.IGNORECASE)
    
    # 2. Normalize hyphenated words
    if cfg.normalize_hyphens:
        # Convert common hyphenated words to single words
        # Examples: to-morrow -> tomorrow, to-day -> today, etc.
        common_hyphens = {
            "to-morrow": "tomorrow", "to-day": "today", "to-night": "tonight",
            "every-thing": "everything", "some-thing": "something", 
            "any-thing": "anything", "every-one": "everyone",
            "some-one": "someone", "any-one": "anyone",
            "every-where": "everywhere", "some-where": "somewhere",
            "any-where": "anywhere", "no-thing": "nothing",
            "no-one": "noone", "no-where": "nowhere"
        }
        
        for hyphenated, normalized_word in common_hyphens.items():
            normalized = re.sub(rf"\b{re.escape(hyphenated)}\b", normalized_word, 
                              normalized, flags=re.IGNORECASE)
        
        # For other hyphenated words, be more careful
        # Only join if both parts are real words (length >= 3)
        hyphen_pattern = r'\b([a-zA-Z]{3,})-([a-zA-Z]{3,})\b'
        normalized = re.sub(hyphen_pattern, r'\1\2', normalized)
    
    # 3. Clean up extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized.strip())
    
    return normalized


def is_valid_word(word: str, cfg: PreprocessConfig) -> bool:
    """Check if a word should be kept based on enhanced criteria."""
    if not word or len(word) < cfg.min_word_length:
        return False
    
    # Remove single characters except meaningful ones
    if cfg.remove_single_chars and len(word) == 1 and word.lower() not in {'a', 'i'}:
        return False
    
    # Check minimum alphabetic ratio
    if cfg.min_alpha_ratio > 0:
        alpha_count = sum(1 for c in word if c.isalpha())
        if alpha_count / len(word) < cfg.min_alpha_ratio:
            return False
    
    # Remove words that are mostly punctuation or numbers
    if len(word) > 1:
        non_alpha_count = sum(1 for c in word if not c.isalpha())
        if non_alpha_count > len(word) * 0.5:  # More than 50% non-alphabetic
            return False
    
    # Remove obvious artifacts (sequences of same character)
    if len(set(word.lower())) <= 2 and len(word) > 3:
        return False
    
    return True
def preprocess_text(text: str, cfg: PreprocessConfig | None = None) -> str:
    """
    Enhanced token‑level cleaning used before embedding / topic modelling.

    Parameters
    ----------
    text : str
        Raw paragraph or sentence.
    cfg : PreprocessConfig or None
        Optional settings; defaults to `PreprocessConfig()`.

    Returns
    -------
    str
        Space‑joined cleaned tokens.
    """
    cfg = cfg or PreprocessConfig()

    if not text or not text.strip():
        return ""
    
    # Apply enhanced normalization first
    normalized_text = normalize_text_enhanced(text, cfg)
    if not normalized_text:
        return ""

    nlp = get_nlp()
    doc = nlp(normalized_text.lower() if cfg.lowercase else normalized_text)

    sw = build_stopwords() if cfg.remove_stopwords else set()
    lemmatizer = get_lemmatizer() if cfg.lemmatize else None

    cleaned: List[str] = []
    for tok in doc:
        # Skip if it's just punctuation or whitespace
        if not tok.text or tok.text.isspace():
            continue
            
        # More sophisticated token filtering
        if not tok.is_alpha and not (tok.is_digit and not cfg.remove_numbers):
            continue
        
        # Apply enhanced word validation
        if not is_valid_word(tok.text, cfg):
            continue
            
        if cfg.allowed_pos and tok.pos_.upper() not in cfg.allowed_pos:
            continue
            
        if cfg.remove_stopwords and tok.text.lower() in sw:
            continue

        # Apply lemmatization if configured
        word = lemmatizer.lemmatize(tok.text.lower()) if lemmatizer else tok.text
        
        # Final check after lemmatization
        if is_valid_word(word, cfg):
            cleaned.append(word)

    result = " ".join(cleaned)
    
    # Remove very short results if configured
    if cfg.remove_short_sentences and len(result.split()) < 3:
        return ""
    
    return result


# ────────────────────────────────────────────────────────────────────────────
# Story‑level metadata extraction
# ────────────────────────────────────────────────────────────────────────────
_STORY_DIV_RE = re.compile(r'<div\s+type="story"[^>]*>(.*?)</div>', re.I | re.S)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_AUTHOR_RE = re.compile(r"<author[^>]*>\s*(.*?)\s*</author>", re.I | re.S)
_HEAD_RE = re.compile(r"<head>.*?</head>", re.I | re.S)


def extract_story_metadata(html_text: str) -> List[Tuple[str, List[str], str]]:
    """
    Extract (title, authors, content) tuples from an Amazing Stories HTML fragment.
    """
    results: List[Tuple[str, List[str], str]] = []
    for block in _STORY_DIV_RE.findall(html_text):
        clean_block = _HEAD_RE.sub("", block).strip()
        if not clean_block:
            continue

        title_match = _TITLE_RE.search(block)
        title = title_match.group(1).strip() if title_match else ""

        authors = [m.strip() for m in _AUTHOR_RE.findall(block)]

        if title:
            results.append((title, authors, clean_block))

    return results


# ────────────────────────────────────────────────────────────────────────────
# Data‑frame‑like helpers
# ────────────────────────────────────────────────────────────────────────────
def process_texts_df(records: Sequence[dict]) -> List[dict]:
    """
    Convert file‑level dicts to story‑level dicts with full text.
    """
    stories: List[dict] = []
    for rec in records:
        works = extract_story_metadata(rec.get("text", ""))
        for title, authors, full_text in works:
            stories.append(
                {
                    "volume_number": rec.get("volume_number"),
                    "issue_number": rec.get("issue_number"),
                    "publication_year": rec.get("publication_year"),
                    "publication_month": rec.get("publication_month"),
                    "title": title,
                    "authors": authors,
                    "full_text": full_text,
                }
            )
    logger.info("Extracted %d story entries", len(stories))
    return stories


def segment_paragraphs(stories: Sequence[dict]) -> List[dict]:
    """
    Split each full_text into paragraphs (>= 1 blank line delimiter).
    """
    paragraphs: List[dict] = []
    for story_idx, story in enumerate(stories, start=1):
        story_id = f"story_{story_idx:05d}"
        parts = [p.strip() for p in re.split(r"\n\s*\n+", story["full_text"]) if p.strip()]

        for p_idx, para in enumerate(parts, start=1):
            entry = {
                **{k: story[k] for k in story if k != "full_text"},
                "story_id": story_id,
                "paragraph_index": p_idx,
                "paragraph_text": para,
            }
            paragraphs.append(entry)

    logger.info("Segmented into %d paragraphs", len(paragraphs))
    return paragraphs


# ────────────────────────────────────────────────────────────────────────────
# File‑name helper
# ────────────────────────────────────────────────────────────────────────────
_FILE_RE = re.compile(r"amazing_stories-(\d+)\.(\d+)\((\d+)\.(\d+)\)")


def get_sorted_file_entries(base_dir: str | os.PathLike, pattern: re.Pattern | None = None) -> List[Tuple[int, int, int, int, str]]:
    """
    List filenames in chronological order based on Amazing Stories naming convention.
    """
    pattern = pattern or _FILE_RE
    entries: List[Tuple[int, int, int, int, str]] = []

    for fname in os.listdir(base_dir):
        m = pattern.match(fname)
        if not m:
            continue
        volume, issue, year, month = map(int, m.groups())
        entries.append((year, month, volume, issue, fname))

    entries.sort()
    logger.debug("Discovered %d files in %s", len(entries), base_dir)
    return entries