# step1_topic_modeling_improved.py  (v3 – 2025‑07‑12)
"""
Improved Topic‑model discovery on paragraph‑level data from Amazing Stories.

Key improvements:
- Uses UMAP for better visualization and clustering
- Balanced topic size handling with HDBSCAN parameter tuning
- Enhanced preprocessing for imbalanced data
- Better embedding model options
- Improved metrics calculation
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from umap import UMAP

from .common import PreprocessConfig, build_stopwords, preprocess_text

LOGFMT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    """Set Python, NumPy, and Torch seeds for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed fixed to %d", seed)


def topic_diversity(topic_model: BERTopic, top_k: int = 20) -> float:
    """
    Fraction of unique words across topic top‑k lists.
    High values ⇒ more diverse topics.
    """
    words = [w for t in topic_model.get_topics().values() for w, _ in t[:top_k]]
    return len(set(words)) / (top_k * len(topic_model.get_topics()))


def balance_data_sampling(paragraphs: List[dict], max_per_author: int = 1000) -> List[dict]:
    """
    Balance data by limiting documents per author to reduce author bias.
    """
    from collections import defaultdict
    import random
    
    # Group by author
    author_docs = defaultdict(list)
    for i, doc in enumerate(paragraphs):
        authors = doc.get('authors', ['Unknown'])
        if isinstance(authors, list):
            author = authors[0] if authors else 'Unknown'
        else:
            author = authors
        author_docs[author].append((i, doc))
    
    # Sample balanced data
    balanced_docs = []
    for author, docs in author_docs.items():
        if len(docs) > max_per_author:
            # Sample randomly
            sampled = random.sample(docs, max_per_author)
            balanced_docs.extend([doc for _, doc in sampled])
            logger.info(f"Sampled {max_per_author} docs from {author} (had {len(docs)})")
        else:
            balanced_docs.extend([doc for _, doc in docs])
    
    logger.info(f"Balanced sampling: {len(paragraphs)} → {len(balanced_docs)} documents")
    return balanced_docs


# ────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ────────────────────────────────────────────────────────────────────────────
def run_improved_topic_modeling(
    paragraphs: List[dict],
    output_dir: Path,
    *,
    embed_model_name: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_n_components: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    clusterer: str,
    k_topics: int | None,
    stage: str = "full",
) -> None:
    # 0 ─ Data preprocessing (no author balancing)
    logger.info("Using all data without author balancing...")
    
    # 1 ─ Enhanced pre‑processing
    logger.info("Pre‑processing paragraphs with enhanced normalization...")
    stop_set = build_stopwords()
    cfg = PreprocessConfig(
        lowercase=True, 
        remove_stopwords=True, 
        lemmatize=True,
        min_word_length=3,  # Increased from 2 to 3
        remove_numbers=True,
        normalize_hyphens=True,  # New: normalize hyphenated words
        min_alpha_ratio=0.7,  # New: require 70% alphabetic characters
        remove_single_chars=True,  # New: remove single character words
        normalize_contractions=True  # New: normalize contractions
    )
    texts_clean = [preprocess_text(p["paragraph_text"], cfg) for p in paragraphs]
    
    # Filter out very short texts
    min_text_length = 10
    filtered_data = [(text, para) for text, para in zip(texts_clean, paragraphs) 
                     if len(text.split()) >= min_text_length]
    
    if len(filtered_data) < len(texts_clean):
        logger.info(f"Filtered {len(texts_clean) - len(filtered_data)} short texts")
        texts_clean, paragraphs = zip(*filtered_data)
        texts_clean = list(texts_clean)
        paragraphs = list(paragraphs)
    
    logger.info("Pre‑processed %d paragraphs", len(texts_clean))

    # 2 ─ Enhanced embeddings
    embed_model = SentenceTransformer(embed_model_name)
    
    embeddings_file = output_dir / "embeddings.npy"
    texts_file = output_dir / "texts_clean.json"
    
    if stage == "embed" or not embeddings_file.exists():
        logger.info("Computing embeddings …")
        # Batch processing for large datasets
        batch_size = 32
        embeddings = embed_model.encode(texts_clean, 
                                       show_progress_bar=True, 
                                       batch_size=batch_size)
        
        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_file, embeddings)
        with open(texts_file, "w", encoding="utf-8") as f:
            json.dump(texts_clean, f, ensure_ascii=False, indent=2)
        logger.info("Embeddings saved to %s", embeddings_file)
        
        if stage == "embed":
            logger.info("Stage 'embed' completed.")
            return
    else:
        logger.info("Loading existing embeddings from %s", embeddings_file)
        embeddings = np.load(embeddings_file)
        with open(texts_file, "r", encoding="utf-8") as f:
            texts_clean = json.load(f)

    # 3 ─ Improved dimensionality reduction (UMAP)
    logger.info("Setting up UMAP for dimensionality reduction...")
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
        n_jobs=1  # Avoid multiprocessing issues
    )

    # 4 ─ Enhanced clustering algorithm
    if clusterer == "hdbscan":
        logger.info(f"Using HDBSCAN with min_cluster_size={hdbscan_min_cluster_size}")
        cluster_model = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    elif clusterer == "kmeans":
        cluster_model = KMeans(n_clusters=k_topics or 20, random_state=42)
    elif clusterer == "gmm":
        cluster_model = GaussianMixture(n_components=k_topics or 20, random_state=42)
    else:
        raise ValueError(f"Unsupported clusterer: {clusterer!r}")

    # 5 ─ Enhanced TF‑IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include trigrams
        stop_words=list(stop_set), 
        min_df=3,  # Stricter minimum frequency
        max_df=0.95,  # Remove very common words
        max_features=10000  # Limit vocabulary size
    )

    # 6 ─ Enhanced BERTopic model
    topic_model_file = output_dir / "topic_model.pkl"
    topics_file = output_dir / "topic_assignments.pkl"
    
    if stage == "cluster" or not topic_model_file.exists():
        logger.info("Creating enhanced BERTopic model...")
        
        # Enhanced representation models
        representation_model = KeyBERTInspired(top_n_words=50)
        
        topic_model = BERTopic(
            embedding_model=embed_model,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            top_n_words=50,
            nr_topics="auto" if k_topics is None else k_topics,
            calculate_probabilities=True,
            verbose=True,
        )

        logger.info("Fitting BERTopic …")
        topics, probs = topic_model.fit_transform(texts_clean)
        
        # Save clustering results
        pickle.dump(topic_model, open(topic_model_file, "wb"))
        pickle.dump((topics, probs), open(topics_file, "wb"))
        logger.info("Topic model saved to %s", topic_model_file)
        
        if stage == "cluster":
            logger.info("Stage 'cluster' completed.")
            return
    else:
        logger.info("Loading existing topic model from %s", topic_model_file)
        topic_model = pickle.load(open(topic_model_file, "rb"))
        topics, probs = pickle.load(open(topics_file, "rb"))

    # 7 ─ Enhanced metric computation
    if stage in ["analyze", "visualize"] or stage == "full":
        logger.info("Computing enhanced metrics...")
        
        # Basic metrics
        try:
            from gensim.models import CoherenceModel
            from gensim.corpora import Dictionary
            
            # Prepare data for coherence calculation
            tokenized_texts = [text.split() for text in texts_clean]
            dictionary = Dictionary(tokenized_texts)
            
            # Get topic words for coherence calculation
            topic_words = []
            for topic_id in set(topics):
                if topic_id != -1:  # Skip noise cluster
                    words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                    topic_words.append(words)
            
            # Calculate coherence
            if topic_words:
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                c_v = coherence_model.get_coherence()
            else:
                c_v = -1.0
                
        except ImportError:
            logger.warning("Gensim not available for coherence calculation")
            c_v = -1.0
        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            c_v = -1.0
        
        # Topic quality metrics
        from collections import Counter
        topic_sizes = Counter(topics)
        
        # Calculate balance metrics
        sizes = list(topic_sizes.values())
        if sizes:
            size_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
            size_std = np.std(sizes)
        else:
            size_ratio = 0
            size_std = 0
        
        metrics = {
            "c_v": c_v,
            "diversity": topic_diversity(topic_model),
            "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
            "noise_ratio": sum(1 for t in topics if t == -1) / len(topics),
            "size_ratio": size_ratio,
            "size_std": size_std,
        }
        
        logger.info(
            "Metrics: c_v=%.4f | diversity=%.3f | n_topics=%d | noise_ratio=%.3f",
            metrics["c_v"], metrics["diversity"], metrics["n_topics"], metrics["noise_ratio"]
        )
        logger.info("Balance: size_ratio=%.2f | size_std=%.2f", metrics["size_ratio"], metrics["size_std"])

        # 8 ─ Save enhanced results
        output_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced metrics CSV
        metrics_row = pd.DataFrame([{
            "date": _dt.date.today().isoformat(),
            "embed": embed_model_name,
            "umap_neighbors": umap_n_neighbors,
            "umap_components": umap_n_components,
            "umap_min_dist": umap_min_dist,
            "clusterer": clusterer,
            "hdbscan_min_cluster": hdbscan_min_cluster_size,
            "k_topics": k_topics or -1,
            **metrics,
        }])
        
        metrics_path = output_dir / "enhanced_topic_metrics.csv"
        metrics_row.to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)

        # Enhanced model card
        card = {
            "date": _dt.datetime.now().isoformat(),
            "input_size": len(texts_clean),
            "parameters": {
                "embed_model": embed_model_name,
                "umap_neighbors": umap_n_neighbors,
                "umap_components": umap_n_components,
                "umap_min_dist": umap_min_dist,
                "clusterer": clusterer,
                "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                "k_topics": k_topics,
            },
            "metrics": metrics,
            "improvements": [
                "UMAP for better dimensionality reduction",
                "Enhanced HDBSCAN parameters",
                "Improved preprocessing",
                "Better coherence calculation"
            ]
        }
        (output_dir / "enhanced_model_card.json").write_text(json.dumps(card, indent=2))
        
        if stage == "analyze":
            logger.info("Stage 'analyze' completed.")
            return

    # 9 ─ Enhanced visualization
    if stage in ["visualize"] or stage == "full":
        logger.info("Creating enhanced visualizations...")
        
        # Use UMAP for 2D visualization (better than PCA)
        umap_2d = UMAP(n_components=2, random_state=42, n_jobs=1)
        emb_2d = umap_2d.fit_transform(embeddings)
        
        vis_df = pd.DataFrame({
            "x": emb_2d[:, 0], 
            "y": emb_2d[:, 1], 
            "topic": topics,
            "probability": [max(prob) for prob in probs] if probs is not None else [1.0] * len(topics)
        })
        vis_df.to_csv(output_dir / "umap_2d_points.csv", index=False)

        # Enhanced plotly visualization
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create enhanced scatter plot
            fig = px.scatter(
                vis_df, 
                x="x", y="y", 
                color="topic", 
                size="probability",
                opacity=0.6,
                width=1000, height=800,
                title="Enhanced Topic Visualization (UMAP)",
                labels={"x": "UMAP 1", "y": "UMAP 2", "topic": "Topic ID"}
            )
            
            # Add topic labels
            topic_centers = vis_df.groupby('topic')[['x', 'y']].mean()
            for topic_id, center in topic_centers.iterrows():
                if topic_id != -1:  # Skip noise
                    fig.add_annotation(
                        x=center['x'], y=center['y'],
                        text=f"T{topic_id}",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
            
            fig.write_html(output_dir / "enhanced_visualization.html")
            fig.write_json(output_dir / "enhanced_scatter.json")
            
        except Exception as exc:
            logger.warning("Enhanced visualization failed: %s", exc)

    # 10 ─ Save extended topic keywords
    if stage in ["analyze", "visualize", "full"]:
        save_extended_topic_keywords(topic_model, output_dir, top_n_words=50)

    logger.info("Enhanced Step‑1 finished; artifacts stored in %s", output_dir)


def save_extended_topic_keywords(topic_model, output_dir: Path, top_n_words: int = 30) -> None:
    """Extract and save extended topic keywords."""
    logger.info(f"Extracting top {top_n_words} keywords per topic...")
    
    # Get topic info with more keywords
    topic_info = topic_model.get_topic_info()
    
    # Extract keywords for each topic
    topic_keywords = {}
    for topic_id in topic_info['Topic']:
        if topic_id != -1:  # Skip noise topic
            # Get topic words with scores
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Take top N words
                keywords = [word for word, score in topic_words[:top_n_words]]
                topic_keywords[str(topic_id)] = keywords
    
    # Save keywords to JSON
    keywords_path = output_dir / "topic_keywords.json"
    with open(keywords_path, 'w', encoding='utf-8') as f:
        json.dump(topic_keywords, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Extended topic keywords saved to {keywords_path}")
    
    # Also save detailed topic info
    detailed_info = []
    for topic_id in topic_info['Topic']:
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                detailed_info.append({
                    'topic_id': int(topic_id),
                    'count': int(topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0]),
                    'name': str(topic_info[topic_info['Topic'] == topic_id]['Name'].iloc[0]),
                    'keywords': [{'word': word, 'score': float(score)} for word, score in topic_words[:top_n_words]]
                })
    
    detailed_path = output_dir / "topic_keywords_detailed.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed topic info saved to {detailed_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced Topic Modeling Pipeline")
    p.add_argument("--input_json", required=True, help="Paragraph JSON from step0")
    p.add_argument("--output_dir", required=True, help="Destination folder")
    p.add_argument("--embed_model", default="all-MiniLM-L12-v2", 
                   help="Embedding model (try larger models for better results)")
    
    # UMAP parameters
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_n_components", type=int, default=10)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    
    # HDBSCAN parameters
    p.add_argument("--hdbscan_min_cluster_size", type=int, default=50)
    p.add_argument("--hdbscan_min_samples", type=int, default=10)
    
    # Other parameters
    p.add_argument("--clusterer", choices=["hdbscan", "kmeans", "gmm"], default="hdbscan")
    p.add_argument("--k_topics", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["embed", "cluster", "analyze", "visualize", "full"], 
                   default="full")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    paragraphs = json.load(open(args.input_json, encoding="utf-8"))

    run_improved_topic_modeling(
        paragraphs,
        Path(args.output_dir),
        embed_model_name=args.embed_model,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        clusterer=args.clusterer,
        k_topics=args.k_topics,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
