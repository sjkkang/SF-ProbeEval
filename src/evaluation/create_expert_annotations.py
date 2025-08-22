#!/usr/bin/env python3
"""
Expert Annotations Dataset Creator for SF-ProbeEval

This module creates expert annotation datasets by:
1. Loading processed text pairs from the SF-ProbeEval pipeline
2. Applying quality filters to remove problematic samples
3. Performing stratified sampling to ensure balanced score distribution  
4. Generating annotation-ready CSV files with guidelines

The resulting dataset contains 250 carefully curated text pairs with equal
representation across quality score ranges (1-5) for human evaluation.

Usage:
    python create_expert_annotations.py --input data/processed/final_pairs.csv --samples 250
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import random

def load_and_analyze_data(file_path: str) -> pd.DataFrame:
    """Load CSV file and perform basic analysis."""
    print(f"Loading data from: {file_path}")
    
    # Load CSV file (assuming no header)
    df = pd.read_csv(file_path, header=None)
    
    # Set column names (based on observed pattern)
    df.columns = ['original_text', 'modified_text', 'combined_text', 'score']
    
    # Convert score column to numeric (handle strings or invalid values)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check score distribution
    print("\n=== Score Distribution ===")
    score_dist = df['score'].value_counts().sort_index()
    print(score_dist)
    
    # Check NaN values
    nan_count = df['score'].isna().sum()
    if nan_count > 0:
        print(f"NaN values in score: {nan_count}")
    
    # Check score percentages
    print("\n=== Score Distribution (Percentage) ===")
    score_pct = df['score'].value_counts(normalize=True).sort_index() * 100
    for score, pct in score_pct.items():
        if not pd.isna(score):
            print(f"Score {score}: {pct:.2f}%")
    
    return df

def filter_invalid_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid or problematic samples."""
    initial_count = len(df)
    
    # Filter out samples where modified_text starts with common error patterns
    error_patterns = ['Answer:', 'Sentence:', 'The most appropriate', 'Rewritten sentence:']
    for pattern in error_patterns:
        df = df[~df['modified_text'].str.startswith(pattern, na=False)]
    
    # Filter out samples where original_text contains CHAPTER
    df = df[~df['original_text'].str.contains('CHAPTER', case=False, na=False)]
    
    # Filter out very short texts (less than 10 characters)
    df = df[(df['original_text'].str.len() >= 10) & (df['modified_text'].str.len() >= 10)]
    
    # Filter out samples where modified_text is just a single word (likely incomplete)
    df = df[df['modified_text'].str.split().str.len() > 1]
    
    filtered_count = len(df)
    print(f"Filtered out {initial_count - filtered_count} invalid samples")
    print(f"Remaining samples: {filtered_count}")
    
    return df

def stratified_sample_by_score(df: pd.DataFrame, total_samples: int = 250) -> pd.DataFrame:
    """Perform stratified sampling by score."""
    
    # Remove rows with NaN values
    df_clean = df.dropna(subset=['score']).copy()
    print(f"Removed {len(df) - len(df_clean)} rows with NaN scores")
    
    # Calculate number of samples per score (equal distribution)
    unique_scores = sorted(df_clean['score'].unique())
    # Exclude -1 scores, 0 scores, and non-integer scores (use only 1-5 integer scores)
    valid_scores = [s for s in unique_scores if s >= 1 and s <= 5 and s == int(s)]
    
    samples_per_score = total_samples // len(valid_scores)
    remaining_samples = total_samples % len(valid_scores)
    
    print(f"\n=== Sampling Strategy ===")
    print(f"Valid score ranges: {valid_scores}")
    print(f"Base samples per score: {samples_per_score}")
    print(f"Extra samples to distribute: {remaining_samples}")
    
    sampled_dfs = []
    
    for i, score in enumerate(valid_scores):
        score_df = df_clean[df_clean['score'] == score].copy()
        
        # Distribute extra samples to the first few scores
        current_sample_size = samples_per_score
        if i < remaining_samples:
            current_sample_size += 1
            
        if len(score_df) < current_sample_size:
            print(f"Warning: Score {score} has only {len(score_df)} samples, less than requested {current_sample_size}")
            current_sample_size = len(score_df)
        
        # Random sampling
        sampled = score_df.sample(n=current_sample_size, random_state=42)
        sampled_dfs.append(sampled)
        
        print(f"Score {score}: {current_sample_size} samples (available: {len(score_df)})")
    
    # Combine all samples and shuffle
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal sample size: {len(final_sample)}")
    
    return final_sample

def create_expert_annotation_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to format for expert annotation."""
    
    # Add new columns for expert annotation
    annotation_df = df.copy()
    
    # Add required columns
    annotation_df['sample_id'] = range(1, len(annotation_df) + 1)
    annotation_df['expert_score'] = ''  # Score to be filled by expert
    annotation_df['expert_comments'] = ''  # Expert comments
    annotation_df['annotation_date'] = ''  # Annotation date
    annotation_df['annotator_id'] = ''  # Annotator ID
    
    # Reorder columns
    columns_order = [
        'sample_id',
        'original_text', 
        'modified_text', 
        'score',  # Original automatic score
        'expert_score',  # Expert score
        'expert_comments',
        'annotation_date',
        'annotator_id'
    ]
    
    annotation_df = annotation_df[columns_order]
    
    return annotation_df

def save_expert_annotations(df: pd.DataFrame, output_path: str):
    """Save expert annotation file."""
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nExpert annotations saved to: {output_path}")
    
    # Print statistics of saved file
    print(f"Saved {len(df)} samples")
    print("\n=== Saved Data Score Distribution ===")
    score_dist = df['score'].value_counts().sort_index()
    print(score_dist)

def create_annotation_guide(output_dir: str):
    """Create annotation guidelines file."""
    
    guide_content = """# Expert Annotation Guidelines

## Overview
This file is a dataset for expert annotation of text pairs in the Amazing Stories project.


## Annotation Method
1. `sample_id`: Sample number (do not modify)
2. `original_text`: Original text
3. `modified_text`: Modified text
4. `score`: Automatically generated score (for reference)
5. `expert_score`: Please enter expert score here (1-5)
6. `expert_comments`: Record reasons for the score or special notes
7. `annotation_date`: Annotation date (YYYY-MM-DD format)
8. `annotator_id`: Annotator identifier

## Important Notes
- Please carefully review each sample
- Enter scores as integers only (1, 2, 3, 4, 5)
- If uncertain, record the reason in expert_comments
- Consider the context and meaning of the text when evaluating
"""
    
    guide_path = Path(output_dir) / "annotation_guidelines.md"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Annotation guidelines saved to: {guide_path}")

def main():
    parser = argparse.ArgumentParser(description='Create Expert Annotations dataset')
    parser.add_argument('--input', '-i', 
                       default='sf_topic_pipeline/output/enhanced_preprocessing_topics_v3/final_analysis/final_pairs_valid_cleaned.csv',
                       help='Input CSV file path')
    parser.add_argument('--output-dir', '-o', 
                       default='expert_annotations/',
                       help='Output directory')
    parser.add_argument('--samples', '-s', type=int, default=250,
                       help='Number of samples to select (default: 250)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Data loading and analysis
        df = load_and_analyze_data(args.input)
        
        # Filter invalid samples
        df_filtered = filter_invalid_samples(df)
        
        # Perform stratified sampling
        sampled_df = stratified_sample_by_score(df_filtered, args.samples)
        
        # Convert to expert annotation format
        annotation_df = create_expert_annotation_format(sampled_df)
        
        # Save file
        output_file = output_dir / f"expert_annotations_{args.samples}samples.csv"
        save_expert_annotations(annotation_df, str(output_file))
        
        # Create guidelines
        create_annotation_guide(str(output_dir))
        
        print("\n=== Process Completed Successfully ===")
        print(f"Files created in: {output_dir}")
        print(f"- {output_file.name}")
        print("- annotation_guidelines.md")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
