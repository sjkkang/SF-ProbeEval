#!/bin/bash
"""
Full SF-ProbeEval Pipeline Runner

This script runs the complete SF-ProbeEval pipeline:
1. Data extraction and preprocessing
2. Topic modeling  
3. Text masking and pair generation
4. Quality filtering
5. Expert annotation dataset creation

Usage:
    ./scripts/run_full_pipeline.sh [OPTIONS]
    
Environment Variables:
    INPUT_DATA_DIR: Directory containing raw Amazing Stories files
    OUTPUT_DIR: Directory for pipeline outputs (default: outputs/)
    SAMPLES: Number of expert annotation samples (default: 250)
"""

set -e  # Exit on any error

# Default configurations
INPUT_DATA_DIR="${INPUT_DATA_DIR:-data/raw/amazing_stories}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
SAMPLES="${SAMPLES:-250}"
SEED="${SEED:-42}"

echo "🚀 Starting SF-ProbeEval Pipeline..."
echo "Input directory: $INPUT_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Expert annotation samples: $SAMPLES"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"/{step0,step1,step2,step3,step4,expert_annotations}

echo "📝 Step 0: Data extraction and preprocessing..."
python -m src.sf_topic_pipeline.data_extraction \
    --input_dir "$INPUT_DATA_DIR" \
    --output_dir "$OUTPUT_DIR/step0" \
    --seed "$SEED"

echo ""
echo "🧠 Step 1: Topic modeling..."
python -m src.sf_topic_pipeline.topic_modeling \
    --input_json "$OUTPUT_DIR/step0/paragraphs.json" \
    --output_dir "$OUTPUT_DIR/step1" \
    --embed_model_name "all-MiniLM-L12-v2" \
    --clusterer "hdbscan" \
    --seed "$SEED"

echo ""  
echo "🎭 Step 2: Topic-based text masking..."
python -m src.sf_topic_pipeline.topic_masking \
    --input_json "$OUTPUT_DIR/step0/paragraphs.json" \
    --topic_model_path "$OUTPUT_DIR/step1/topic_model.pkl" \
    --output_dir "$OUTPUT_DIR/step2" \
    --seed "$SEED"

echo ""
echo "👥 Step 3: Text pair generation..."
python -m src.sf_topic_pipeline.pair_generation \
    --input_dir "$OUTPUT_DIR/step2" \
    --output_dir "$OUTPUT_DIR/step3" \
    --seed "$SEED"

echo ""
echo "🔤 Step 4: Text infilling and quality scoring..."
python -m src.sf_topic_pipeline.infill \
    --input_file "$OUTPUT_DIR/step3/pairs.jsonl" \
    --output_dir "$OUTPUT_DIR/step4" \
    --batch_size 16

echo ""
echo "🧹 Filtering and quality control..."
python -m src.sf_topic_pipeline.filter_pairs \
    --input "$OUTPUT_DIR/step4/infilled_pairs_with_hybrid.jsonl" \
    --output "$OUTPUT_DIR/step4/final_pairs_clean.csv" \
    --th_lp -3.0 \
    --th_hyb 0.3 \
    --min_len 10 \
    --max_len 200

echo ""
echo "👨‍💼 Creating expert annotation dataset..."
python -m src.evaluation.create_expert_annotations \
    --input "$OUTPUT_DIR/step4/final_pairs_clean.csv" \
    --output-dir "$OUTPUT_DIR/expert_annotations" \
    --samples "$SAMPLES" \
    --seed "$SEED"

echo ""
echo "✅ SF-ProbeEval Pipeline Complete!"
echo ""
echo "📊 Results:"
echo "  - Topic model: $OUTPUT_DIR/step1/"
echo "  - Text pairs: $OUTPUT_DIR/step4/final_pairs_clean.csv" 
echo "  - Expert annotations: $OUTPUT_DIR/expert_annotations/"
echo ""
echo "📝 Next steps:"
echo "  1. Review the expert annotation guidelines"
echo "  2. Begin human annotation using the generated CSV"
echo "  3. Use the annotated data for model evaluation"
