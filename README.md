# SF-ProbeEval: Science Fiction Text Probing Evaluation Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A comprehensive evaluation dataset and pipeline for probing language model understanding of science fiction narratives through topic modeling, text infilling, and expert annotation.

## 🌟 Overview

SF-ProbeEval provides:

- **📚 Text Processing Pipeline**: Advanced topic modeling and paragraph segmentation for science fiction stories
- **🎭 Text Infilling System**: Masked language modeling for narrative coherence evaluation  
- **👨‍💼 Expert Annotations**: Human-annotated dataset with 250 carefully curated samples
- **📊 Evaluation Metrics**: Comprehensive scoring system for generated text quality

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sjkkang/SF-ProbeEval.git
cd SF-ProbeEval

# Setup environment and install dependencies
./scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### Run the Full Pipeline

```bash
# Run complete SF-ProbeEval pipeline
./scripts/run_full_pipeline.sh

# Or run individual steps
python -m src.sf_topic_pipeline.topic_modeling --help
python -m src.evaluation.create_expert_annotations --help
```

### Create Expert Annotations

```bash
# Generate expert annotation dataset
python -m src.evaluation.create_expert_annotations \
    --input data/processed/final_pairs_cleaned.csv \
    --samples 250 \
    --output-dir data/expert_annotations/
```

## 📊 Dataset

### Domain-Specific Corpus Construction

This research employs a curated collection of early 20th-century science fiction texts to demonstrate the effectiveness of the proposed framework under challenging adaptation scenarios. The dataset comprises **531 documents** with approximately **18,618 text segments** and an estimated **35,000 distinct sentences** after comprehensive preprocessing and quality control. 

The corpus represents a domain characterized by:
- **🔬 Specialized Vocabulary**: Scientific and technical terminology unique to SF narratives
- **✍️ Distinctive Stylistic Conventions**: Period-specific writing patterns and narrative structures  
- **🧠 Domain-Specific Semantic Relationships**: Complex thematic and conceptual associations
- **📈 Resource-Constrained Conditions**: Ideal testbed for domain adaptation methodologies

### SF-ProbeEval: A Domain-Specific Probing Benchmark

To evaluate embedding quality in early science fiction texts, this research constructs **SF-ProbeEval**, a specialized benchmark comprising **five probing tasks** tailored to pulp-era linguistic patterns. Each task contains **1000 test items** derived from the specialized domain corpus, capturing the unique challenges of historical science fiction text understanding. 

SF-ProbeEval provides the **first standardized benchmark** for assessing sentence embedding quality in early 20th-century science fiction texts, addressing the gap in domain-specific evaluation frameworks for historical literary corpora.

#### Construction Methodology
SF-ProbeEval construction follows a **two-stage process**:
1. **Automated Generation**: Rule-based algorithms and domain-specific linguistic patterns
2. **Expert Validation**: Three experts from Department of English Language and Literature
   - Focus on linguistic accuracy, difficulty appropriateness, and artifact elimination
   - Independent evaluation by at least two reviewers per test item
   - Disagreements resolved through consensus-building

#### Probing Tasks

| Task | Description |
|------|-------------|
| **Word Contents** | Identify science fiction terminology and archaic vocabulary, evaluating adaptation to period-specific lexicons including scientific devices, astronomical terms, and technological concepts from pulp-era narratives |
| **Tree Depth** | Predict syntactic complexity levels in vintage prose, assessing understanding of elaborate sentence constructions characteristic of 1920s-1930s science fiction writing styles |
| **BShift (Bigram Shift)** | Detect local syntactic perturbations in period-appropriate word sequences, measuring sensitivity to historical word order patterns and archaic grammatical structures |
| **SOMO (Semantic Odd Man Out)** | Identify semantic anomalies within science fiction contexts, evaluating understanding of genre-specific relationships including scientific speculation and technological innovation concepts |
| **Coord_Inv (Coordinate Inversion)** | Detect structural modifications in complex vintage sentences, assessing comprehension of elaborate discourse patterns typical of early science fiction literary style |

### Probing Task Dataset
- **🎯 Purpose**: Evaluating language model understanding of science fiction narratives
- **📚 Source**: Amazing Stories magazine corpus (1926+)
- **🔍 Task Types**: 5 specialized probing tasks (1000 items each)
- **📊 Format**: Domain-specific linguistic pattern evaluation

### Expert Annotations (Contrastive Learning Validation)
- **🎯 Purpose**: Quality validation for contrastive learning dataset
- **📋 Size**: 250 text pairs with balanced score distribution
- **🎯 Score Range**: 1-5 (integer scores, equal stratification)
- **🔍 Quality**: Multi-stage filtering for completeness and coherence
- **📈 Balance**: 50 samples per score level (1-5)

### Dataset Statistics
| Component | Count | Description |
|-----------|-------|-------------|
| **Raw Corpus** | 531 documents | Early 20th-century SF texts |
| **Text Segments** | ~18,618 | Processed paragraph units |
| **Distinct Sentences** | ~35,000 | After preprocessing & QC |
| **Probing Pairs** | Variable | Language model evaluation |
| **Expert Annotations** | 250 | Contrastive learning validation |

### Filtering Statistics
| Metric | Value |
|--------|-------|
| Total Expert Samples | 250 |
| Score Distribution | Equal (50 per score 1-5) |
| Filtering Applied | 7.7% samples removed |
| Quality Criteria | Completeness + coherence |

### Sample Data Format
```csv
sample_id,original_text,modified_text,score,expert_score,expert_comments,annotation_date,annotator_id
1,"His impression of the method...","His impression of the method...",4.0,,,, 
```

## 🏗️ Architecture

<!-- Replace with your custom architecture diagram -->
![SF-ProbeEval Architecture](docs/images/Figure1_workflow.pdf)

<!-- Alternative: Keep mermaid diagram as backup -->
<details>
<summary>View Text-based Architecture Diagram</summary>

```mermaid
graph LR
    A[Raw Stories] --> B[Data Extraction]
    B --> C[Topic Modeling] 
    C --> D[Text Masking]
    D --> E[Pair Generation]
    E --> F[Text Infilling]
    F --> G[Quality Filtering]
    G --> H[Expert Annotations]
```
</details>

### Pipeline Components

1. **📝 Data Extraction** (`data_extraction.py`)
   - HTML story extraction and metadata parsing
   - Paragraph segmentation and deduplication
   - Train/dev/test splits generation

2. **🧠 Topic Modeling** (`topic_modeling.py`)
   - Enhanced preprocessing with UMAP + HDBSCAN
   - Balanced topic discovery with quality metrics
   - Coherence-based evaluation

3. **🎭 Text Masking** (`topic_masking.py`) 
   - Topic-aware masking strategies
   - Context-preserving token selection
   - Multiple masking patterns

4. **👥 Pair Generation** (`pair_generation.py`)
   - Systematic text pair creation
   - Quality-based filtering
   - Similarity scoring

5. **🔤 Text Infilling** (`infill.py`)
   - Transformer-based text completion
   - Multiple model support (T5, BERT, etc.)
   - Confidence scoring

6. **🧹 Quality Control** (`filter_pairs.py`)
   - Multi-criteria filtering
   - Deduplication by similarity scores
   - Length and completeness validation

## 📚 Repository Structure

```
SF-ProbeEval/
├── src/                          # Source code
│   ├── sf_topic_pipeline/        # Core pipeline modules
│   └── evaluation/               # Evaluation tools
├── data/                         # Datasets and annotations
│   ├── raw/                      # Original story files
│   ├── processed/                # Processed datasets  
│   └── expert_annotations/       # Human annotation data
├── scripts/                      # Utility and run scripts
├── notebooks/                    # Analysis notebooks
├── configs/                      # Configuration files
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## 🔬 Reproducibility

### Preprocessing Configuration
```python
PreprocessConfig(
    lowercase=True,                    # Convert to lowercase
    remove_stopwords=True,             # Remove NLTK English stopwords
    lemmatize=True,                    # WordNet lemmatization
    min_word_length=3,                 # Minimum 3 characters per word
    min_alpha_ratio=0.7,              # 70% alphabetic characters required
    normalize_contractions=True,       # "won't" → "will not"
    normalize_hyphens=True             # "to-morrow" → "tomorrow"
)
```

### Quality Filtering Criteria
- ❌ **Error Patterns**: Remove "Answer:", "Sentence:" prefixes
- ❌ **Structural Text**: Filter "CHAPTER" headers and metadata  
- ❌ **Incomplete Text**: Remove single words or very short content
- ❌ **Invalid Format**: Ensure proper text pair structure
- ✅ **Length Validation**: 10+ characters and 3+ words minimum

### Deduplication Strategy
- **Primary Key**: `pair_id` deduplication
- **Similarity Metrics**: Semantic + lexical + structural scoring

## 📈 Results & Evaluation

### Model Performance
- **Topic Coherence**:  Scores with enhanced preprocessing
- **Quality Improvement**: ~7.7% noise reduction through filtering
- **Expert Agreement**: Balanced annotation distribution achieved

### Key Findings
1. **Enhanced Preprocessing**: Significant improvement in topic coherence
2. **Quality Filtering**: Effective noise reduction without bias
3. **Expert Validation**: Consistent annotation quality across score ranges


### Development Setup
```bash
# Install development dependencies
./scripts/setup_environment.sh --dev

# Run tests
pytest tests/

# Format code
black src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

