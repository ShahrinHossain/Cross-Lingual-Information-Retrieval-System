# Cross-Lingual Information Retrieval (CLIR) System

A bilingual information retrieval system for Bengali and English news articles with support for cross-lingual search, semantic matching, and hybrid retrieval strategies.

## 🎯 Features

- **Bilingual Document Collection**: Crawls and processes Bengali and English news articles
- **Cross-Lingual Search**: Query in one language, retrieve relevant documents in both languages
- **Multiple Retrieval Models**:
  - **BM25**: Lexical retrieval with term frequency normalization
  - **TF-IDF**: Traditional term weighting
  - **Fuzzy Matching**: Handles transliteration and spelling variations
  - **Semantic Search**: Multilingual embeddings for conceptual matching
  - **Hybrid**: Combines lexical and semantic signals for optimal results
- **Query Processing**:
  - Automatic language detection
  - Translation between Bengali and English
  - Named entity recognition and mapping
  - Query normalization and expansion
- **Comprehensive Evaluation**: Precision@K, Recall@K, nDCG, MRR metrics
- **Error Analysis Tools**: Identify translation failures, NER mismatches, and model weaknesses

## 📁 Project Structure

```
.
├── crawler/                    # Web crawling and data collection
│   ├── article_extractor.py   # Extract article content from web pages
│   ├── url_discovery.py       # Discover URLs via sitemaps and RSS
│   └── validate_and_save.py   # Validate and save processed documents
│
├── clir/                       # Core CLIR system
│   ├── query_processor.py     # Query normalization and translation
│   ├── query_retrieval.py     # Document retrieval engine
│   └── evaluation.py           # Ranking and evaluation metrics
│
├── scripts/                    # Utility scripts
│   ├── build_dataset.py        # Crawl and build document corpus
│   ├── verify_dataset.py       # Validate dataset integrity
│   ├── error_analysis.py       # Analyze system failures
│   ├── model_comparison.py     # Compare retrieval models
│   ├── qrels_generator.py      # Generate evaluation queries
│   ├── labeling_tool.py        # Manual relevance labeling
│   └── verify_metrics.py       # Verify metrics against targets
│
└── data/
    ├── processed/              # Processed document corpus
    │   ├── bn.jsonl           # Bengali documents
    │   └── en.jsonl           # English documents
    └── eval/                   # Evaluation data
        ├── qrels.jsonl        # Query relevance judgments
        └── labels.csv         # Manual relevance labels
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install requests beautifulsoup4 lxml feedparser
pip install langdetect googletrans==4.0.0-rc1
pip install scikit-learn sentence-transformers
pip install numpy tqdm
```

### 1. Build the Dataset

Crawl Bengali and English news articles:

```bash
cd scripts
python build_dataset.py
```

This will create:
- `data/processed/bn.jsonl` - Bengali articles
- `data/processed/en.jsonl` - English articles

Verify the dataset:

```bash
python verify_dataset.py
```

### 2. Run Queries

```python
from clir.query_retrieval import QueryRetrievalEngine

# Initialize engine
engine = QueryRetrievalEngine(
    bangla_jsonl_path="data/processed/bn.jsonl",
    english_jsonl_path="data/processed/en.jsonl"
)

# Search with automatic language detection and translation
results = engine.search(
    user_query="ঢাকার খবর",  # Bengali query
    top_k=10,
    model="hybrid"
)

# Results contain documents in both languages
for lang, models in results.items():
    print(f"\n{lang.upper()} Results:")
    for doc in models['hybrid'][:5]:
        print(f"  - {doc['title']}")
```

### 3. Query Processing

```python
from clir.query_processor import QueryProcessor

processor = QueryProcessor()
result = processor.process("recent flood in sylhet")

print(f"Language: {result.detected_language}")
print(f"Normalized: {result.normalized_query}")
print(f"Translated: {result.translated_query}")
print(f"Entities: {result.named_entities}")
```

## 🔧 System Components

### Web Crawler

The crawler discovers and extracts articles from Bangladeshi news websites:

**Supported Sites:**
- Bengali: Prothom Alo, bdnews24, Bangla Tribune, Dhaka Post
- English: Dhaka Tribune, The Daily Star, New Age, Banglanews24

**Features:**
- Sitemap-based URL discovery
- Article extraction (title, body, date)
- Language detection and validation
- Duplicate detection

```bash
# Customize crawling
python scripts/build_dataset.py
```

### Retrieval Models

#### BM25 (Best Match 25)
- Probabilistic lexical matching
- Document length normalization
- Best for: Exact keyword queries

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Statistical term weighting
- Emphasizes rare terms
- Best for: Distinguishing unique vocabulary

#### Fuzzy Matching
- Character-level edit distance
- Handles transliteration variations
- Best for: Names and places (e.g., ঢাকা ↔ Dhaka)

#### Semantic Search
- Multilingual sentence embeddings
- Captures conceptual similarity
- Best for: Cross-lingual and paraphrase matching

#### Hybrid Model
- Combines lexical (BM25) and semantic signals
- Weighted fusion: `0.6 * BM25 + 0.4 * Semantic`
- Best for: General-purpose retrieval

### Query Processing Pipeline

1. **Language Detection**: Identify query language (Bengali/English/Mixed)
2. **Normalization**: Lowercase, remove extra spaces, clean punctuation
3. **Named Entity Recognition**: Extract person names, locations, organizations
4. **Translation**: Translate query to opposite language
5. **Query Expansion**: Add synonyms and entity mappings

### Evaluation Framework

#### Metrics

- **Precision@K**: Fraction of retrieved docs that are relevant
- **Recall@K**: Fraction of relevant docs that are retrieved
- **nDCG@K**: Normalized Discounted Cumulative Gain (position-aware)
- **MRR**: Mean Reciprocal Rank (first relevant result position)

#### Generate Evaluation Queries

```bash
# Auto-generate queries from dataset
python scripts/qrels_generator.py \
    --en_jsonl data/processed/en.jsonl \
    --bn_jsonl data/processed/bn.jsonl \
    --output data/eval/qrels.jsonl \
    --num_queries_en 10 \
    --num_queries_bn 10
```

#### Manual Labeling

```bash
# Interactive labeling tool
python scripts/labeling_tool.py \
    --queries queries.txt \
    --output data/eval/labels.csv \
    --annotator "your_name"
```

#### Verify Metrics

```bash
python scripts/verify_metrics.py \
    --qrels data/eval/qrels.jsonl \
    --model hybrid \
    --debug
```

**Target Thresholds:**
- Precision@10: ≥ 0.60
- Recall@50: ≥ 0.50
- nDCG@10: ≥ 0.50
- MRR: ≥ 0.40

## 📊 Analysis Tools

### Error Analysis

Identify system weaknesses:

```bash
python scripts/error_analysis.py \
    --queries test_queries.txt \
    --output data/eval/error_report.md
```

**Analysis Categories:**
- Translation failures (query meaning changed)
- Named entity mismatches
- Semantic vs. lexical model differences
- Cross-script ambiguities
- Code-switching issues

### Model Comparison

Compare different retrieval models:

```bash
python scripts/model_comparison.py \
    --queries test_queries.txt \
    --output data/eval/comparison_report.md \
    --analysis all
```

## 🌐 Supported Languages

- **Bengali (বাংলা)**: Full support with stemming and stopword removal
- **English**: Full support with standard NLP preprocessing
- **Mixed Queries**: Automatic detection and handling of code-switched queries

## 📝 Dataset Format

### Document Format (JSONL)

```json
{
  "title": "ঢাকায় নতুন মেট্রো রেল চালু",
  "body": "রাজধানী ঢাকায় আজ নতুন মেট্রো রেল সেবা চালু হয়েছে...",
  "url": "https://example.com/article/123",
  "date": "2024-02-15T10:30:00",
  "language": "bn",
  "tokens_count": 245
}
```

### QRELS Format (JSONL)

```json
{
  "query": "ঢাকার মেট্রো রেল",
  "relevant_urls": [
    "https://example.com/article/123",
    "https://example.com/article/456"
  ]
}
```

## 🎯 Use Cases

1. **News Monitoring**: Track news across Bengali and English sources
2. **Research**: Find relevant articles regardless of language
3. **Translation Validation**: Verify query translations work correctly
4. **Comparative Analysis**: Compare how topics are covered in different languages

## 🔬 Example Queries

### Bengali Queries
- `ঢাকার খবর` (Dhaka news)
- `বাংলাদেশের অর্থনীতি` (Bangladesh economy)
- `ক্রিকেট ম্যাচ` (Cricket match)

### English Queries
- `flood in Sylhet`
- `Bangladesh cricket team`
- `Dhaka traffic`

### Mixed Queries
- `Dhaka এর খবর` (News of Dhaka)
- `Bangladesh economy সম্পর্কে` (About Bangladesh economy)

## 🛠️ Advanced Configuration

### Customize Retrieval Weights

```python
from clir.query_retrieval import QueryRetrievalEngine

engine = QueryRetrievalEngine(
    bangla_jsonl_path="data/processed/bn.jsonl",
    english_jsonl_path="data/processed/en.jsonl"
)

# Adjust hybrid weights (default: 0.6 BM25, 0.4 Semantic)
results = engine.search(
    user_query="your query",
    model="hybrid",
    top_k=20
)
```

### Add Custom Stopwords

Edit `clir/query_processor.py`:

```python
BANGLA_STOPWORDS = {'এবং', 'বা', 'কিন্তু', ...}
ENGLISH_STOPWORDS = {'and', 'or', 'but', ...}
```

### Extend Named Entity Mapping

Add entries to `CROSS_LINGUAL_ENTITY_MAP` in `query_processor.py`:

```python
CROSS_LINGUAL_ENTITY_MAP = {
    'dhaka': 'ঢাকা',
    'ঢাকা': 'dhaka',
    'bangladesh': 'বাংলাদেশ',
    # Add more mappings
}
```

## 📈 Performance Tips

1. **Use Hybrid Model**: Best overall performance for most queries
2. **Enable Caching**: Cache embeddings for semantic search
3. **Tune Top-K**: Higher K improves recall but increases latency
4. **Filter by Date**: Add date-based filtering for recent news
5. **Batch Processing**: Process multiple queries together for efficiency

## 🐛 Troubleshooting

### All Metrics Return 0.0

```bash
# Run debug mode
python scripts/verify_metrics.py \
    --qrels data/eval/qrels.jsonl \
    --model hybrid \
    --debug
```

**Common causes:**
- URLs in QRELS don't match dataset URLs
- Dataset files are missing or empty
- Retrieval models not initialized properly

### Translation Not Working

Check if `googletrans` is properly installed:

```bash
pip install googletrans==4.0.0-rc1
```

### Slow Semantic Search

Semantic search uses sentence transformers which can be slow. Consider:
- Using smaller embedding models
- Caching embeddings
- Switching to BM25 for faster results

## 📄 License

MIT License - feel free to use for research and commercial projects

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- News sources for providing public content
- Sentence Transformers for multilingual embeddings
- Google Translate for translation services

---

**Built with ❤️ for cross-lingual information retrieval**
