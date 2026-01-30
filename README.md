# GDELT Cleaning & Embedding Pipeline

> State-of-the-art semantic indexing of the GDELT Global Database of Events, Language, and Tone.

## 🎯 Mission

GDELT is the world's largest open dataset of global events, updated every 15 minutes. But raw GDELT data is noisy, redundant, and lacks semantic structure. This project transforms raw GDELT into a **semantically searchable, embedding-indexed knowledge base** — enabling millisecond retrieval of geopolitical events by meaning, not just keywords.

**We do NOT duplicate GDELT.** We clean, embed, and index — creating lightweight semantic layers that point back to source data.

## 🏗️ Architecture

```
GDELT API (Events 2.0 + GKG)
        │
        ▼
┌─────────────────────┐
│  Ingestion Layer     │  ← api-ingestion-framework (Databricks)
│  (Raw → Bronze)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cleaning Pipeline   │  ← THIS REPO
│  • Deduplication     │
│  • Entity Resolution │
│  • Text Normalization│
│  • Quality Scoring   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Embedding Layer     │
│  • Modern Transformer│
│    Embeddings        │
│  • NOT USEv4 (too    │
│    weak for prod)    │
│  • Candidate models: │
│    - BGE-M3          │
│    - E5-large-v2     │
│    - Cohere embed-v3 │
│    - Voyage-3        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Index & Retrieval   │
│  • Vector DB         │
│    (FAISS / Qdrant / │
│     Databricks VS)   │
│  • Semantic Search   │
│  • Temporal Filtering│
│  • Geo-spatial Index │
└─────────────────────┘
```

## 🔑 Design Principles

1. **No data duplication** — We index and embed, we don't clone GDELT. Embeddings + metadata = lightweight.
2. **Databricks-native** — Everything runs on Databricks. Delta Lake, Unity Catalog, MLflow for model tracking.
3. **Continuous pipeline** — Not a one-shot script. Cron-based ingestion every 15 minutes, incremental embedding.
4. **Embedding model agnostic** — Pluggable embedding layer. Start with open-source, benchmark, upgrade.
5. **Production-grade** — Tests, monitoring, retry logic, data quality checks.

## 📋 Roadmap

### Phase 1: Data Cleaning (Current)
- [ ] Connect to api-ingestion-framework Bronze layer
- [ ] Deduplication pipeline (GDELT has ~15-30% redundancy across updates)
- [ ] Entity normalization (actor codes → canonical names)
- [ ] Text cleaning for source URLs and event descriptions
- [ ] Data quality scoring (completeness, consistency, freshness)
- [ ] Output: Clean Silver-layer Delta tables

### Phase 2: Embedding & Indexing
- [ ] Benchmark embedding models on GDELT event descriptions
- [ ] **Prefer Matryoshka-compatible models** (text-embedding-3, BGE-M3) for truncation-friendly storage
- [ ] Build embedding pipeline (batch + incremental)
- [ ] **Storage Optimization Pipeline** (see below):
  - [ ] Scalar Quantization (SQ8): float32 → int8 (4x reduction, <1% accuracy loss)
  - [ ] Matryoshka Truncation: Store first 256 dims of 1024 (4x reduction, ~98% recall)
  - [ ] Combined: ~16x reduction (1.5 TB → ~100 GB)
- [ ] **Two-Stage Funnel Search**:
  - [ ] Stage 1 (Shortlist): Binary/PCA-reduced index for fast top-100 candidates
  - [ ] Stage 2 (Rerank): Full-precision vectors for final top-5 selection
- [ ] Vector DB setup (evaluate Databricks Vector Search vs. self-hosted)
- [ ] Temporal + geo-spatial index layers
- [ ] Output: Searchable vector index with metadata filters

### Phase 2b: Storage Architecture
- [ ] **Cold Storage**: Azure Blob / S3 Standard-IA for quantized embeddings (~$5-10/Mo for 100 GB)
- [ ] **Hot Compute**: On-demand copy to Databricks DBFS / SSD for analysis sessions
- [ ] **Lifecycle**: Cold → Hot (1-2 min transfer) → Analyze → Flush hot storage
- [ ] **Cost Target**: <$20/Mo storage, $0 when idle (no persistent hot compute)
- [ ] Egress-Kalkulation: Immer VOR großen Downloads/Transfers durchrechnen

### Phase 3: Retrieval API
- [ ] Semantic search endpoint
- [ ] Temporal range queries ("events in Syria last 72h")
- [ ] Actor-based queries ("all Russia-Ukraine interactions")
- [ ] Goldstein scale filtering (conflict intensity)
- [ ] Integration with Vector News AI platform

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Databricks Workflows / Delta Live Tables |
| Storage | Delta Lake (Unity Catalog) |
| Compute | Databricks (Spark + single-node for embedding) |
| Embeddings | TBD — benchmarking BGE-M3, E5-large-v2, Voyage-3 |
| Vector DB | TBD — evaluating Databricks VS, Qdrant, FAISS |
| Monitoring | MLflow + Great Expectations |
| Language | Python 3.11+ |

## 💾 Storage Optimization Strategy

Raw embeddings (float32, 1024 dims) for full GDELT history = ~1.5 TB. Unacceptable for a lean setup.

**Three-layer compression pipeline:**

| Technique | Reduction | Accuracy Impact | How |
|-----------|-----------|-----------------|-----|
| **Matryoshka Truncation** | 4x | <2% recall loss | Store first 256 of 1024 dims (MRL-trained models) |
| **Scalar Quantization (SQ8)** | 4x | <1% recall loss | float32 → int8 per dimension |
| **Product Quantization (PQ)** | 20-50x | ~5% recall loss | Dictionary-based sub-vector compression |
| **Combined (MRL + SQ8)** | **~16x** | **<3%** | 1.5 TB → **~100 GB** |

**Funnel Search Architecture:**
```
Query
  │
  ▼
┌────────────────────────┐
│ Stage 1: Shortlist     │  Binary/PCA-reduced index (~50 GB)
│ → Top 100 candidates   │  Latency: <10ms
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ Stage 2: Rerank        │  Full MRL-256 + SQ8 vectors
│ → Top 5 results        │  Latency: <50ms
└────────────────────────┘
```

**Storage Tiers:**
- **Cold** (Azure Blob / S3-IA): Quantized embeddings, ~$5-10/Mo for 100 GB
- **Hot** (Databricks DBFS / local SSD): On-demand for active analysis, flush after use
- **Transfer**: Cold → Hot in 1-2 min, acceptable startup cost

> 💡 **Key Insight:** Matryoshka embeddings (text-embedding-3, BGE-M3) front-load information into early dimensions. Truncating to 25% of dims preserves >98% retrieval quality.

## 📊 Why Not USEv4?

We evaluated Google's Universal Sentence Encoder v4 for GDELT embeddings. Results:
- Poor semantic separation for geopolitical event types
- Weak on multilingual content (GDELT is global)
- Embedding dimension (512) insufficient for fine-grained similarity
- Modern alternatives (BGE-M3, E5) outperform by 15-25% on MTEB benchmarks

→ We embed with state-of-the-art models from day one.

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/HedgingHarald/gdelt-cleaning-pipeline.git
cd gdelt-cleaning-pipeline

# Setup (requires Databricks CLI configured)
pip install -e ".[dev]"

# Run cleaning pipeline
python -m src.pipeline.run --source events_v2 --mode incremental
```

## 📁 Project Structure

```
gdelt-cleaning-pipeline/
├── README.md
├── pyproject.toml
├── databricks.yml
├── src/
│   ├── cleaning/          # Dedup, normalization, quality
│   ├── embedding/         # Model loading, batch embedding
│   ├── indexing/          # Vector DB operations
│   └── pipeline/          # Orchestration, scheduling
├── configs/
│   ├── cleaning.yaml      # Cleaning rules & thresholds
│   └── embedding.yaml     # Model config & hyperparams
├── notebooks/             # Databricks notebooks
├── tests/
└── benchmarks/            # Embedding model comparisons
```

## 📝 License

MIT

---

*Part of the Vector News AI ecosystem. Built for speed, accuracy, and scale.*
