# Finance Auto-Interp Pipeline

This folder contains a complete, end-to-end implementation of finance-specific auto-interpretability using Delphi, with FAISS hard-negatives and domain-specific prompts.

## 🚀 Quick Start

```bash
# Make the runner executable
chmod +x run_finance_autointerp.sh

# Run the complete pipeline
./run_finance_autointerp.sh
```

## 📁 Folder Structure

```
paper/
├── README.md                           # This file
├── run_finance_autointerp.sh          # Main pipeline runner
├── run_delphi_cli.py                  # CLI approach (supported flags only)
├── run_delphi_programmatic.py         # Programmatic approach (FAISS + prompts)
├── build_faiss.py                     # FAISS index builder
├── delphi/
│   └── config.yaml                    # Central configuration
├── data/
│   ├── finance_spans.jsonl            # Finance text spans
│   ├── ontology_labels.txt            # Finance label ontology
│   └── fewshots.jsonl                 # Example prompts
├── faiss/                             # FAISS index files (generated)
└── runs/                              # Output results (generated)
```

## 🎯 What This Pipeline Does

### 1. **CLI Approach** (Supported Flags Only)
- Uses Delphi's default end-to-end pipeline
- Processes activations → explanations → detection scoring
- **Limitation**: No FAISS hard-negatives, generic prompts

### 2. **Programmatic Approach** (Full Control)
- Enables FAISS hard-negatives for contrastive learning
- Uses finance-specific prompts and label constraints
- Provides ContrastiveExplainer with domain knowledge
- **Advantage**: Better explanations, domain-specific labels

## 🔧 Configuration

Edit `delphi/config.yaml` to customize:

```yaml
# Model Configuration
model: "meta-llama/Llama-2-7b-hf"
sparse_model: "/path/to/your/sae/model"

# Layer Configuration
hookpoints: ["layers.16"]
max_latents: 100

# FAISS Configuration
faiss:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  cache_enabled: true
  cache_dir: ".embedding_cache"

# Explainer Configuration
explainer:
  model: "Qwen/Qwen2.5-72B-Instruct"
  provider: "offline"
  max_len: 8192
  temperature: 0.2
  num_gpus: 4
```

## 📊 Finance Labels

The pipeline includes 21 finance-specific categories:

- **Performance**: EARNINGS_BEAT, GUIDANCE_UP, STOCK_PERFORMANCE
- **Risk**: DOWNGRADE, CREDIT_DOWNGRADE, REGULATORY_RISK
- **Events**: LAYOFFS, M&A_RUMOR, CFO_CHANGE
- **Macro**: MACRO_INFLATION, RATE_CUT, FX_HEADWIND
- **Other**: COMPANY_NEWS, ECONOMIC_INDICATOR, OTHER

## 🏗️ How It Works

### Step 1: Build FAISS Index
```bash
python build_faiss.py \
  --spans data/finance_spans.jsonl \
  --out_index faiss/index.faiss \
  --out_idmap faiss/idmap.npy \
  --out_embeddings faiss/embeddings.npy
```

**What this does:**
- Loads finance text spans from JSONL
- Encodes using sentence transformers
- Builds FAISS index for semantic similarity
- Enables hard-negative sampling in Delphi

### Step 2: Run CLI Approach
```bash
python run_delphi_cli.py
```

**What this does:**
- Uses Delphi's supported CLI flags only
- Runs default pipeline (cache → explain → detect)
- Saves results to `runs/` directory

### Step 3: Run Programmatic Approach
```bash
python run_delphi_programmatic.py
```

**What this does:**
- Enables FAISS hard-negatives via ConstructorConfig
- Uses ContrastiveExplainer with finance prompts
- Generates domain-specific explanations
- Saves structured JSON results

## 🔍 Key Features

### ✅ **Officially Supported by Delphi**
- CLI flags: `--n_tokens`, `--max_latents`, `--hookpoints`, `--scorers`
- FAISS integration: `ConstructorConfig(non_activating_source="FAISS")`
- ContrastiveExplainer: Automatically enabled with FAISS
- Supported scorers: RecallScorer, FuzzingScorer

### ✅ **Finance-Specific Prompts**
- Constrained label space (21 categories)
- Domain-specific examples and few-shots
- Semantic focus (not grammatical)
- JSON output format for structured results

### ✅ **Hard-Negative Sampling**
- FAISS-based semantic similarity
- Contrastive learning for better explanations
- Mix of easy and hard negative examples
- Improved explanation quality

## 🚀 Running the Pipeline

### Option 1: Complete Pipeline
```bash
./run_finance_autointerp.sh
```

### Option 2: Individual Steps
```bash
# Build FAISS index
python build_faiss.py

# Run CLI approach
python run_delphi_cli.py

# Run programmatic approach
python run_delphi_programmatic.py
```

### Option 3: Custom Configuration
```bash
# Edit config.yaml first, then run
python run_delphi_programmatic.py
```

## 📈 Expected Results

### CLI Approach
- **Output**: `runs/llama2_7b_layer16_finance_autointerp/`
- **Format**: Standard Delphi explanations
- **Time**: ~15-30 minutes for 100 latents
- **Quality**: Generic, grammatical focus

### Programmatic Approach
- **Output**: `runs/finance_autointerp/`
- **Format**: JSON with structured labels
- **Time**: ~30-60 minutes for 100 latents
- **Quality**: Domain-specific, semantic focus

## 🔧 Troubleshooting

### Common Issues
1. **Delphi not found**: `pip install -e ../delphi/`
2. **Missing packages**: `pip install faiss-cpu sentence-transformers pyyaml`
3. **FAISS build fails**: Check spans file format and dependencies
4. **GPU memory**: Reduce `num_gpus` in config

### Debug Mode
```bash
# Check dependencies
python -c "import delphi, faiss, sentence_transformers, yaml"

# Test FAISS build
python build_faiss.py --help

# Validate config
python -c "import yaml; yaml.safe_load(open('delphi/config.yaml'))"
```

## 📚 Research Use Cases

### 1. **Domain Generalization**
- Test SAE performance on financial vs. general text
- Compare explanation quality across domains
- Analyze feature robustness

### 2. **Label Quality Analysis**
- Evaluate finance-specific vs. generic explanations
- Measure detection F1 scores
- Analyze label distribution

### 3. **Feature Interpretation**
- Understand what financial concepts SAE learns
- Identify domain-specific vs. general features
- Analyze feature activation patterns

## �� Paper-Ready Outputs

The pipeline generates:

1. **Structured Explanations**: JSON with labels and rationales
2. **FAISS Index**: Semantic similarity for hard-negatives
3. **Configuration**: Reproducible experiment setup
4. **Results**: Organized output for analysis

## 🔬 Extending the Pipeline

### Add New Labels
Edit `data/ontology_labels.txt` and `data/fewshots.jsonl`

### Change Models
Modify `delphi/config.yaml` explainer settings

### Add Scorers
Extend `run_delphi_programmatic.py` with new scorer pipes

### Custom Prompts
Modify `create_finance_prompt()` method

## 📖 References

- **Delphi Documentation**: [Official README](https://github.com/EleutherAI/sae-auto-interp)
- **FAISS Integration**: Supported via ConstructorConfig
- **Contrastive Explainer**: Automatically enabled with FAISS
- **Supported Scorers**: Recall, Fuzzing, Simulation, Surprisal

## 🆘 Support

- **CLI Issues**: Check Delphi documentation
- **API Issues**: Review Delphi README
- **Configuration**: Modify `delphi/config.yaml`
- **Customization**: Edit prompt templates in scripts

---

**Note**: This pipeline uses only officially supported Delphi features. All flags and API calls are documented and tested.
