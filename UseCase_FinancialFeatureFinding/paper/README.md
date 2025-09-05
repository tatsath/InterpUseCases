# Delphi Auto-Interp Pipeline 

## 🚀 Quick Start

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY='your_api_key_here'

# Make the script executable
chmod +x run_delphi_working_openrouter.sh

# Run the complete pipeline
./run_delphi_working_openrouter.sh
```

## 📊 Script Overview

The main script `run_delphi_working_openrouter.sh` provides:

- ✅ **OpenRouter API Integration** - No local GPU memory issues
- ✅ **FAISS Contrastive Learning** - Better explanations with hard negatives
- ✅ **Automatic Versioning** - Each run gets a unique version number
- ✅ **CSV Output Generation** - Structured results with Feature_ID, Label, F1_Score

## 🔧 Configuration Parameters

| Parameter | Value | Purpose | Impact |
|-----------|-------|---------|---------|
| `--n_tokens` | 10000 | Number of tokens to process | More tokens = better coverage, longer runtime |
| `--max_latents` | 5 | Number of latent features to analyze | More latents = comprehensive analysis |
| `--hookpoints` | layers.16 | Which model layer to analyze | Layer 16 = middle layer with rich representations |
| `--scorers` | detection | Scoring method (F1-based) | Detection scoring measures explanation accuracy |
| `--explainer_model` | openai/gpt-3.5-turbo | AI model for explanations | GPT-3.5-turbo = good balance of quality/speed |
| `--explainer_provider` | openrouter | API provider | OpenRouter = no local GPU memory needed |
| `--explainer_model_max_len` | 512 | Max context length | Shorter = faster, longer = more context |
| `--n_non_activating` | 50 | Number of negative examples | More negatives = better contrastive learning |
| `--non_activating_source` | FAISS | Method for finding negatives | FAISS = semantic similarity-based hard negatives |
| `--dataset_repo` | wikitext | Dataset source | Wikitext = general knowledge text |
| `--dataset_name` | wikitext-103-raw-v1 | Specific dataset | Raw text without preprocessing |
| `--dataset_split` | train[:1%] | Dataset subset | 1% = ~4M tokens, good for testing |

## 🧠 FAISS Contrastive Learning

### How FAISS Works:
1. **Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` to create text embeddings
2. **Similarity Search**: Builds FAISS index of non-activating examples
3. **Hard Negative Selection**: Finds semantically similar but non-activating examples
4. **Contrastive Prompting**: Shows both activating and non-activating examples to the AI

### Value Add:
- **Better Explanations**: AI can distinguish between similar-looking content
- **Semantic Understanding**: Focuses on meaning, not just surface patterns
- **Robust Features**: Reduces false positives and improves accuracy

## 📝 Prompt Engineering

### System Prompt (SYSTEM_CONTRASTIVE):
```
You are an AI researcher analyzing neural network activations to understand what patterns the model has learned. Your task is to provide a SHORT, PHRASE-BASED explanation of what concepts or patterns the latent represents.

CRITICAL REQUIREMENTS:
- Your explanation must be EXACTLY ONE PHRASE, no more than 10 words
- Focus on SEMANTIC CONCEPTS and MEANINGS, not grammatical parts of speech
- Do NOT use sentences or multiple phrases
- Do NOT use grammatical descriptions like "nouns, verbs, articles"

EXAMPLES OF GOOD EXPLANATIONS:
- "Financial earnings and market data"
- "Scientific concepts and terminology"
- "Historical events and dates"
- "Literary content and themes"
```

### User Prompt Structure:
```
ACTIVATING EXAMPLES:
Example 1: <<financial>> <<earnings>> report shows <<strong>> <<growth>>
Example 2: <<market>> <<data>> indicates <<positive>> <<trends>>

NON-ACTIVATING EXAMPLES:
Example 1: The weather today is sunny and warm
Example 2: I went to the store to buy groceries
```

### Prompt Benefits:
- **Conciseness**: Forces focused, single-phrase explanations
- **Semantic Focus**: Emphasizes meaning over grammar
- **Contrastive Learning**: Shows what activates vs. what doesn't
- **Domain Agnostic**: Works across different text types

## 📈 F1 Score & Detection Scoring

### How Detection Scoring Works:
1. **Explanation Generation**: AI generates explanation for each latent
2. **Test Set Creation**: Creates test examples with known activations
3. **Classification**: Tests if explanation correctly identifies activating examples
4. **F1 Calculation**: Measures precision and recall of the explanation

### F1 Score Interpretation:
- **0.0-0.3**: Poor explanation, low accuracy
- **0.3-0.6**: Moderate explanation, some accuracy
- **0.6-0.8**: Good explanation, high accuracy
- **0.8-1.0**: Excellent explanation, very high accuracy

### Why F1 Score Matters:
- **Quality Metric**: Measures how well the explanation captures the feature
- **Comparability**: Allows comparison across different latents
- **Validation**: Confirms the explanation is actually useful
- **Research Value**: Provides quantitative measure for analysis