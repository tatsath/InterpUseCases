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

### Prompt Selection Logic:

- **DEFAULT**: `SYSTEM` prompt is used for standard analysis (when `--non_activating_source` is not set or set to "random")
- **FAISS CONTRASTIVE**: `SYSTEM_CONTRASTIVE` prompt is automatically used when `--non_activating_source FAISS` is specified
- **CHAIN OF THOUGHT**: `COT` prompt can be optionally enabled for more detailed analysis

### 1. System Prompt (SYSTEM) - **DEFAULT PROMPT FOR STANDARD ANALYSIS**:

```python
SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

IMPORTANT: Focus on the SEMANTIC MEANING and CONCEPTS that the latent represents, NOT on grammatical parts of speech. Instead of saying "nouns, pronouns, prepositions", explain WHAT IDEAS, CONCEPTS, or MEANINGS the latent has learned to recognize.

- Try to produce a concise final description that explains WHAT the latent represents conceptually, not grammatically.
- Focus on the semantic patterns: what topics, concepts, entities, or ideas does this latent recognize?
- Avoid generic grammatical descriptions like "nouns, verbs, articles" - instead explain the meaning or purpose.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the semantic patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

{prompt}
"""
```

### 2. System Prompt (SYSTEM_CONTRASTIVE) - **USED FOR FAISS CONTRASTIVE SEARCH**:

```python
SYSTEM_CONTRASTIVE = """You are an AI researcher analyzing neural network activations to understand what patterns the model has learned. Your task is to provide a SHORT, PHRASE-BASED explanation of what concepts or patterns the latent represents.

CRITICAL REQUIREMENTS:
- Your explanation must be EXACTLY ONE PHRASE, no more than 10 words
- Focus on SEMANTIC CONCEPTS and MEANINGS, not grammatical parts of speech
- Do NOT use sentences or multiple phrases
- Do NOT use grammatical descriptions like "nouns, verbs, articles"
- Do NOT use generic linguistic descriptions

ANALYSIS APPROACH: You are analyzing a language model's internal representations. The latent activations you're interpreting should relate to:
- Semantic concepts (topics, entities, ideas)
- Content categories (news, fiction, technical, etc.)
- Linguistic patterns (sentiment, formality, etc.)
- Domain-specific knowledge (science, history, culture, etc.)

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>.

EXAMPLES OF GOOD EXPLANATIONS (single phrases, ≤10 words):
- "Financial earnings and market data"
- "Scientific concepts and terminology"
- "Historical events and dates"
- "Technical documentation and procedures"
- "News articles and current events"
- "Literary fiction and storytelling"
- "Educational content and explanations"
- "Business communications and reports"

EXAMPLES OF BAD EXPLANATIONS (avoid these):
- "Plot elements and character interactions" (too specific)
- "Delimiters mark significant content" (linguistic, not semantic)
- "Formal descriptors and qualifiers" (generic, not meaningful)

The last line of your response must be the formatted explanation, using [EXPLANATION]: followed by your short phrase.

Example response format:
Your analysis here...
[EXPLANATION]: Financial earnings and market data
"""
```

### 3. Chain of Thought (COT) Prompt - **FOR DETAILED ANALYSIS**:

```python
COT = """
To better find the explanation for the language patterns go through the following stages:

1.Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down general shared latents of the text examples. This could be related to the full sentence or to the words surrounding the marked words.

3. Formulate an hypothesis and write down the final explanation using [EXPLANATION]:.

"""
```

### Example Prompts from Delphi Codebase:

#### Example 1 - Idioms with Positive Sentiment:
```
Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd

[EXPLANATION]: Common idioms in text conveying positive sentiment.
```

#### Example 2 - Comparative Adjectives:
```
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the

[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.
```

#### Example 3 - Containment Objects:
```
Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".
Example 4:  Patrick: "why are you getting in the << way?>>" Later,

[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
```

### User Prompt Structure (Contrastive):
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
- **Chain of Thought**: Provides structured analysis approach
- **Real Examples**: Uses actual patterns from language models

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
- **Research Value**: Provides quantitative measure for analysis can't go away