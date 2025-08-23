#!/usr/bin/env python3
"""
Generate feature labels for top financial features from BERT and FinBERT
Using the same logic as interpret_bert_sae.py but for specific features
"""
import asyncio
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from safetensors import safe_open
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add the sae_autointerp directory to the path
sys.path.append('/home/nvidia/Documents/Hariom/sae_autointerp')

# Import Delphi components
from delphi.clients import Offline
from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig
from delphi.explainers import DefaultExplainer
from delphi.latents.latents import Latent, LatentRecord, ActivatingExample
from delphi.utils import load_tokenized_data

class BertSAE(torch.nn.Module):
    """Simple SAE implementation compatible with the trained weights."""
    
    def __init__(self, d_model: int = 768, d_sae: int = 200, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        
        # Initialize parameters (will be loaded from checkpoint)
        self.encoder = torch.nn.Linear(d_model, d_sae, bias=True)
        self.decoder = torch.nn.Linear(d_sae, d_model, bias=True)
        
    def encode(self, x):
        """Encode input to sparse latents."""
        pre_acts = self.encoder(x)
        top_acts, top_indices = torch.topk(pre_acts, self.k, dim=-1, sorted=False)
        return top_acts, top_indices, pre_acts
        
    def decode(self, latents):
        """Decode latents back to original space."""
        return self.decoder(latents)
        
    def forward(self, x):
        """Full forward pass."""
        top_acts, top_indices, pre_acts = self.encode(x)
        sparse_latents = torch.zeros_like(pre_acts)
        sparse_latents.scatter_(-1, top_indices, top_acts)
        reconstruction = self.decode(sparse_latents)
        return reconstruction, top_acts, top_indices

class FeatureLabelGenerator:
    def __init__(self, 
                 sae_path: str,
                 model_name: str = "bert-base-uncased",
                 target_layer: int = 6,
                 explainer_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """Initialize the feature label generator."""
        self.sae_path = sae_path
        self.model_name = model_name
        self.target_layer = target_layer
        self.explainer_model = explainer_model
        
        print(f"🔍 Feature Label Generator Setup:")
        print(f"   SAE Path: {sae_path}")
        print(f"   Model: {model_name}")
        print(f"   Target Layer: {target_layer}")
        print(f"   Explainer: {explainer_model}")
        
    def load_sae(self) -> BertSAE:
        """Load the trained SAE from safetensors."""
        print(f"\n📂 Loading SAE...")
        
        # Create SAE instance
        sae = BertSAE(d_model=768, d_sae=200, k=32)
        
        # Load weights
        with safe_open(self.sae_path, framework="pt") as f:
            # Load encoder weights and bias
            sae.encoder.weight.data = f.get_tensor("encoder.weight")
            sae.encoder.bias.data = f.get_tensor("encoder.bias")
            
            # Load decoder weights and bias
            sae.decoder.weight.data = f.get_tensor("W_dec").T  # Transpose for Linear layer
            sae.decoder.bias.data = f.get_tensor("b_dec")
            
        sae.eval()
        sae.cuda()  # Move to GPU
        print(f"   ✅ Loaded SAE with {sae.d_sae} latents, k={sae.k}")
        return sae
        
    def load_model_and_tokenizer(self):
        """Load the BERT/FinBERT model and tokenizer."""
        print(f"\n🤖 Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16
        )
        
        print(f"   ✅ Loaded {self.model_name}")
        
    def collect_activations_for_features(self, sae: BertSAE, target_features: List[int], n_samples: int = 1000) -> Dict[int, List[Tuple[str, float]]]:
        """
        Collect SAE feature activations for specific features.
        Returns a dictionary mapping feature_id -> list of (text, activation_strength).
        """
        print(f"\n💾 Collecting activations for {len(target_features)} target features...")
        
        # Load dataset
        cache_cfg = CacheConfig(
            dataset_repo="jyanimaulik/yahoo_finance_stockmarket_news",
            dataset_split="train[:1%]",
            cache_ctx_len=128,
            batch_size=8
        )
        
        tokens = load_tokenized_data(
            cache_cfg.cache_ctx_len,
            self.tokenizer,
            cache_cfg.dataset_repo,
            cache_cfg.dataset_split,
            "",  # dataset_name
            "text",  # dataset_column
            seed=42
        )
        
        # Limit to n_samples
        if len(tokens) > n_samples:
            tokens = tokens[:n_samples]
            
        feature_activations = {i: [] for i in target_features}
        feature_activity_counts = {i: 0 for i in target_features}
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(tokens), cache_cfg.batch_size):
                batch_tokens = tokens[i:i+cache_cfg.batch_size].cuda()
                
                # Get BERT layer 6 activations
                outputs = self.model(batch_tokens, output_hidden_states=True)
                layer_6_activations = outputs.hidden_states[self.target_layer + 1]  # +1 because includes embedding
                
                # Apply SAE
                batch_size, seq_len, d_model = layer_6_activations.shape
                flat_activations = layer_6_activations.view(-1, d_model)
                
                # Convert to float32 to match SAE weights
                flat_activations = flat_activations.float()
                
                top_acts, top_indices, _ = sae.encode(flat_activations)
                
                # Process each position in the batch
                for b in range(batch_size):
                    for s in range(seq_len):
                        pos_idx = b * seq_len + s
                        
                        # Get text for this position
                        token_ids = batch_tokens[b, max(0, s-5):s+6]  # Context window
                        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        
                        # Record activations for this position
                        acts = top_acts[pos_idx]  # (k,)
                        indices = top_indices[pos_idx]  # (k,)
                        
                        for act_val, feat_idx in zip(acts, indices):
                            feat_idx_item = feat_idx.item()
                            if feat_idx_item in target_features and act_val > 0.1:  # Only record significant activations
                                feature_activations[feat_idx_item].append((text, act_val.item()))
                                feature_activity_counts[feat_idx_item] += 1
                
                if (i // cache_cfg.batch_size + 1) % 10 == 0:
                    print(f"   Processed {i + len(batch_tokens)}/{len(tokens)} samples")
        
        # Sort activations by strength for each feature
        for feat_id in feature_activations:
            feature_activations[feat_id].sort(key=lambda x: x[1], reverse=True)
            
        print(f"   ✅ Collected activations for {len(feature_activations)} target features")
        
        # Print activity statistics
        active_features = [(feat_id, count) for feat_id, count in feature_activity_counts.items() if count > 0]
        active_features.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   📊 Activity Statistics:")
        print(f"      Features with activations: {len(active_features)}")
        if active_features:
            print(f"      Most active feature: {active_features[0][0]} ({active_features[0][1]} activations)")
            print(f"      Least active feature: {active_features[-1][0]} ({active_features[-1][1]} activations)")
        
        return feature_activations
        
    async def generate_explanations_for_features(self, feature_activations: Dict[int, List[Tuple[str, float]]]):
        """Generate explanations for the target features using the same logic as interpret_bert_sae.py."""
        print(f"\n🧠 Generating explanations for {len(feature_activations)} target features...")
        
        # Set up explainer with same settings as interpret_bert_sae.py
        client = Offline(
            self.explainer_model,
            max_memory=0.8,
            max_model_len=4096,
            num_gpus=1
        )
        
        explainer = DefaultExplainer(client, threshold=0.3, verbose=True)
        
        explanations = {}
        
        total_features = len(feature_activations)
        for i, (feat_id, examples) in enumerate(feature_activations.items()):
            print(f"   [{i+1:3d}/{total_features}] Explaining latent {feat_id:3d} ({len(examples)} activations)...")
            
            # Prepare examples for the explainer - use same logic as interpret_bert_sae.py
            max_examples = min(20, len(examples))  # Same as interpret_bert_sae.py
            raw_examples = examples[:max_examples]
            
            # Create proper ActivatingExample objects
            activating_examples = []
            
            # Handle cases where there are no examples (dead features)
            if len(raw_examples) == 0:
                # Create a dummy example for dead features
                dummy_text = "no activations found"
                tokens = self.tokenizer.encode(dummy_text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")[0]
                activations = torch.zeros_like(tokens, dtype=torch.float32)
                normalized_activations = torch.zeros_like(tokens, dtype=torch.float32)
                
                example = ActivatingExample(
                    tokens=tokens,
                    activations=activations,
                    normalized_activations=normalized_activations,
                    str_tokens=self.tokenizer.convert_ids_to_tokens(tokens)
                )
                activating_examples.append(example)
            else:
                for text, activation_val in raw_examples:
                    # Tokenize the text with same max_length as interpret_bert_sae.py
                    tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")[0]
                
                    # Create a simple activation pattern (highlighting the most important tokens)
                    activations = torch.zeros_like(tokens, dtype=torch.float32)
                    activations[1:-1] = activation_val * 0.1  # Distribute activation across non-special tokens
                    
                    # Create normalized activations (quantized to integers 0-10)
                    max_act = activations.max().item() if activations.max().item() > 0 else 1.0
                    normalized_activations = ((activations / max_act) * 10).long().float()
                    
                    # Create ActivatingExample
                    example = ActivatingExample(
                        tokens=tokens,
                        activations=activations,
                        normalized_activations=normalized_activations,
                        str_tokens=self.tokenizer.convert_ids_to_tokens(tokens)
                    )
                    activating_examples.append(example)
            
            # Create Latent object
            latent = Latent(
                module_name=f"encoder.layer.{self.target_layer}",
                latent_index=feat_id
            )
            
            # Create LatentRecord
            record = LatentRecord(
                latent=latent,
                train=activating_examples
            )
            
            try:
                # Generate explanation
                result = await explainer(record)
                explanation = result.explanation if hasattr(result, 'explanation') else str(result)
                
                # Create short summary
                short_summary = self.create_short_summary(explanation)
                
                explanations[feat_id] = {
                    'explanation': explanation,
                    'short_summary': short_summary,
                    'activity_count': len(examples),
                    'top_examples': examples[:10] if examples else [],
                    'avg_activation': np.mean([ex[1] for ex in examples[:10]]) if examples else 0.0
                }
                
                print(f"     → {short_summary}")
                
            except Exception as e:
                print(f"     ❌ Failed to explain feature {feat_id}: {e}")
                explanations[feat_id] = {
                    'explanation': f"Failed to generate: {e}",
                    'short_summary': "Failed to generate",
                    'activity_count': len(examples),
                    'top_examples': examples[:10] if examples else [],
                    'avg_activation': np.mean([ex[1] for ex in examples[:10]]) if examples else 0.0
                }
        
        # Clean up client
        try:
            await client.aclose()
        except:
            pass
            
        return explanations
        
    def create_short_summary(self, explanation: str) -> str:
        """Create a short summary (less than 10 words) from a long explanation."""
        # Remove common prefixes and make it concise
        explanation = explanation.lower()
        
        # Common patterns to extract key phrases
        patterns = [
            "text fragments containing",
            "text snippets containing", 
            "texts discussing",
            "text describing",
            "text containing",
            "text is",
            "text data containing",
            "texts are composed of",
            "texts describing",
            "the token",
            "a sequence of",
            "a collection of",
            "repeated sequences of",
            "repeated instances of",
            "a template with",
            "a fixed sequence of",
            "a consistent pattern of"
        ]
        
        for pattern in patterns:
            if explanation.startswith(pattern):
                explanation = explanation[len(pattern):].strip()
                break
        
        # Take first meaningful phrase (up to first period or comma)
        for sep in ['.', ',', ';', ':', ' -', ' but', ' and', ' with']:
            if sep in explanation:
                explanation = explanation.split(sep)[0].strip()
                break
        
        # Limit to ~8 words
        words = explanation.split()[:8]
        short_summary = ' '.join(words).strip()
        
        # Capitalize first letter
        if short_summary:
            short_summary = short_summary[0].upper() + short_summary[1:]
        
        return short_summary if short_summary else "Financial content patterns"

async def generate_labels_for_model(model_name: str, sae_path: str, target_features: List[int]):
    """Generate labels for a specific model."""
    print(f"\n{'='*60}")
    print(f"GENERATING LABELS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    generator = FeatureLabelGenerator(
        sae_path=sae_path,
        model_name=model_name,
        target_layer=6
    )
    
    # Load components
    sae = generator.load_sae()
    generator.load_model_and_tokenizer()
    
    # Collect activations
    activations = generator.collect_activations_for_features(sae, target_features, n_samples=1000)
    
    # Generate explanations
    explanations = await generator.generate_explanations_for_features(activations)
    
    return explanations

async def generate_all_feature_labels():
    """Generate feature labels for all top features from both models."""
    print("="*80)
    print("GENERATING FEATURE LABELS FOR BERT AND FINBERT")
    print("="*80)
    
    # Define paths
    bert_sae_path = "test_output/bert_layer6_k32_latents200/encoder.layer.6/sae.safetensors"
    finbert_sae_path = "test_output/finbert_layer6_k32_latents200/encoder.layer.6/sae.safetensors"
    
    # Get feature indices from the summary table
    # Top 20 features by improvement
    top_20_features = [127, 103, 51, 174, 115, 59, 150, 42, 54, 56, 184, 83, 43, 52, 133, 48, 106, 109, 65, 146]
    
    # Emerging features (new in FinBERT)
    emerging_features = [127, 103, 51, 174, 115, 59, 150, 42, 54, 184, 83, 43, 52, 48, 120]
    
    # Consistent features
    consistent_features = [65, 34]
    
    # Combine all unique features
    all_features = list(set(top_20_features + emerging_features + consistent_features))
    all_features.sort()
    
    print(f"Total unique features to label: {len(all_features)}")
    print(f"Features: {all_features}")
    
    # Generate BERT labels
    print(f"\n🔍 Generating BERT labels...")
    bert_explanations = await generate_labels_for_model(
        "bert-base-uncased",
        bert_sae_path,
        all_features
    )
    
    # Clear memory and wait
    print(f"\n🧹 Clearing GPU memory...")
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    await asyncio.sleep(5)  # Wait for memory to clear
    
    # Generate FinBERT labels
    print(f"\n🔍 Generating FinBERT labels...")
    finbert_explanations = await generate_labels_for_model(
        "ProsusAI/finbert",
        finbert_sae_path,
        all_features
    )
    
    # Create comparison table
    comparison_data = []
    for feature_id in all_features:
        bert_data = bert_explanations.get(feature_id, {})
        finbert_data = finbert_explanations.get(feature_id, {})
        
        # Determine feature type
        if feature_id in consistent_features:
            feature_type = "Consistent Financial"
        elif feature_id in emerging_features:
            feature_type = "Emerging Financial"
        elif feature_id in top_20_features:
            feature_type = "Top Financial"
        else:
            feature_type = "Other"
        
        comparison_data.append({
            'feature_idx': feature_id,
            'feature_type': feature_type,
            'bert_label': bert_data.get('short_summary', f'Feature_{feature_id}'),
            'finbert_label': finbert_data.get('short_summary', f'Feature_{feature_id}'),
            'bert_full_explanation': bert_data.get('explanation', f'Feature_{feature_id}'),
            'finbert_full_explanation': finbert_data.get('explanation', f'Feature_{feature_id}'),
            'bert_activity_count': bert_data.get('activity_count', 0),
            'finbert_activity_count': finbert_data.get('activity_count', 0),
            'bert_avg_activation': bert_data.get('avg_activation', 0),
            'finbert_avg_activation': finbert_data.get('avg_activation', 0)
        })
    
    # Create DataFrame and save
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('feature_idx')
    
    # Save detailed comparison
    df_comparison.to_csv('/home/nvidia/Documents/Hariom/saetrain/feature_labels_comparison.csv', index=False)
    
    # Create summary table for the blog post
    summary_data = []
    for _, row in df_comparison.iterrows():
        summary_data.append({
            'Feature': row['feature_idx'],
            'Type': row['feature_type'],
            'BERT Label': row['bert_label'],
            'FinBERT Label': row['finbert_label']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('/home/nvidia/Documents/Hariom/saetrain/feature_labels_summary.csv', index=False)
    
    # Save raw explanations as JSON
    explanations_data = {
        'bert_explanations': bert_explanations,
        'finbert_explanations': finbert_explanations,
        'feature_types': {
            'top_20': top_20_features,
            'emerging': emerging_features,
            'consistent': consistent_features
        }
    }
    
    with open('/home/nvidia/Documents/Hariom/saetrain/feature_explanations_data.json', 'w') as f:
        json.dump(explanations_data, f, indent=2, default=str)
    
    # Print summary
    print(f"\n" + "="*80)
    print("FEATURE LABELS SUMMARY")
    print("="*80)
    print(f"{'Feature':<8} {'Type':<20} {'BERT Label':<30} {'FinBERT Label':<30}")
    print("-" * 90)
    
    for _, row in df_summary.iterrows():
        print(f"{row['Feature']:<8} {row['Type']:<20} {row['BERT Label']:<30} {row['FinBERT Label']:<30}")
    
    print(f"\nFiles generated:")
    print(f"  - feature_labels_comparison.csv: Detailed comparison with explanations and activity counts")
    print(f"  - feature_labels_summary.csv: Summary table for blog post")
    print(f"  - feature_explanations_data.json: Raw explanation data")
    
    return df_comparison, df_summary

if __name__ == "__main__":
    asyncio.run(generate_all_feature_labels())
