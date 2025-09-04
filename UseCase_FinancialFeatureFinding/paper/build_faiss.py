#!/usr/bin/env python3
"""
Build FAISS Index for Hard Negatives
====================================

This script builds a FAISS index from finance spans to enable hard-negative
sampling during auto-interpretability. It creates semantically similar but
non-activating examples for contrastive learning.

Based on Delphi's official FAISS support for hard-negatives.
"""

import json
import numpy as np
import faiss
import argparse
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_spans(spans_file):
    """Load finance spans from JSONL file."""
    print(f"📖 Loading spans from {spans_file}...")
    
    ids, texts = [], []
    with open(spans_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                ids.append(obj["id"])
                texts.append(obj["text"])
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid JSON at line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"⚠️  Warning: Missing key at line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(ids)} spans")
    return ids, texts

def encode_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Encode texts using sentence transformers."""
    print(f"🔧 Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)
    
    print("📝 Encoding texts...")
    embeddings = encoder.encode(
        texts, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    print(f"✅ Encoded {embeddings.shape[0]} texts to {embeddings.shape[1]} dimensions")
    return embeddings

def build_faiss_index(embeddings, ids, output_index, output_idmap, output_embeddings=None):
    """Build and save FAISS index."""
    print("🔨 Building FAISS index...")
    
    # Build FAISS IP (Inner Product) index for cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings (FAISS expects float32)
    index.add(embeddings.astype(np.float32))
    
    # Save index
    print(f"💾 Saving FAISS index to {output_index}")
    faiss.write_index(index, output_index)
    
    # Save ID mapping
    print(f"💾 Saving ID mapping to {output_idmap}")
    np.save(output_idmap, np.array(ids))
    
    # Optionally save embeddings for later use
    if output_embeddings:
        print(f"💾 Saving embeddings to {output_embeddings}")
        np.save(output_embeddings, embeddings)
    
    print(f"✅ FAISS index built successfully!")
    print(f"   - Index file: {output_index}")
    print(f"   - ID mapping: {output_idmap}")
    if output_embeddings:
        print(f"   - Embeddings: {output_embeddings}")
    print(f"   - Total vectors: {index.ntotal}")
    print(f"   - Vector dimension: {dimension}")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for finance spans")
    parser.add_argument("--spans", default="data/finance_spans.jsonl",
                       help="Input finance spans JSONL file")
    parser.add_argument("--out_index", default="faiss/index.faiss",
                       help="Output FAISS index file")
    parser.add_argument("--out_idmap", default="faiss/idmap.npy",
                       help="Output ID mapping file")
    parser.add_argument("--out_embeddings", default="faiss/embeddings.npy",
                       help="Output embeddings file (optional)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model to use")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_idmap), exist_ok=True)
    if args.out_embeddings:
        os.makedirs(os.path.dirname(args.out_embeddings), exist_ok=True)
    
    # Load spans
    ids, texts = load_spans(args.spans)
    
    if not ids:
        print("❌ No valid spans found. Exiting.")
        return
    
    # Encode texts
    embeddings = encode_texts(texts, args.model)
    
    # Build FAISS index
    build_faiss_index(
        embeddings, 
        ids,
        args.out_index, 
        args.out_idmap, 
        args.out_embeddings
    )
    
    print("\n🎉 FAISS index construction completed!")
    print("\n📋 Next steps:")
    print("1. Use the index in Delphi with ConstructorConfig(non_activating_source='FAISS')")
    print("2. The index will automatically provide hard-negatives for contrastive learning")
    print("3. Delphi will switch to ContrastiveExplainer when FAISS is enabled")

if __name__ == "__main__":
    main()
