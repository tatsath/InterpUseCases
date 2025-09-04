#!/usr/bin/env python3
"""Test FAISS index functionality."""

import numpy as np
import faiss
import os

def test_faiss_index():
    """Test that the FAISS index can be loaded and used."""
    print("🔍 Testing FAISS index...")
    
    # Check if files exist
    index_file = "faiss/index.faiss"
    idmap_file = "faiss/idmap.npy"
    embeddings_file = "faiss/embeddings.npy"
    
    for file in [index_file, idmap_file, embeddings_file]:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Load the index
    try:
        index = faiss.read_index(index_file)
        print(f"✅ FAISS index loaded: {index.ntotal} vectors, {index.d} dimensions")
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False
    
    # Load embeddings
    try:
        embeddings = np.load(embeddings_file)
        print(f"✅ Embeddings loaded: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return False
    
    # Test similarity search
    try:
        # Use first embedding as query
        query = embeddings[0:1]
        D, I = index.search(query, k=3)
        print(f"✅ Similarity search works: found {len(I[0])} neighbors")
        print(f"   Distances: {D[0]}")
        print(f"   Indices: {I[0]}")
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")
        return False
    
    print("🎉 FAISS index test completed successfully!")
    return True

if __name__ == "__main__":
    test_faiss_index()
