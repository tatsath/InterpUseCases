And I love this stuff on its own but I would also be great as a vehicle for non-or pita or just a little salad research as well inspired by#!/bin/bash

echo "🚀 Finance Auto-Interp Pipeline Runner"
echo "======================================"
echo ""
echo "This script runs the complete finance-specific auto-interpretability pipeline"
echo "using Delphi with FAISS hard-negatives and finance-specific prompts."
echo ""

# Check if we're in the right directory
if [ ! -f "delphi/config.yaml" ]; then
    echo "❌ Please run this script from the paper/ directory"
    exit 1
fi

# Function to check dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found"
        exit 1
    fi
    
    # Check Delphi
    if ! python -c "import delphi" &> /dev/null; then
        echo "❌ Delphi not available. Please install: pip install -e ../delphi/"
        exit 1
    fi
    
    # Check required packages
    python -c "import faiss, sentence_transformers, yaml" 2>/dev/null || {
        echo "❌ Missing required packages. Installing..."
        pip install faiss-cpu sentence-transformers pyyaml
    }
    
    echo "✅ Dependencies check passed"
}

# Function to build FAISS index
build_faiss_index() {
    echo ""
    echo "🔨 Building FAISS Index..."
    echo "=========================="
    
    if [ -f "faiss/index.faiss" ] && [ -f "faiss/idmap.npy" ]; then
        echo "✅ FAISS index already exists, skipping build"
        return 0
    fi
    
    echo "Building FAISS index from finance spans..."
    python build_faiss.py \
        --spans data/finance_spans.jsonl \
        --out_index faiss/index.faiss \
        --out_idmap faiss/idmap.npy \
        --out_embeddings faiss/embeddings.npy
    
    if [ $? -eq 0 ]; then
        echo "✅ FAISS index built successfully"
        return 0
    else
        echo "❌ FAISS index build failed"
        return 1
    fi
}

# Function to run enhanced pipeline with F1 calculation
run_enhanced_pipeline() {
    echo ""
    echo "🚀 Running Delphi Enhanced Pipeline (with F1 calculation)..."
    echo "=========================================================="
    echo "This uses the enhanced pipeline with F1 score calculation and"
    echo "research paper approach (50 samples for interpretation, 200 for scoring)."
    echo ""
    
    read -p "Continue with enhanced pipeline (includes F1 calculation)? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python run_enhanced_pipeline.py
        return $?
    else
        echo "Skipping enhanced pipeline"
        return 0
    fi
}

# Function to run programmatic approach
run_programmatic_approach() {
    echo ""
    echo "🚀 Running Delphi Programmatic Approach..."
    echo "========================================="
    echo "This enables FAISS hard-negatives and finance-specific prompts."
    echo ""
    
    read -p "Continue with programmatic approach? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python run_delphi_programmatic.py
        return $?
    else
        echo "Skipping programmatic approach"
        return 0
    fi
}

# Function to show results
show_results() {
    echo ""
    echo "📊 Pipeline Results"
    echo "=================="
    
    # Check for enhanced pipeline results
    if [ -d "runs/enhanced_finance_autointerp_full_scale" ]; then
        echo "✅ Enhanced pipeline results found in: runs/enhanced_finance_autointerp_full_scale/"
        ls -la runs/enhanced_finance_autointerp_full_scale/
        
        # Check for F1 scores
        if [ -f "runs/enhanced_finance_autointerp_full_scale/detection_f1_scores.json" ]; then
            echo ""
            echo "📊 F1 Scores Summary:"
            python -c "
import json
with open('runs/enhanced_finance_autointerp_full_scale/detection_f1_scores.json') as f:
    scores = json.load(f)
valid_scores = [s for s in scores.values() if s is not None]
if valid_scores:
    print(f'   - Features processed: {len(valid_scores)}')
    print(f'   - Mean F1: {sum(valid_scores)/len(valid_scores):.3f}')
    print(f'   - Features with F1 > 0.65: {sum(1 for s in valid_scores if s > 0.65)}')
else:
    print('   - No valid F1 scores found')
"
        fi
    fi
    
    # Check for programmatic results
    if [ -d "runs/finance_autointerp" ]; then
        echo ""
        echo "✅ Programmatic results found in: runs/finance_autointerp/"
        ls -la runs/finance_autointerp/
        
        if [ -d "runs/finance_autointerp/explanations" ]; then
            echo ""
            echo "📝 Generated explanations:"
            ls -la runs/finance_autointerp/explanations/ | head -10
        fi
    fi
}

# Function to show next steps
show_next_steps() {
    echo ""
    echo "📋 Next Steps"
    echo "============="
    echo "1. Review generated explanations in the runs/ directory"
    echo "2. Analyze the quality of finance-specific labels"
    echo "3. Run detection scoring to evaluate performance"
    echo "4. Use results for research paper or further analysis"
    echo ""
    echo "📚 Files generated:"
    echo "   - FAISS index: faiss/index.faiss"
    echo "   - ID mapping: faiss/idmap.npy"
    echo "   - Embeddings: faiss/embeddings.npy"
    echo "   - Explanations: runs/*/explanations/"
    echo "   - F1 Scores: runs/*/detection_f1_scores.json"
    echo "   - Configuration: delphi/config.yaml"
}

# Main execution
main() {
    echo "🚀 Starting Finance Auto-Interp Pipeline..."
    echo "=========================================="
    
    # Check dependencies
    check_dependencies
    
    # Build FAISS index
    if ! build_faiss_index; then
        echo "❌ Failed to build FAISS index. Exiting."
        exit 1
    fi
    
    # Run enhanced pipeline with F1 calculation
    run_enhanced_pipeline
    
    # Run programmatic approach
    run_programmatic_approach
    
    # Show results
    show_results
    
    # Show next steps
    show_next_steps
    
    echo ""
    echo "🎉 Finance Auto-Interp Pipeline completed!"
}

# Run main function
main "$@"
