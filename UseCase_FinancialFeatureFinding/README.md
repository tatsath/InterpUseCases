# UseCase_FinancialFeatureFinding: Advanced SAE Auto-Interpretation Pipeline

A comprehensive research pipeline for training Sparse Autoencoders (SAEs) on financial language models and performing automated feature interpretation using Delphi's advanced auto-interpretation framework.

## 🎯 **Research Overview**

This use case demonstrates a complete end-to-end pipeline for:
- Training SAEs on Llama-2-7B and FinLlama models
- Comprehensive layer-wise feature analysis
- Automated feature interpretation using Delphi with FAISS hard-negatives
- Financial domain-specific prompting and evaluation
- F1 score calculation and quality assessment

## 📁 **Pipeline Structure**

### **Phase 1: SAE Training**
- `1.0. Train_sae_scripts_finllama.sh` - SAE training for FinLlama model
- `1.1. Train_sae_scripts_llama.sh` - SAE training for Llama-2-7B model
- `1.2_llama2_7b_sae_training_results.md` - Training results and analysis

### **Phase 2: Comprehensive Analysis**
- `2.0. comprehensive_layer_analysis.py` - Layer-wise feature analysis
- `2.1. COMPREHENSIVE_ANALYSIS_README.md` - Analysis methodology
- `2.2_comprehensive_layer_analysis.json` - Detailed analysis results
- `2.3_layer_analysis_summary.csv` - Summary statistics
- `2.4_layer_analysis_summary.html` - Interactive visualization
- `2.5_cross_layer_feature_analysis.json` - Cross-layer comparisons

### **Phase 3: Financial Feature Analysis**
- `3.0.financial_feature_activation_analysis.py` - Financial feature analysis
- `3.1_financial_feature_activation_analysis.png` - Visualization results

### **Phase 4: Delphi Auto-Interpretation**
- `4.0. run_delphi_autointerp.sh` - Main Delphi pipeline
- `4.0. run_delphi_autointerp_ModelAPI.sh` - Model API version
- `4.0. Autointerp_base.sh` - Base configuration
- `4.1_test_faiss_debug.py` - FAISS debugging and testing
- `4.2_fixed_delphi_script.py` - Fixed Delphi implementation

## 🔧 **Key Components**

### **Autointerp Module**
- `autointerp/finance_autointerp.py` - Main auto-interpretation pipeline
- `autointerp/config.py` - Configuration management
- `autointerp/run_autointerp.sh` - Execution script
- `autointerp/run_cli_small.py` - CLI interface

### **Paper Implementation**
- `paper/run_delphi_programmatic.py` - Programmatic Delphi pipeline
- `paper/run_delphi_cli.py` - CLI-based approach
- `paper/build_faiss.py` - FAISS index construction
- `paper/analyze_results.py` - Results analysis
- `paper/delphi_f1_analyzer.py` - F1 score analysis
- `paper/generate_csv.py` - CSV export functionality

### **Data and Configuration**
- `paper/data/finance_spans.jsonl` - Financial text spans
- `paper/data/ontology_labels.txt` - Financial ontology
- `paper/data/fewshots.jsonl` - Few-shot examples
- `paper/delphi/config.yaml` - Delphi configuration
- `paper/delphi/enhanced_config.yaml` - Enhanced settings

### **Delphi Framework**
- Complete Delphi auto-interpretation framework
- FAISS integration for hard-negative sampling
- ContrastiveExplainer with financial prompts
- Detection and Fuzzing scorers
- Comprehensive result analysis tools

## 🚀 **Quick Start**

### **1. SAE Training**
```bash
# Train SAE on Llama-2-7B
bash 1.1. Train_sae_scripts_llama.sh

# Train SAE on FinLlama
bash 1.0. Train_sae_scripts_finllama.sh
```

### **2. Comprehensive Analysis**
```bash
# Run comprehensive layer analysis
python 2.0. comprehensive_layer_analysis.py

# Generate financial feature analysis
python 3.0.financial_feature_activation_analysis.py
```

### **3. Delphi Auto-Interpretation**
```bash
# Build FAISS index
python paper/build_faiss.py

# Run programmatic pipeline
python paper/run_delphi_programmatic.py

# Or use CLI approach
python paper/run_delphi_cli.py
```

## 📊 **Key Results**

### **SAE Training Performance**
- **Model**: Llama-2-7B with 400 latents per layer
- **Layers**: 4, 10, 16, 22, 28
- **Training Data**: WikiText-103
- **Success Rate**: 99.5%+ feature interpretation

### **Auto-Interpretation Results**
- **Framework**: Delphi with FAISS hard-negatives
- **Explainer**: Qwen2.5-72B-Instruct
- **Scorers**: DetectionScorer, FuzzingScorer
- **F1 Scores**: 0.5-0.7 range for high-quality features
- **Processing**: 400 latents with contrastive learning

### **Financial Feature Analysis**
- **Domain**: Financial and business text
- **Features**: 21 financial categories
- **Quality**: High-precision explanations
- **Coverage**: Comprehensive feature taxonomy

## 🔬 **Research Methodology**

### **SAE Training**
- Sparse Autoencoder architecture with 400 latents
- Multi-layer training (layers 4, 10, 16, 22, 28)
- WikiText-103 dataset for general language understanding
- Comprehensive evaluation and analysis

### **Auto-Interpretation**
- Delphi framework with FAISS hard-negative sampling
- ContrastiveExplainer for semantic understanding
- Financial domain-specific prompting
- F1 score evaluation and quality assessment

### **Feature Analysis**
- Layer-wise activation analysis
- Cross-layer feature comparison
- Financial domain specialization
- Comprehensive statistical analysis

## 📈 **Performance Metrics**

- **Training Time**: ~2-3 hours per layer
- **Interpretation Time**: ~10-15 seconds per latent
- **Memory Usage**: 74GB GPU memory (4x A100)
- **Success Rate**: 99.5%+ feature interpretation
- **F1 Scores**: 0.5-0.7 for high-quality features

## 🛠️ **Technical Stack**

- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model library
- **Delphi**: SAE auto-interpretation framework
- **FAISS**: Efficient similarity search
- **Qwen2.5-72B**: Large language model for interpretation
- **vLLM**: High-throughput inference engine

## 📚 **Documentation**

- `DELPHI_SAE_AUTOINTERPRETATION_README.md` - Delphi framework documentation
- `paper/README.md` - Paper implementation guide
- `paper/RESEARCH_APPROACH_ALIGNMENT.md` - Research methodology
- `autointerp/README.md` - Autointerp module documentation

## 🎯 **Research Applications**

This pipeline enables:
- **Feature Discovery**: Automated identification of meaningful neural features
- **Domain Analysis**: Understanding how models represent financial concepts
- **Model Comparison**: Comparing feature representations across models
- **Interpretability**: Providing human-readable explanations of model behavior
- **Research**: Supporting interpretability research and model analysis

## 🔗 **Related Work**

Based on the [InterpUseCases repository](https://github.com/tatsath/InterpUseCases), this implementation extends the research framework with:
- Advanced SAE training pipelines
- Comprehensive feature analysis tools
- Delphi auto-interpretation integration
- Financial domain specialization
- Production-ready evaluation frameworks

## 📄 **License**

This project is part of the InterpUseCases research collection. See the main repository for licensing information.

## 🤝 **Contributing**

This is a research implementation. For contributions or questions, please refer to the main InterpUseCases repository.
