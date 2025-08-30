# InterpUseCases: AI Interpretability Research Collection

A comprehensive collection of AI interpretability research use cases, demonstrating advanced techniques for understanding neural network behavior and feature evolution.

## 📁 **Available Use Cases**

### **UseCase_BERTvsFinBERT**

_Neural Feature Evolution in Domain Adaptation: A Comprehensive Analysis of BERT to FinBERT Fine-tuning_

**Description**: This use case presents a complete analysis of how neural features evolve during fine-tuning from BERT to FinBERT. Using Sparse Autoencoders (SAEs), comprehensive feature comparison, and automated neural interpretation, we reveal the systematic transformation of neural representations during domain adaptation.

**Key Features**:

* Complete SAE training pipeline for BERT and FinBERT
* Comprehensive feature comparison across 200 neural features
* Automated feature interpretation using Delphi and Llama 3.1 8B
* Activation delta analysis revealing dramatic feature transformations
* Professional research blog post with methodology and findings

**Files**: 17 files organized in 5 research stages

* Training scripts and SAE configuration
* Feature analysis and comparison tools
* Neural interpretation and labeling pipeline
* Complete documentation and visualizations

**Repository**: UseCase_BERTvsFinBERT/

### **UseCase_Trading**

_Decoding Financial Intelligence: How AI Features Drive Trading Strategy Performance_

**Description**: This use case demonstrates how domain-specific fine-tuning affects feature representations in language models for financial applications. Using Sparse Autoencoders (SAEs), we extract interpretable features from BERT and FinBERT models to understand how financial fine-tuning creates specialized features that can predict market movements and price changes. The research identifies key financial features that show dramatic improvements in FinBERT compared to BERT, with backtesting results showing 554% total return and 65% win rate.

**Key Features**:

* Financial feature extraction and analysis pipeline
* Trading strategy development and implementation
* Comprehensive backtesting with exceptional performance metrics (554% total return, 65% win rate)
* Systematic trading framework with risk management
* Market prediction capabilities using neural features
* Multi-GPU SAE training infrastructure
* Automated feature labeling using Qwen LLM

**Key Findings**:

* **Feature Specialization**: FinBERT shows 1-4% additional financial focus with 1,200+ specialized features
* **Layer-wise Evolution**: Financial expertise emerges primarily in final layers (Layer 11 shows 4.6% max specialization)
* **Trading Performance**: 554% total return with 65% win rate, Sharpe ratio of 7.62
* **Risk Management**: Maximum drawdown of only 20% with Sortino ratio of 9.66

**Files**: 25+ files organized in 6 research stages

* BERT vs FinBERT SAE comparison analysis
* Financial feature evolution analysis across all layers
* LLM-based feature labeling pipeline
* Mega FinBERT model with comprehensive financial dataset
* Backtesting framework and performance analysis
* Multi-GPU training infrastructure

**Repository**: UseCase_Trading/

## 🎯 **Research Impact**

This collection demonstrates:

1. **Systematic neural feature evolution** during fine-tuning
2. **Intuitive feature specializations** that align with domain requirements
3. **Mathematical regularity** in activation deltas and domain relevance
4. **Interpretable transformations** that can be described in natural language
5. **Practical applications** in financial trading with exceptional performance metrics
6. **Domain-specific feature engineering** for real-world applications

## 🔧 **Technical Stack**

* **PyTorch**: Deep learning framework
* **Transformers**: HuggingFace model library
* **Delphi**: SAE auto-interpretation library
* **Llama 3.1 8B**: Large language model for feature interpretation
* **Qwen**: Large language model for automated feature labeling
* **WandB**: Training monitoring and experiment tracking
* **Financial Libraries**: Backtesting frameworks and market data analysis
* **Multi-GPU Training**: Distributed SAE training infrastructure

## 📈 **Future Use Cases**

This repository will continue to grow with additional interpretability research:

* Cross-domain feature evolution analysis
* Temporal feature tracking during training
* Architectural comparison studies
* Intervention and ablation studies
* Real-time trading system integration
* Cross-asset class feature analysis
* Regulatory compliance and explainability frameworks

## 📝 **Citation**

If you use this research or code, please cite:

```
Neural Feature Evolution in Domain Adaptation: A Comprehensive Analysis of BERT to FinBERT Fine-tuning
Decoding Financial Intelligence: How AI Features Drive Trading Strategy Performance
```

## 🤝 **Contributing**

We welcome contributions of new interpretability use cases. Please ensure:

* Complete code and documentation
* Reproducible research methodology
* Clear file organization
* Comprehensive README

---

_Advancing AI interpretability through systematic research and open collaboration._
