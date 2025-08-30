# InterpUseCases: AI Interpretability Research Collection

A comprehensive collection of AI interpretability research use cases, demonstrating advanced techniques for understanding neural network behavior and feature evolution.

## 📁 **Available Use Cases**

### **UseCase_BERTvsFinBERT**

_Neural Feature Evolution in Domain Adaptation: A Comprehensive Analysis of BERT to FinBERT Fine-tuning_

This use case presents a complete analysis of how neural features evolve during fine-tuning from BERT to FinBERT. Using Sparse Autoencoders (SAEs), we reveal the systematic transformation of neural representations during domain adaptation. The research demonstrates systematic neural feature evolution during fine-tuning, with intuitive feature specializations that align with domain requirements and mathematical regularity in activation deltas.

**Key Results**: Complete SAE training pipeline for BERT and FinBERT, comprehensive feature comparison across 200 neural features, automated feature interpretation using Delphi and Llama 3.1 8B, activation delta analysis revealing dramatic feature transformations, and professional research documentation.

**Repository**: UseCase_BERTvsFinBERT/

### **UseCase_Trading**

_Decoding Financial Intelligence: How AI Features Drive Trading Strategy Performance_

This use case demonstrates how domain-specific fine-tuning affects feature representations in language models for financial applications. Using Sparse Autoencoders (SAEs), we extract interpretable features from BERT and FinBERT models to understand how financial fine-tuning creates specialized features that can predict market movements and price changes.

**Key Results**: 554% total return with 65% win rate in backtesting, Sharpe ratio of 7.62, maximum drawdown of only 20%, Sortino ratio of 9.66. The research identifies 1,200+ specialized financial features with dramatic improvements in FinBERT compared to BERT, particularly in final layers where financial expertise emerges.

**Repository**: UseCase_Trading/

## 🔧 **Technical Stack**

* **PyTorch**: Deep learning framework
* **Transformers**: HuggingFace model library
* **Delphi**: SAE auto-interpretation library
* **Llama 3.1 8B**: Large language model for feature interpretation
* **Qwen**: Large language model for automated feature labeling
* **WandB**: Training monitoring and experiment tracking
