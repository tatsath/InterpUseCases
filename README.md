# UseCase_BERTvsFinBERT: Complete Research Pipeline

This folder contains all the code, data, and documentation used in the comprehensive analysis of neural feature evolution from BERT to FinBERT. The files are organized sequentially by research stage.

## 📁 **File Organization**

### **Stage 1: Sparse Autoencoder Training**
*Training SAEs on BERT and FinBERT models to extract interpretable features*

- **1.0_Train_BERT_SAE.sh**: Shell script to train SAE on BERT's 6th layer
- **1.1_Train_FinBERT_SAE.sh**: Shell script to train SAE on FinBERT's 6th layer
- **1.2_saetrain_package/**: Custom SAE training package with evaluation metrics

### **Stage 2: Feature Comparison and Analysis**
*Comprehensive comparison of all 200 features between BERT and FinBERT*

- **2.0_financial_feature_activation_analysis.py**: Main analysis script testing features on 30 sentences across 6 categories
- **2.1_financial_features_summary_table.md**: Top 20 financial features with improvement statistics
- **2.2_financial_feature_analysis_comprehensive.csv**: Complete analysis of all 200 features
- **2.3_top_financial_features_table.csv**: Focused table of top financial features
- **2.4_financial_feature_analysis_summary.json**: Statistical summary of the analysis

### **Stage 3: Feature Ranking and Delta Analysis**
*Identification and ranking of features by activation improvement*

- **3.0_feature_labels_by_activation_delta.md**: Features ranked by activation improvement magnitude
- **3.1_financial_feature_activation_analysis.png**: Visualizations of feature comparison

### **Stage 4: Neural Interpretation and Labeling**
*Automated generation of natural language explanations for neural features*

- **4.0_generate_feature_labels_simple.py**: BERT feature labeling script using Delphi
- **4.1_generate_finbert_labels_only.py**: FinBERT feature labeling and combination script
- **4.2_feature_labels_comparison.csv**: Detailed comparison with full explanations
- **4.3_feature_labels_summary.csv**: Summary table for analysis
- **4.4_feature_explanations_data.json**: Raw explanation data from Llama 3.1 8B

### **Stage 5: Final Analysis and Documentation**
*Compilation of all results and final documentation*

- **5.0_final_feature_labels_summary.md**: Comprehensive analysis and insights
- **5.1_finbert_feature_evolution_blog_post.md**: Complete research blog post

## 🚀 **How to Use This Pipeline**

### **Step 1: Train the SAEs**
```bash
# Train SAE on BERT
bash 1.0_Train_BERT_SAE.sh

# Train SAE on FinBERT
bash 1.1_Train_FinBERT_SAE.sh
```

### **Step 2: Run Feature Comparison**
```bash
# Run comprehensive feature analysis
python 2.0_financial_feature_activation_analysis.py
```

### **Step 3: Generate Feature Labels**
```bash
# Generate BERT feature labels
python 4.0_generate_feature_labels_simple.py

# Generate FinBERT feature labels
python 4.1_generate_finbert_labels_only.py
```

### **Step 4: Review Results**
- Check `2.1_financial_features_summary_table.md` for top features
- Review `3.0_feature_labels_by_activation_delta.md` for activation improvements
- Examine `4.2_feature_labels_comparison.csv` for feature interpretations
- Read `5.1_finbert_feature_evolution_blog_post.md` for complete analysis

## 📊 **Key Findings**

### **Top Features by Activation Delta**
1. **Feature 103**: +69,679 activations (Financial News Specialist)
2. **Feature 48**: +63,242 activations (Financial Data Processor)
3. **Feature 146**: +59,306 activations (Structural Marker Specialist)
4. **Feature 109**: +50,704 activations (Variable-Length Text Handler)
5. **Feature 127**: +49,211 activations (Causal Relationship Detector)

### **Feature Evolution Patterns**
- **75% of top FinBERT features** were not in BERT's top financial features
- **Emerging features**: Features that became newly active in FinBERT
- **Consistent features**: Features that maintained similar activity levels
- **Enhanced features**: Features that showed moderate improvements

### **Neural Specialization Categories**
1. **Market Analysis Features**: Market trends, stock analysis, investment content
2. **Data Processing Features**: Numerical, temporal, and structured data
3. **News Processing Features**: Headlines, articles, financial journalism
4. **Sentiment Analysis Features**: Opinions, predictions, causal relationships
5. **Comparative Language Features**: Comparative terms ("more", "higher")

## 🔧 **Technical Requirements**

### **Dependencies**
- PyTorch
- Transformers (HuggingFace)
- Delphi SAE auto-interpretation library
- Llama 3.1 8B (for feature interpretation)
- WandB (for training monitoring)

### **Models Used**
- **BERT**: bert-base-uncased
- **FinBERT**: ProsusAI/finbert
- **Dataset**: Yahoo Finance Stock Market News

### **SAE Configuration**
- **Architecture**: 768 → 200 → 768
- **Sparsity**: TopK with k=32
- **Training**: 1000 epochs with Adam optimizer
- **Layer**: 6th layer of both models

## 📈 **Research Impact**

This research demonstrates:
1. **Systematic neural feature evolution** during fine-tuning
2. **Intuitive feature specializations** that align with domain requirements
3. **Mathematical regularity** in activation deltas and domain relevance
4. **Interpretable transformations** that can be described in natural language

The findings advance our understanding of neural adaptation mechanisms and provide a foundation for more targeted fine-tuning strategies.

## 📝 **Citation**

If you use this research or code, please cite:
```
Neural Feature Evolution in Domain Adaptation: A Comprehensive Analysis of BERT to FinBERT Fine-tuning
```

---

*This complete research pipeline provides the first comprehensive analysis of neural feature evolution during domain adaptation, revealing both the intuitive and systematic nature of neural specialization.*
