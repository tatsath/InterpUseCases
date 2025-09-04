#!/usr/bin/env python3
"""
Comprehensive Feature Comparison Script
Compares feature explanations from Llama-2-7B vs FinLLama-7B for layer 16
Creates a combined Excel file with side-by-side comparison
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_excel_data(file_path):
    """Load Excel file and extract feature summary"""
    try:
        df = pd.read_excel(file_path, sheet_name='Feature_Summary')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_comparison_table(llama_df, finllama_df):
    """Create a comprehensive comparison table"""
    
    # Ensure both dataframes have the same structure
    if llama_df is None or finllama_df is None:
        print("❌ Cannot create comparison - one or both dataframes are None")
        return None
    
    # Create comparison dataframe
    comparison_data = []
    
    for i in range(len(llama_df)):
        llama_row = llama_df.iloc[i]
        finllama_row = finllama_df.iloc[i]
        
        # Extract feature labels (first 10 words max)
        llama_label = llama_row['Feature_Label']
        finllama_label = finllama_row['Feature_Label']
        
        # Clean labels to max 10 words
        llama_label_clean = " ".join(str(llama_label).split()[:10])
        finllama_label_clean = " ".join(str(finllama_label).split()[:10])
        
        # Add ellipsis if truncated
        if len(str(llama_label).split()) > 10:
            llama_label_clean += "..."
        if len(str(finllama_label).split()) > 10:
            finllama_label_clean += "..."
        
        comparison_data.append({
            'Feature_Index': llama_row['Feature_Index'],
            'Feature_Number': llama_row['Feature_Number'],
            'Llama_Base_Label': llama_label_clean,
            'FinLLama_Label': finllama_label_clean,
            'Llama_Explanations': llama_row['All_Explanations'],
            'FinLLama_Explanations': finllama_row['All_Explanations'],
            'Label_Similarity': calculate_label_similarity(llama_label_clean, finllama_label_clean),
            'Llama_Num_Explanations': llama_row['Num_Explanations'],
            'FinLLama_Num_Explanations': finllama_row['Num_Explanations']
        })
    
    return pd.DataFrame(comparison_data)

def calculate_label_similarity(label1, label2):
    """Calculate simple similarity between two labels"""
    if not label1 or not label2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(str(label1).lower().split())
    words2 = set(str(label2).lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return round(intersection / union, 3)

def create_summary_statistics(llama_df, finllama_df, comparison_df):
    """Create summary statistics for the comparison"""
    
    stats = {
        'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total_Features_Analyzed': len(comparison_df),
        'Layer': 16,
        'Models_Compared': ['Llama-2-7B Base', 'FinLLama-7B Fine-tuned'],
        
        # Label analysis
        'Average_Label_Similarity': round(comparison_df['Label_Similarity'].mean(), 3),
        'High_Similarity_Features': len(comparison_df[comparison_df['Label_Similarity'] >= 0.5]),
        'Medium_Similarity_Features': len(comparison_df[(comparison_df['Label_Similarity'] >= 0.2) & (comparison_df['Label_Similarity'] < 0.5)]),
        'Low_Similarity_Features': len(comparison_df[comparison_df['Label_Similarity'] < 0.2]),
        
        # Explanation analysis
        'Llama_Total_Explanations': llama_df['Num_Explanations'].sum(),
        'FinLLama_Total_Explanations': finllama_df['Num_Explanations'].sum(),
        'Llama_Avg_Explanations_Per_Feature': round(llama_df['Num_Explanations'].mean(), 2),
        'FinLLama_Avg_Explanations_Per_Feature': round(finllama_df['Num_Explanations'].mean(), 2)
    }
    
    return stats

def main():
    print("🚀 Starting Comprehensive Feature Comparison Analysis")
    print("=" * 80)
    
    # Load data from both Excel files
    print("📥 Loading Llama-2-7B feature explanations...")
    llama_file = "simple_llama_base_layer16_feature_explanations.xlsx"
    llama_df = load_excel_data(llama_file)
    
    print("📥 Loading FinLLama-7B feature explanations...")
    finllama_file = "simple_finllama_layer16_feature_explanations.xlsx"
    finllama_df = load_excel_data(finllama_file)
    
    if llama_df is None or finllama_df is None:
        print("❌ Failed to load one or both Excel files")
        return
    
    print("✅ Both datasets loaded successfully!")
    
    # Create comparison table
    print("\n🔍 Creating comparison table...")
    comparison_df = create_comparison_table(llama_df, finllama_df)
    
    # Create summary statistics
    print("📊 Calculating summary statistics...")
    stats = create_summary_statistics(llama_df, finllama_df, comparison_df)
    
    # Create comprehensive Excel file
    print("💾 Saving comprehensive comparison to Excel...")
    output_path = "comprehensive_feature_comparison_layer16.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Comparison table
        comparison_df.to_excel(writer, sheet_name='Feature_Comparison', index=False)
        
        # Individual model summaries
        llama_df.to_excel(writer, sheet_name='Llama_Base_Summary', index=False)
        finllama_df.to_excel(writer, sheet_name='FinLLama_Summary', index=False)
        
        # Summary statistics
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        stats_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Similarity analysis
        similarity_analysis = comparison_df[['Feature_Index', 'Feature_Number', 'Label_Similarity', 'Llama_Base_Label', 'FinLLama_Label']].copy()
        similarity_analysis = similarity_analysis.sort_values('Label_Similarity', ascending=False)
        similarity_analysis.to_excel(writer, sheet_name='Similarity_Analysis', index=False)
    
    print(f"✅ Comprehensive comparison saved to: {output_path}")
    
    # Print summary
    print("\n📋 COMPREHENSIVE FEATURE COMPARISON SUMMARY:")
    print("=" * 80)
    print(f"📊 Total Features Analyzed: {stats['Total_Features_Analyzed']}")
    print(f"🔗 Average Label Similarity: {stats['Average_Label_Similarity']}")
    print(f"✅ High Similarity Features (≥0.5): {stats['High_Similarity_Features']}")
    print(f"🟡 Medium Similarity Features (0.2-0.5): {stats['Medium_Similarity_Features']}")
    print(f"❌ Low Similarity Features (<0.2): {stats['Low_Similarity_Features']}")
    print(f"📝 Llama Base Total Explanations: {stats['Llama_Total_Explanations']}")
    print(f"📝 FinLLama Total Explanations: {stats['FinLLama_Total_Explanations']}")
    
    print(f"\n🔍 FEATURE-BY-FEATURE COMPARISON:")
    print("-" * 80)
    for i, row in comparison_df.iterrows():
        similarity_icon = "✅" if row['Label_Similarity'] >= 0.5 else "🟡" if row['Label_Similarity'] >= 0.2 else "❌"
        print(f"{similarity_icon} Feature {row['Feature_Number']:2d}: Similarity = {row['Label_Similarity']:.3f}")
        print(f"   Llama: {row['Llama_Base_Label']}")
        print(f"   FinLLama: {row['FinLLama_Label']}")
        print()
    
    print(f"\n🎉 Analysis complete! Check the comprehensive Excel file for detailed results.")
    print(f"📁 Files generated:")
    print(f"   - {output_path} (comprehensive comparison)")
    print(f"   - {llama_file} (Llama base model)")
    print(f"   - {finllama_file} (FinLLama fine-tuned model)")

if __name__ == "__main__":
    main()
