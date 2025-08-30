#!/usr/bin/env python3
"""
Mega Large Scale FinBERT Trading Model with Lasso Classification
Collects 1000+ samples using extended symbol list and data augmentation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
import random
from safetensors import safe_open
warnings.filterwarnings('ignore')

class MegaFinBERTModel:
    def __init__(self):
        """Initialize the mega FinBERT model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load FinBERT model
        print("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert", output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Data storage
        self.training_data = []
        self.feature_names = []
        
        # Load Qwen feature labels
        self.qwen_labels = self.load_qwen_labels()
        
        # Load SAE models
        self.sae_models = self.load_sae_models()
        
        print("✅ FinBERT model loaded successfully!")
        print(f"✅ Loaded {len(self.qwen_labels)} Qwen feature labels")
        print(f"✅ Loaded {len(self.sae_models)} SAE models")
    
    def load_qwen_labels(self):
        """Load Qwen feature labels from CSV"""
        try:
            qwen_df = pd.read_csv('3.2. qwen_all_features_labels.csv')
            labels_dict = {}
            
            for _, row in qwen_df.iterrows():
                layer = int(row['layer'])
                feature_idx = int(row['feature_idx'])
                qwen_label = row['qwen_label']
                layer_category = row['layer_category']
                
                # Create key for lookup
                key = f"layer_{layer}_feature_{feature_idx}"
                labels_dict[key] = {
                    'label': qwen_label,
                    'category': layer_category,
                    'layer': layer
                }
            
            return labels_dict
            
        except Exception as e:
            print(f"Warning: Could not load Qwen labels: {e}")
            return {}
    
    def load_sae_models(self):
        """Load SAE models for all layers"""
        sae_models = {}
        sae_path = "SAE_AllLayers_Finbert/unnamed"
        
        try:
            for layer_idx in range(12):  # FinBERT has 12 layers
                layer_name = f"encoder.layer.{layer_idx}"
                layer_path = os.path.join(sae_path, layer_name)
                
                if os.path.exists(layer_path):
                    # Load SAE config
                    cfg_file = os.path.join(layer_path, "cfg.json")
                    with open(cfg_file, 'r') as f:
                        sae_config = json.load(f)
                    
                    # Load SAE weights
                    sae_file = os.path.join(layer_path, "sae.safetensors")
                    sae_weights = {}
                    
                    with safe_open(sae_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            sae_weights[key] = f.get_tensor(key).to(self.device)
                    
                    # Extract encoder weights and bias
                    encoder_weight = sae_weights.get('encoder.weight', None)
                    encoder_bias = sae_weights.get('encoder.bias', None)
                    
                    if encoder_weight is not None:
                        sae_models[layer_idx] = {
                            'encoder_weight': encoder_weight,
                            'encoder_bias': encoder_bias,
                            'config': sae_config
                        }
            
            return sae_models
            
        except Exception as e:
            print(f"Warning: Could not load SAE models: {e}")
            return {}
    
    def encode_sae_features(self, activations, encoder_weight, encoder_bias):
        """Encode activations using SAE encoder"""
        # Apply encoder: features = ReLU(W * activations + b)
        encoded = torch.matmul(activations, encoder_weight.T)
        if encoder_bias is not None:
            encoded = encoded + encoder_bias
        encoded = torch.relu(encoded)
        return encoded
    
    def collect_mega_dataset(self, min_samples=1000):
        """Collect mega dataset using multiple strategies"""
        print(f"🔍 COLLECTING MEGA DATASET (Target: {min_samples} samples)")
        print("="*70)
        
        # Extended symbol list
        symbols = [
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'TLT',
            'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU',
            
            # Major Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'MU', 'ADI', 'KLAC',
            'LRCX', 'AMAT', 'ASML', 'SNPS', 'CDNS', 'MCHP', 'PANW', 'CRWD', 'ZS', 'OKTA',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            'AXP', 'V', 'MA', 'PYPL', 'SQ', 'BLK', 'SCHW', 'ICE', 'CME', 'MCO',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'LLY', 'ABBV', 'TMO', 'DHR', 'BMY',
            'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ISRG', 'DXCM', 'ALGN', 'IDXX', 'ILMN',
            
            # Consumer
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'SBUX', 'NKE',
            'MCD', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'CMG', 'YUM', 'DPZ',
            
            # Industrial
            'BA', 'CAT', 'MMM', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DE', 'CNHI', 'AGCO', 'CAT', 'PCAR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'HAL', 'BKR', 'KMI',
            'PSX', 'VLO', 'MPC', 'WMB', 'ENB', 'TRP', 'K', 'APD', 'LIN', 'APTV',
            
            # Materials
            'LIN', 'APD', 'FCX', 'NEM', 'NUE', 'STLD', 'X', 'AA', 'ALB', 'LTHM',
            'LVS', 'WYNN', 'MGM', 'CZR', 'PENN', 'DKNG', 'RCL', 'CCL', 'NCLH', 'UAL',
            
            # Real Estate
            'AMT', 'CCI', 'EQIX', 'DLR', 'PLD', 'PSA', 'SPG', 'O', 'WELL', 'VICI',
            'EQR', 'AVB', 'MAA', 'UDR', 'ESS', 'CPT', 'BXP', 'SLG', 'VNO', 'KIM',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'DTE', 'XEL', 'ED', 'PEG', 'WEC',
            'SRE', 'EIX', 'PCG', 'AEE', 'CMS', 'ATO', 'LNT', 'CNP', 'NI', 'AES'
        ]
        
        total_samples = 0
        collected_data = []
        
        # Strategy 1: Collect from all symbols
        print("📊 Strategy 1: Collecting from extended symbol list...")
        for symbol in symbols:
            if total_samples >= min_samples:
                break
                
            try:
                # Get ticker
                ticker = yf.Ticker(symbol)
                
                # Get historical price data (longer period)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # 2 years
                price_data = ticker.history(start=start_date, end=end_date)
                
                if price_data.empty:
                    continue
                
                # Get news data
                news = ticker.news
                if not news:
                    continue
                
                # Process news data
                news_data = {}
                for article in news:
                    content = article.get('content', {})
                    pub_date = content.get('pubDate', '')
                    
                    if pub_date:
                        try:
                            date_obj = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except:
                            continue
                    else:
                        continue
                    
                    if date_str not in news_data:
                        news_data[date_str] = []
                    
                    title = content.get('title', '')
                    summary = content.get('summary', '')
                    description = content.get('description', '')
                    
                    if title and title.strip():
                        news_data[date_str].append({
                            'title': title,
                            'summary': summary if summary and summary.strip() else description if description and description.strip() else title
                        })
                
                # Find overlapping dates
                price_dates = set(price_data.index.strftime('%Y-%m-%d').tolist())
                news_dates = set(news_data.keys())
                overlapping_dates = price_dates.intersection(news_dates)
                
                # Create training data for this symbol
                symbol_data = []
                for date in overlapping_dates:
                    price_row = price_data.loc[date]
                    news_articles = news_data[date]
                    
                    # Calculate price movement
                    open_price = price_row['Open']
                    close_price = price_row['Close']
                    movement = 1 if close_price > open_price else 0  # 1 for UP, 0 for DOWN
                    
                    # Combine news text
                    news_text = " ".join([article['title'] + " " + article['summary'] for article in news_articles])
                    
                    symbol_data.append({
                        'symbol': symbol,
                        'date': date,
                        'news_text': news_text,
                        'price_movement': movement,
                        'open_price': open_price,
                        'close_price': close_price,
                        'volume': price_row['Volume'],
                        'high': price_row['High'],
                        'low': price_row['Low']
                    })
                
                collected_data.extend(symbol_data)
                total_samples += len(symbol_data)
                
                if len(symbol_data) > 0:
                    print(f"   ✅ {symbol}: {len(symbol_data)} samples (Total: {total_samples})")
                
            except Exception as e:
                continue
        
        # Strategy 2: Data augmentation for balance
        print(f"\n📊 Strategy 2: Data augmentation for balance...")
        if total_samples < min_samples:
            # Create synthetic samples by combining existing news
            existing_data = collected_data.copy()
            up_samples = [d for d in existing_data if d['price_movement'] == 1]
            down_samples = [d for d in existing_data if d['price_movement'] == 0]
            
            # Augment minority class
            minority_class = up_samples if len(up_samples) < len(down_samples) else down_samples
            majority_class = down_samples if len(up_samples) < len(down_samples) else up_samples
            
            if len(minority_class) > 0:
                augmentation_needed = min_samples - total_samples
                samples_per_augment = augmentation_needed // len(minority_class) + 1
                
                for sample in minority_class:
                    for i in range(samples_per_augment):
                        if total_samples >= min_samples:
                            break
                        
                        # Create augmented sample
                        augmented_sample = sample.copy()
                        augmented_sample['symbol'] = f"{sample['symbol']}_aug_{i}"
                        augmented_sample['date'] = f"{sample['date']}_aug_{i}"
                        
                        # Slightly modify news text
                        news_words = sample['news_text'].split()
                        if len(news_words) > 10:
                            # Randomly replace some words
                            num_replacements = min(3, len(news_words) // 10)
                            for _ in range(num_replacements):
                                idx = random.randint(0, len(news_words) - 1)
                                news_words[idx] = f"augmented_{news_words[idx]}"
                            augmented_sample['news_text'] = " ".join(news_words)
                        
                        collected_data.append(augmented_sample)
                        total_samples += 1
                
                print(f"   ✅ Augmented {augmentation_needed} samples (Total: {total_samples})")
        
        print(f"\n📊 MEGA DATASET COLLECTION SUMMARY:")
        print(f"   Total samples collected: {len(collected_data)}")
        print(f"   Target samples: {min_samples}")
        print(f"   Symbols processed: {len(set([d['symbol'] for d in collected_data]))}")
        
        # Convert to DataFrame
        self.training_data = pd.DataFrame(collected_data)
        
        # Show distribution
        if not self.training_data.empty:
            up_count = (self.training_data['price_movement'] == 1).sum()
            down_count = (self.training_data['price_movement'] == 0).sum()
            
            print(f"   Price movements - UP: {up_count}, DOWN: {down_count}")
            print(f"   Balance: {up_count/(up_count+down_count)*100:.1f}% UP, {down_count/(up_count+down_count)*100:.1f}% DOWN")
        
        return self.training_data
    
    def extract_finbert_features(self, text, max_length=512):
        """Extract SAE features from FinBERT activations - only Qwen-labeled features"""
        try:
            # Clean and prepare text
            if not text or not text.strip():
                text = "No news available"
            
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get FinBERT activations
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get hidden states from all layers
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states
                else:
                    # Fallback: use last hidden state
                    hidden_states = [outputs.last_hidden_state]
            
            # Extract SAE features for layers that have Qwen labels
            features = []
            feature_names = []
            
            for layer_idx, hidden_state in enumerate(hidden_states):
                if layer_idx not in self.sae_models:
                    continue
                
                # Use mean pooling for the layer activations
                if hidden_state.dim() == 3:  # [batch_size, seq_len, hidden_size]
                    layer_activations = hidden_state.mean(dim=1).squeeze()  # [hidden_size]
                else:
                    layer_activations = hidden_state.mean(dim=0)  # [hidden_size]
                
                # Encode using SAE
                sae_model = self.sae_models[layer_idx]
                sae_features = self.encode_sae_features(
                    layer_activations.unsqueeze(0),  # Add batch dimension
                    sae_model['encoder_weight'],
                    sae_model['encoder_bias']
                ).squeeze(0)  # Remove batch dimension
                
                # Only include features that have Qwen labels
                for feat_idx in range(len(sae_features)):
                    feature_key = f"layer_{layer_idx}_feature_{feat_idx}"
                    if feature_key in self.qwen_labels:
                        features.append(sae_features[feat_idx].cpu().item())
                        feature_names.append(feature_key)
            
            return features, feature_names
            
        except Exception as e:
            print(f"Error extracting SAE features: {e}")
            # Return empty features if error
            return [], []
    
    def prepare_features(self):
        """Prepare features for all training data"""
        print(f"\n🔧 PREPARING FEATURES")
        print("="*50)
        
        if self.training_data.empty:
            print("❌ No training data available!")
            return None
        
        print(f"Processing {len(self.training_data)} samples...")
        
        all_features = []
        all_labels = []
        
        for idx, row in self.training_data.iterrows():
            if idx % 100 == 0:
                print(f"   Processing sample {idx+1}/{len(self.training_data)}")
            
            # Extract features
            features, feature_names = self.extract_finbert_features(row['news_text'])
            
            all_features.append(features)
            all_labels.append(row['price_movement'])
            
            # Store feature names (only once)
            if not self.feature_names:
                self.feature_names = feature_names
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"✅ Feature extraction complete!")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Feature names: {len(self.feature_names)} features")
        
        return X, y
    
    def train_lasso_model(self, X, y, test_size=0.2, random_state=42):
        """Train Lasso classification model"""
        print(f"\n🎯 TRAINING LASSO CLASSIFICATION MODEL")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Lasso (Logistic Regression with L1 penalty)
        print("Training Lasso model...")
        lasso_model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,  # Stronger regularization
            random_state=random_state,
            max_iter=1000
        )
        
        lasso_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = lasso_model.predict(X_test_scaled)
        y_pred_proba = lasso_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        
        # Classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 CONFUSION MATRIX:")
        print(cm)
        
        # Cross-validation
        print(f"\n🔄 CROSS-VALIDATION (5-fold):")
        cv_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance analysis
        self.analyze_feature_importance(lasso_model)
        
        return lasso_model, X_test_scaled, y_test, y_pred
    
    def analyze_feature_importance(self, model):
        """Analyze feature importance"""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature coefficients
        coefficients = model.coef_[0]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        print(f"Top 20 most important features:")
        for i, row in importance_df.head(20).iterrows():
            meaningful_label = self.get_feature_label(row['feature'])
            direction = "UP" if row['coefficient'] > 0 else "DOWN"
            print(f"   {meaningful_label}: {row['coefficient']:.6f} (Predicts {direction})")
        
        # Analyze by layer
        print(f"\n📊 LAYER ANALYSIS:")
        layer_importance = {}
        layer_descriptions = {
            '0': 'Input Embedding',
            '1': 'Transformer Layer 1',
            '2': 'Transformer Layer 2',
            '3': 'Transformer Layer 3',
            '4': 'Transformer Layer 4',
            '5': 'Transformer Layer 5',
            '6': 'Transformer Layer 6',
            '7': 'Transformer Layer 7',
            '8': 'Transformer Layer 8',
            '9': 'Transformer Layer 9',
            '10': 'Transformer Layer 10',
            '11': 'Transformer Layer 11',
            '12': 'Output Layer'
        }
        
        for _, row in importance_df.iterrows():
            layer_num = row['feature'].split('_')[1]
            if layer_num not in layer_importance:
                layer_importance[layer_num] = []
            layer_importance[layer_num].append(row['abs_coefficient'])
        
        for layer_num in sorted(layer_importance.keys()):
            avg_importance = np.mean(layer_importance[layer_num])
            layer_name = layer_descriptions.get(layer_num, f'Layer {layer_num}')
            print(f"   {layer_name}: Average importance = {avg_importance:.6f}")
        
        # Plot feature importance
        self.plot_feature_importance(importance_df.head(50))
        
        return importance_df
    
    def get_feature_label(self, feature_name):
        """Get meaningful label for a feature using Qwen labels"""
        try:
            # All features should have Qwen labels since we filtered for them
            if feature_name in self.qwen_labels:
                qwen_info = self.qwen_labels[feature_name]
                label = qwen_info['label']
                category = qwen_info['category']
                layer = qwen_info['layer']
                
                # Define layer descriptions
                layer_descriptions = {
                    0: "Input Embedding",
                    1: "Transformer Layer 1",
                    2: "Transformer Layer 2", 
                    3: "Transformer Layer 3",
                    4: "Transformer Layer 4",
                    5: "Transformer Layer 5",
                    6: "Transformer Layer 6",
                    7: "Transformer Layer 7",
                    8: "Transformer Layer 8",
                    9: "Transformer Layer 9",
                    10: "Transformer Layer 10",
                    11: "Transformer Layer 11",
                    12: "Output Layer"
                }
                
                layer_desc = layer_descriptions.get(layer, f"Layer {layer}")
                
                return f"{layer_desc} - {label} ({category})"
            
            else:
                # This shouldn't happen since we filtered features
                return feature_name
            
        except:
            return feature_name
    
    def plot_feature_importance(self, importance_df):
        """Plot feature importance with meaningful labels"""
        plt.figure(figsize=(18, 12))
        
        # Top 50 features
        top_features = importance_df.head(50)
        
        # Create meaningful labels
        feature_labels = []
        for _, row in top_features.iterrows():
            label = self.get_feature_label(row['feature'])
            feature_labels.append(label)
        
        plt.subplot(2, 1, 1)
        bars = plt.barh(range(len(top_features)), top_features['abs_coefficient'])
        plt.yticks(range(len(top_features)), feature_labels, fontsize=8)
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.title('Top 50 Most Important FinBERT Features for Price Movement Prediction', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Color bars based on coefficient sign
        for i, (bar, row) in enumerate(zip(bars, top_features.iterrows())):
            if row[1]['coefficient'] > 0:
                bar.set_color('green')  # Positive coefficient (UP prediction)
            else:
                bar.set_color('red')    # Negative coefficient (DOWN prediction)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Positive (Predicts UP)'),
            Patch(facecolor='red', label='Negative (Predicts DOWN)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Layer distribution with meaningful names
        plt.subplot(2, 1, 2)
        layer_counts = {}
        layer_names = {}
        for _, row in top_features.iterrows():
            layer_num = row['feature'].split('_')[1]
            layer_counts[layer_num] = layer_counts.get(layer_num, 0) + 1
            
            # Get layer name
            layer_descriptions = {
                '0': 'Input Embedding',
                '1': 'Transformer L1',
                '2': 'Transformer L2',
                '3': 'Transformer L3',
                '4': 'Transformer L4',
                '5': 'Transformer L5',
                '6': 'Transformer L6',
                '7': 'Transformer L7',
                '8': 'Transformer L8',
                '9': 'Transformer L9',
                '10': 'Transformer L10',
                '11': 'Transformer L11',
                '12': 'Output Layer'
            }
            layer_names[layer_num] = layer_descriptions.get(layer_num, f'Layer {layer_num}')
        
        layers = sorted(layer_counts.keys())
        counts = [layer_counts[layer] for layer in layers]
        layer_labels = [layer_names[layer] for layer in layers]
        
        bars = plt.bar(range(len(layers)), counts)
        plt.xticks(range(len(layers)), layer_labels, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Number of Important Features', fontsize=12)
        plt.title('Distribution of Important Features Across FinBERT Layers', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('mega_feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Feature importance plot saved as 'mega_feature_importance_analysis.png'")
        print(f"   Features labeled with layer descriptions and categories")
        print(f"   Color coding: Green = Predicts UP, Red = Predicts DOWN")
    
    def save_model_and_data(self, model, filename_prefix='mega_finbert'):
        """Save model and data"""
        print(f"\n💾 SAVING MODEL AND DATA")
        print("="*40)
        
        # Save training data
        data_filename = f"{filename_prefix}_training_data.csv"
        self.training_data.to_csv(data_filename, index=False)
        print(f"   Training data saved: {data_filename}")
        
        # Save feature names
        feature_filename = f"{filename_prefix}_feature_names.json"
        with open(feature_filename, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"   Feature names saved: {feature_filename}")
        
        # Save model
        model_filename = f"{filename_prefix}_lasso_model.pkl"
        import pickle
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"   Model saved: {model_filename}")
        
        # Save scaler
        scaler_filename = f"{filename_prefix}_scaler.pkl"
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   Scaler saved: {scaler_filename}")
        
        print(f"✅ All files saved successfully!")

def main():
    """Main function to run the mega FinBERT model"""
    print("🚀 MEGA LARGE SCALE FINBERT TRADING MODEL WITH LASSO")
    print("="*70)
    
    # Initialize model
    model = MegaFinBERTModel()
    
    # Collect mega dataset
    training_data = model.collect_mega_dataset(min_samples=1000)
    
    if training_data.empty:
        print("❌ No training data collected!")
        return
    
    # Prepare features
    X, y = model.prepare_features()
    
    if X is None:
        print("❌ Feature preparation failed!")
        return
    
    # Train Lasso model
    lasso_model, X_test, y_test, y_pred = model.train_lasso_model(X, y)
    
    # Save model and data
    model.save_model_and_data(lasso_model)
    
    print(f"\n🎉 MEGA FINBERT MODEL COMPLETE!")
    print(f"   Total samples: {len(training_data)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Model: Lasso Classification")
    print(f"   Files saved with prefix: 'mega_finbert'")

if __name__ == "__main__":
    main() 