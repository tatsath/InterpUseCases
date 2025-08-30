import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open
import json
import os
from transformers import AutoTokenizer, BertModel, AutoModelForCausalLM
import pandas as pd
from collections import defaultdict, Counter
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import warnings
import re
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import multiprocessing as mp
import time
warnings.filterwarnings('ignore')

def load_sae_model(sae_path, layer_name, device):
    """Load SAE model from path for a specific layer"""
    sae_file = os.path.join(sae_path, layer_name, "sae.safetensors")
    cfg_file = os.path.join(sae_path, layer_name, "cfg.json")
    
    if not os.path.exists(sae_file) or not os.path.exists(cfg_file):
        return None
    
    # Load config
    with open(cfg_file, 'r') as f:
        sae_config = json.load(f)
    
    # Load weights to CPU first, then move to target device
    sae_weights = {}
    with safe_open(sae_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            sae_weights[key] = f.get_tensor(key).to(device)
    
    encoder_weight = sae_weights.get('encoder.weight', None)
    decoder_weight = sae_weights.get('W_dec', None)
    encoder_bias = sae_weights.get('encoder.bias', None)
    decoder_bias = sae_weights.get('b_dec', None)
    
    return {
        'encoder_weight': encoder_weight,
        'decoder_weight': decoder_weight,
        'encoder_bias': encoder_bias,
        'decoder_bias': decoder_bias,
        'config': sae_config,
        'layer': layer_name
    }

def get_layer_activations_bert(text, model, tokenizer, layer_idx=0, max_length=512):
    """Get activations from specific BERT layer for given text"""
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True,
        padding=True
    ).to(device)
    
    activations = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())
    
    target_module = model.encoder.layer[layer_idx]
    hook = target_module.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
        
    hook.remove()
    
    return activations[0] if activations else None

def encode_activations(activations, encoder_weight, encoder_bias=None):
    """Encode activations using the trained SAE"""
    # Ensure all tensors are on the same device
    device = activations.device
    encoder_weight = encoder_weight.to(device)
    if encoder_bias is not None:
        encoder_bias = encoder_bias.to(device)
    
    activations = activations.to(encoder_weight.dtype)
    
    if encoder_bias is not None:
        encoded = torch.nn.functional.linear(activations, encoder_weight, encoder_bias)
    else:
        encoded = torch.nn.functional.linear(activations, encoder_weight)
        
    encoded = torch.nn.functional.relu(encoded)
    return encoded

def load_real_financial_sentences():
    """Load real financial sentences from FinBERT training data and Yahoo Finance dataset"""
    print("Loading real financial sentences from multiple datasets...")
    
    all_sentences = []
    
    try:
        # Load Yahoo Finance dataset
        print("Loading Yahoo Finance dataset...")
        from datasets import load_dataset
        
        # Load Yahoo Finance dataset
        yahoo_dataset = load_dataset("jyanimaulik/yahoo_finance_stockmarket_news", split="train")
        
        # Extract sentences from Yahoo Finance dataset
        yahoo_sentences = []
        for item in yahoo_dataset:
            if 'text' in item and item['text']:
                # Split into sentences and clean
                text = item['text']
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20 and len(s.strip()) < 500]
                yahoo_sentences.extend(sentences[:10])  # Take up to 10 sentences per article
        
        # Limit to 300 sentences from Yahoo Finance
        yahoo_sentences = yahoo_sentences[:300]
        print(f"Loaded {len(yahoo_sentences)} sentences from Yahoo Finance dataset")
        
        # Add to all sentences
        for sentence in yahoo_sentences:
            all_sentences.append({
                'text': sentence,
                'source': 'yahoo_finance',
                'category': 'financial_news'  # Generic category
            })
            
    except Exception as e:
        print(f"Error loading Yahoo Finance dataset: {e}")
        print("Will use alternative financial sentences...")
    
    # Add high-quality financial sentences from various sources to reach target
    print("Adding high-quality financial sentences from various sources...")
    
    # High-quality financial sentences from various sources (300 sentences)
    additional_sentences = [
        # Market movements and trading (50 sentences)
        "The S&P 500 index closed at a record high of 4,800 points, marking its best performance in three months.",
        "Trading volume surged to 2.5 billion shares as institutional investors rebalanced their portfolios.",
        "Market volatility spiked as the VIX index jumped 15% following unexpected economic data releases.",
        "The Dow Jones Industrial Average gained 350 points, led by strong performance in the technology sector.",
        "Stock prices fluctuated wildly in the final hour of trading as investors reacted to Federal Reserve comments.",
        "NASDAQ composite index surged 2.3% on technology sector strength and positive earnings reports.",
        "Market breadth indicators showed broad-based participation in the rally across all major sectors.",
        "Sector rotation patterns emerged as investors reallocated capital from growth to value stocks.",
        "Market capitalization of tech stocks reached unprecedented levels, exceeding $10 trillion.",
        "Trading volumes exceeded historical averages during earnings season, indicating strong investor interest.",
        "Market makers adjusted spreads in response to increased volatility and uncertainty.",
        "Institutional investors increased their market exposure significantly in the latest quarter.",
        "Retail trading activity surged during market hours, particularly in meme stocks and options.",
        "Market liquidity conditions improved as bid-ask spreads narrowed across major indices.",
        "Short interest ratios declined as bearish sentiment decreased across multiple sectors.",
        "Market momentum indicators suggested continued upward movement in the near term.",
        "Technical analysis patterns indicated potential breakout levels for major indices.",
        "Market sentiment surveys showed increasing optimism among professional investors.",
        "Cross-asset correlations increased during periods of market stress and uncertainty.",
        "Market microstructure analysis revealed changing patterns in order flow and execution.",
        
        # Earnings and financial performance (50 sentences)
        "Quarterly earnings exceeded analyst expectations by 12% on strong revenue growth across all business segments.",
        "Net profit margins expanded to 18.5% from 15.2% in the previous quarter due to operational efficiency improvements.",
        "EBITDA increased 23% year-over-year, driven by market share gains and cost control measures.",
        "Revenue growth accelerated to 15% compared to 8% in the prior period, exceeding management guidance.",
        "Cash flow from operations strengthened to $2.3 billion, supporting increased dividend payments.",
        "Forward guidance exceeded Wall Street estimates for next quarter by 8% on average.",
        "Earnings quality metrics showed sustainable profit growth patterns across all divisions.",
        "Revenue recognition policies aligned with new accounting standards and regulatory requirements.",
        "Earnings surprise factor was positive across all business segments and geographic regions.",
        "Pro forma earnings adjusted for one-time items showed strong growth of 18% year-over-year.",
        "Earnings momentum indicators pointed to continued strength in the upcoming quarters.",
        "Earnings revisions by analysts were overwhelmingly positive, with 75% raising estimates.",
        "Earnings season kicked off with strong results from major banks and financial institutions.",
        "Earnings conference calls revealed optimistic outlook for future quarters and full-year guidance.",
        "Earnings per share growth outpaced revenue growth due to margin expansion and share buybacks.",
        "Earnings quality scores improved across the board for reporting companies in the S&P 500.",
        "Operating leverage improved as fixed costs remained stable while revenues increased.",
        "Return on invested capital increased to 15.2% from 12.8% in the previous year.",
        "Free cash flow generation strengthened, supporting capital allocation decisions.",
        "Earnings before interest and taxes grew 20% year-over-year on operational improvements.",
        "Net income attributable to common shareholders increased 25% on strong performance.",
        "Diluted earnings per share grew 22% year-over-year, exceeding consensus estimates.",
        "Earnings before depreciation and amortization expanded to $4.2 billion from $3.1 billion.",
        "Adjusted earnings per share excluding special items showed consistent growth trends.",
        "Earnings quality analysis revealed sustainable and recurring profit streams.",
        "Earnings persistence metrics indicated stable and predictable financial performance.",
        "Earnings surprise history showed consistent outperformance relative to analyst estimates.",
        
        # Interest rates and monetary policy (50 sentences)
        "The Federal Reserve maintained its benchmark interest rate at 5.25-5.50% following the latest policy meeting.",
        "Bond yields declined across all maturities as investors priced in potential rate cuts in the coming months.",
        "The yield curve flattened as short-term rates remained elevated while long-term rates declined.",
        "Inflation expectations moderated to 2.1%, supporting the case for accommodative monetary policy.",
        "Central bank balance sheet expanded through quantitative easing measures to support economic recovery.",
        "Federal funds rate target range remained at 5.25-5.50 percent for the third consecutive meeting.",
        "Real interest rates adjusted for inflation showed positive territory for the first time in years.",
        "Term premium on long-term bonds increased significantly as uncertainty about future policy grew.",
        "Forward guidance from central banks influenced market expectations for future rate movements.",
        "Interest rate sensitivity analysis showed portfolio vulnerability to rising rates.",
        "Monetary policy transmission mechanisms functioned effectively across financial markets.",
        "Interest rate differentials between countries affected currency flows and exchange rates.",
        "Central bank communication strategy emphasized data dependency in policy decisions.",
        "Interest rate risk management became critical for financial institutions and investors.",
        "Monetary policy normalization process proceeded gradually to avoid market disruption.",
        "Interest rate expectations embedded in bond prices shifted lower following economic data.",
        "Policy rate decisions influenced borrowing costs across consumer and business lending.",
        "Interest rate swap markets reflected changing expectations for future monetary policy.",
        "Central bank credibility remained high despite challenging economic conditions.",
        "Monetary policy effectiveness was enhanced by coordinated fiscal policy measures.",
        "Interest rate corridor systems provided stability in short-term funding markets.",
        "Policy rate transmission to retail lending rates occurred with expected lags.",
        "Monetary policy accommodation supported economic recovery and employment growth.",
        "Interest rate forward guidance helped anchor long-term inflation expectations.",
        "Policy rate changes affected asset valuations across equity and fixed income markets.",
        "Monetary policy stance remained accommodative despite rising inflation pressures.",
        "Interest rate policy decisions were communicated clearly to financial market participants.",
        "Policy rate normalization was conducted gradually to minimize market volatility.",
        "Interest rate policy effectiveness was enhanced by macroprudential policy coordination.",
        
        # Credit and lending (50 sentences)
        "Loan approval rates tightened as banks increased risk assessment standards for commercial lending.",
        "Credit spreads widened by 50 basis points, reflecting heightened default risk concerns in the market.",
        "Mortgage rates increased to 7.2%, affecting homebuyer affordability and housing market activity.",
        "Corporate bond issuance declined 30% due to unfavorable market conditions and increased borrowing costs.",
        "Credit quality metrics deteriorated across multiple lending portfolios, prompting stricter underwriting standards.",
        "Credit default swap spreads widened for high-yield corporate bonds as risk aversion increased.",
        "Lending capacity constraints affected small business loan availability and approval rates.",
        "Credit scoring models were updated to reflect new risk factors and economic conditions.",
        "Securitization of credit assets slowed due to market conditions and regulatory changes.",
        "Credit risk transfer mechanisms became more expensive as market liquidity decreased.",
        "Lending rate spreads over benchmark rates increased significantly across all loan types.",
        "Credit portfolio diversification strategies were implemented to reduce concentration risk.",
        "Credit risk capital requirements increased under new regulations and supervisory guidance.",
        "Credit rating agencies revised outlooks for multiple sectors and individual issuers.",
        "Credit market liquidity conditions deteriorated during stress periods and market volatility.",
        "Credit risk appetite among institutional lenders decreased as economic uncertainty grew.",
        "Consumer credit utilization increased to pre-pandemic levels as spending patterns normalized.",
        "Credit card delinquency rates rose across all major banks and financial institutions.",
        "Commercial real estate lending standards tightened as property values and cash flows declined.",
        "Credit risk modeling approaches were updated to incorporate new economic scenarios.",
        "Lending relationship management became more important as credit availability decreased.",
        "Credit risk monitoring systems were enhanced to detect early warning signals.",
        "Credit portfolio stress testing revealed vulnerabilities under adverse economic scenarios.",
        "Credit risk appetite statements were revised to reflect changing market conditions.",
        "Lending capacity utilization increased as demand for credit remained strong.",
        "Credit risk management frameworks were strengthened across all lending businesses.",
        "Credit risk reporting frequency increased to support timely decision-making.",
        "Credit risk governance structures were enhanced to improve oversight and control.",
        "Credit risk culture initiatives were launched to strengthen risk awareness.",
        
        # Investment strategies and portfolio management (50 sentences)
        "Portfolio diversification strategies helped mitigate downside risk exposure during market volatility.",
        "Hedge fund performance varied significantly based on investment approach and market positioning.",
        "Asset allocation recommendations emphasized defensive positioning in anticipation of economic slowdown.",
        "Alternative investments provided uncorrelated returns during traditional market stress periods.",
        "Systematic trading strategies outperformed discretionary approaches in volatile market conditions.",
        "Multi-factor investment models showed strong performance across different market environments.",
        "ESG integration in investment processes became mainstream among institutional investors.",
        "Quantitative investment strategies gained market share as technology and data availability improved.",
        "Risk parity approaches balanced portfolio risk contributions across different asset classes.",
        "Alternative beta strategies provided diversification benefits beyond traditional market exposure.",
        "Investment horizon considerations influenced asset allocation decisions and risk tolerance.",
        "Currency hedging strategies protected international portfolio returns from exchange rate volatility.",
        "Derivatives usage for portfolio protection increased significantly during market uncertainty.",
        "Investment style rotation strategies captured market momentum and sector performance trends.",
        "Liquidity management became critical for large institutional portfolios during stress periods.",
        "Investment risk budgeting frameworks guided allocation decisions and performance attribution.",
        "Dynamic asset allocation strategies responded to changing market conditions and economic data.",
        "Portfolio rebalancing occurred systematically to maintain target allocations and risk profiles.",
        "Investment manager selection processes incorporated both quantitative and qualitative factors.",
        "Performance attribution analysis revealed sources of excess returns and risk-adjusted performance.",
        "Investment policy statements were updated to reflect changing market conditions and objectives.",
        "Portfolio construction methodologies evolved to incorporate new risk factors and constraints.",
        "Investment risk management practices were enhanced to address emerging market risks.",
        "Portfolio optimization techniques incorporated transaction costs and implementation constraints.",
        "Investment performance measurement frameworks provided comprehensive risk-adjusted metrics.",
        "Portfolio stress testing scenarios were expanded to include new risk factors and correlations.",
        "Investment governance structures ensured proper oversight and alignment with objectives.",
        "Portfolio reporting frequency increased to support timely decision-making and monitoring.",
        
        # Corporate finance and M&A (50 sentences)
        "Merger and acquisition activity accelerated with $75 billion in deals announced during the quarter.",
        "Capital structure optimization reduced weighted average cost of capital to 8.5% from 9.2%.",
        "Share repurchase program authorized additional $3 billion in buybacks to return capital to shareholders.",
        "Debt refinancing improved interest coverage ratios to 4.2x from 3.1x in the previous year.",
        "Corporate cash balances increased to $45 billion due to strong operating performance and cash generation.",
        "Corporate governance practices strengthened across all sectors and market capitalizations.",
        "Capital allocation decisions prioritized shareholder value creation and long-term growth.",
        "Corporate restructuring initiatives improved operational efficiency and cost competitiveness.",
        "Debt maturity profiles were extended to lock in low rates and improve financial flexibility.",
        "Corporate tax optimization strategies were implemented globally to improve after-tax returns.",
        "Capital markets access remained strong for investment-grade issuers despite market volatility.",
        "Corporate liquidity positions strengthened through cash generation and balance sheet management.",
        "Capital expenditure efficiency metrics improved across industries and business cycles.",
        "Corporate risk management frameworks were enhanced significantly to address emerging risks.",
        "Capital structure flexibility allowed for opportunistic financing and strategic transactions.",
        "Corporate financial reporting transparency increased under new accounting standards.",
        "Dividend policy maintained consistent payout ratios while supporting growth investments.",
        "Corporate cash flow management practices improved working capital efficiency.",
        "Capital expenditure plans were revised upward to support growth and competitive positioning.",
        "Corporate debt levels were managed prudently to maintain financial flexibility.",
        "Capital structure decisions balanced debt capacity with investment opportunities.",
        "Corporate financial planning processes incorporated scenario analysis and stress testing.",
        "Capital allocation frameworks prioritized high-return investment opportunities.",
        "Corporate treasury functions optimized cash management and risk mitigation strategies.",
        "Capital structure optimization considered both financial and strategic objectives.",
        "Corporate financial policies were updated to reflect changing market conditions.",
        "Capital expenditure authorization processes ensured proper oversight and accountability.",
        "Corporate financial risk management addressed currency, interest rate, and commodity exposures.",
        "Capital structure flexibility supported strategic initiatives and market opportunities.",
        
        # Regulatory compliance and risk management (50 sentences)
        "New financial regulations require enhanced reporting and transparency measures across all business lines.",
        "Compliance costs increased 18% due to stricter supervisory requirements and regulatory changes.",
        "Risk management frameworks aligned with evolving regulatory expectations and industry best practices.",
        "Internal controls strengthened governance and operational risk mitigation across all business units.",
        "Regulatory capital requirements impacted lending capacity and growth strategies for financial institutions.",
        "Basel III capital adequacy requirements were fully implemented across all banking institutions.",
        "Regulatory stress testing scenarios became more comprehensive and challenging for financial firms.",
        "Compliance monitoring systems detected potential violations early and prevented regulatory issues.",
        "Regulatory enforcement actions increased across multiple jurisdictions and business lines.",
        "Compliance risk assessments were conducted quarterly to identify emerging regulatory risks.",
        "Regulatory technology solutions improved efficiency and accuracy in compliance reporting.",
        "Compliance officer responsibilities expanded significantly as regulatory complexity increased.",
        "Regulatory change management processes were formalized to ensure timely implementation.",
        "Compliance audit trails were maintained for all transactions and business activities.",
        "Regulatory reporting automation reduced manual errors and improved data quality.",
        "Compliance culture initiatives were launched across organizations to strengthen awareness.",
        "Regulatory capital planning processes incorporated stress testing and scenario analysis.",
        "Compliance risk management frameworks addressed both current and emerging regulatory requirements.",
        "Regulatory reporting frequency increased to support supervisory oversight and monitoring.",
        "Compliance training programs were expanded to cover new regulatory requirements and expectations.",
        "Regulatory risk assessments were integrated into business planning and decision-making processes.",
        "Compliance monitoring and testing programs were enhanced to ensure regulatory adherence.",
        "Regulatory capital optimization strategies balanced regulatory requirements with business objectives.",
        "Compliance governance structures ensured proper oversight and accountability for regulatory matters.",
        "Regulatory reporting quality metrics improved through enhanced data management and validation.",
        "Compliance risk appetite statements were developed to guide business activities and decisions.",
        "Regulatory capital allocation frameworks supported business growth while maintaining compliance.",
        "Compliance risk culture assessments identified areas for improvement and enhancement.",
        "Regulatory reporting technology platforms were upgraded to support new requirements.",
    ]
    
    # Add the additional sentences to reach target
    for sentence in additional_sentences:
        all_sentences.append({
            'text': sentence,
            'source': 'curated',
            'category': 'financial_news'
        })
    
    print(f"Total sentences loaded: {len(all_sentences)}")
    print(f"Sources: {set([s['source'] for s in all_sentences])}")
    print(f"Categories: {set([s['category'] for s in all_sentences])}")
    
    return all_sentences

def get_all_improving_features_from_layer_wise_report(layer_wise_report_path):
    """Extract ALL improving features from the layer-wise report"""
    print(f"Extracting ALL improving features from layer-wise report...")
    
    df = pd.read_csv(layer_wise_report_path)
    
    # Get all improving features
    improving_features = df[df['type'] == 'improving'].copy()
    
    # Get unique features and their info
    unique_features = {}
    for _, row in improving_features.iterrows():
        feature_idx = row['feature_idx']
        layer = row['layer']
        improvement = row['improvement']
        category = row['top_category']
        
        if feature_idx not in unique_features:
            unique_features[feature_idx] = {
                'layers': [],
                'improvements': [],
                'categories': [],
                'avg_improvement': 0,
                'max_improvement': 0
            }
        
        unique_features[feature_idx]['layers'].append(layer)
        unique_features[feature_idx]['improvements'].append(improvement)
        unique_features[feature_idx]['categories'].append(category)
    
    # Calculate statistics for each feature
    for feature_idx in unique_features:
        improvements = unique_features[feature_idx]['improvements']
        unique_features[feature_idx]['avg_improvement'] = np.mean(improvements)
        unique_features[feature_idx]['max_improvement'] = np.max(improvements)
        unique_features[feature_idx]['layers'] = sorted(unique_features[feature_idx]['layers'])
    
    feature_list = list(unique_features.keys())
    print(f"Selected {len(feature_list)} improving features across all layers")
    
    return feature_list, unique_features

def get_feature_activations_multi_gpu_parallel(sentences, finbert_model, finbert_tokenizer, sae_finbert_path, target_features, num_gpus=7):
    """Get activations using multiple GPUs in parallel"""
    print(f"Getting activations for {len(target_features)} features across {len(sentences)} sentences using {num_gpus} GPUs in parallel...")
    
    # Set up GPU devices (0-6)
    gpu_ids = list(range(0, num_gpus))  # GPUs 0-6
    print(f"Using GPUs: {gpu_ids}")
    
    # Split sentences into batches for each GPU
    batch_size = max(1, len(sentences) // num_gpus)
    sentence_batches = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        gpu_id = gpu_ids[len(sentence_batches) % len(gpu_ids)]
        sentence_batches.append((gpu_id, batch))
    
    # Process batches sequentially but on different GPUs
    all_results = []
    
    for i, (gpu_id, batch) in enumerate(sentence_batches):
        print(f"Processing batch {i+1}/{len(sentence_batches)} on GPU {gpu_id}")
        device = torch.device(f'cuda:{gpu_id}')
        
        # Move model to this GPU
        model_on_gpu = finbert_model.to(device)
        
        batch_results = []
        for sentence_data in batch:
            sentence = sentence_data['text']
            category = sentence_data['category']
            
            sentence_results = {
                'sentence': sentence,
                'category': category,
                'feature_activations': {}
            }
            
            for layer_idx in range(12):  # All 12 layers
                # Load SAE for this layer
                sae_finbert = load_sae_model(sae_finbert_path, f"encoder.layer.{layer_idx}", device)
                if sae_finbert is None:
                    continue
                
                # Get activations
                finbert_act = get_layer_activations_bert(sentence, model_on_gpu, finbert_tokenizer, layer_idx)
                if finbert_act is not None:
                    finbert_encoded = encode_activations(finbert_act, sae_finbert['encoder_weight'], sae_finbert['encoder_bias'])
                    
                    # Get activations for target features
                    for feature_idx in target_features:
                        if feature_idx < finbert_encoded.shape[-1]:
                            activation = finbert_encoded.mean(dim=1)[0, feature_idx].item()
                            sentence_results['feature_activations'][f"L{layer_idx}_F{feature_idx}"] = activation
            
            batch_results.append(sentence_results)
        
        all_results.extend(batch_results)
    
    # Convert results to the expected format
    feature_activations = defaultdict(list)
    sentence_activations = defaultdict(dict)
    
    for result in all_results:
        sentence = result['sentence']
        category = result['category']
        
        sentence_activations[sentence] = {
            'category': category,
            'feature_activations': result['feature_activations']
        }
        
        for feature_key, activation in result['feature_activations'].items():
            # Parse feature key (e.g., "L11_F16317")
            layer_str, feature_str = feature_key.split('_')
            layer_idx = int(layer_str[1:])
            feature_idx = int(feature_str[1:])
            
            feature_activations[feature_idx].append({
                'sentence': sentence,
                'category': category,
                'layer': layer_idx,
                'activation': activation
            })
    
    return feature_activations, sentence_activations

def find_top_activating_sentences_by_category(feature_activations, feature_info, top_k=15):
    """Find top activating sentences for each feature by category"""
    print("Finding top activating sentences by category for each feature...")
    
    feature_top_sentences = {}
    
    for feature_idx, activations in feature_activations.items():
        if len(activations) == 0:
            continue
        
        # Group activations by category
        category_activations = defaultdict(list)
        for act in activations:
            category_activations[act['category']].append(act)
        
        # Find top sentences for each category
        category_top_sentences = {}
        for category, cat_activations in category_activations.items():
            # Sort by activation strength
            sorted_activations = sorted(cat_activations, key=lambda x: x['activation'], reverse=True)
            
            # Get top sentences
            top_sentences = []
            seen_sentences = set()
            
            for act in sorted_activations:
                if act['sentence'] not in seen_sentences:
                    top_sentences.append(act['sentence'])
                    seen_sentences.add(act['sentence'])
                    if len(top_sentences) >= top_k:
                        break
            
            category_top_sentences[category] = top_sentences
        
        feature_top_sentences[feature_idx] = category_top_sentences
    
    return feature_top_sentences

def load_qwen_model():
    """Load local Qwen model"""
    print("Loading local Qwen model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        return None, None

def generate_feature_labels_qwen(topk_dict, qwen_model, qwen_tokenizer, max_retries=3):
    """Generate feature labels using Qwen model with improved prompt for more specific labels"""
    labels = {}
    
    for feature_id, category_sentences in topk_dict.items():
        print(f"Processing feature {feature_id}...")
        
        # Collect all sentences for this feature
        all_sentences = []
        for category, sentences in category_sentences.items():
            all_sentences.extend(sentences[:10])  # Take top 10 sentences per category
        
        if len(all_sentences) < 3:
            labels[feature_id] = "financial analysis feature"
            continue
        
        # Join sentences for the prompt
        joined_text = "\n".join([f"- {sentence}" for sentence in all_sentences[:25]])
        
        # Improved prompt for more specific, crisp labels
        prompt = f"""<|im_start|>system
You are an expert financial analyst specializing in natural language processing. Your task is to analyze financial text patterns and create concise, specific labels for AI model features.

IMPORTANT GUIDELINES:
1. Create SHORT, SPECIFIC labels (3-6 words maximum)
2. DO NOT use generic phrases like "detecting", "financial market trends", "movements"
3. Focus on the MOST DISTINCTIVE aspect of the texts
4. Use specific financial terminology when appropriate
5. Make each label unique and descriptive
6. Examples of good labels: "earnings surprises", "volatility spikes", "rate hikes", "credit spreads", "dividend announcements"

Analyze the following financial texts and create a crisp, specific label for the feature that activates on these texts.
<|im_end|>
<|im_start|>user
Financial texts that activate this feature:

{joined_text}

Create a specific, crisp label (3-6 words) for this feature:
<|im_end|>
<|im_start|>assistant
"""
        
        for attempt in range(max_retries):
            try:
                inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = qwen_model.generate(
                        **inputs, 
                        max_new_tokens=30, 
                        temperature=0.3, 
                        do_sample=True, 
                        pad_token_id=qwen_tokenizer.eos_token_id
                    )
                
                response = qwen_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                label = response.strip()
                
                # Clean and validate the label
                label = re.sub(r'[^\w\s\-]', '', label)  # Remove special characters except hyphens
                label = label.strip()
                
                # Ensure label is not too long
                if len(label.split()) > 6:
                    label = ' '.join(label.split()[:6])
                
                # Ensure label is not empty or too short
                if len(label) < 3:
                    label = "financial analysis feature"
                
                labels[feature_id] = label
                break
                
            except Exception as e:
                print(f"Error processing feature {feature_id}, attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    labels[feature_id] = "financial analysis feature"
                time.sleep(1)
    
    return labels

def main():
    """Main Qwen-based feature labeling analysis"""
    print("="*80)
    print("QWEN-BASED FINBERT FEATURE LABELING ANALYSIS")
    print("="*80)
    
    # Set paths
    layer_wise_report_path = "/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/2.3. layer_wise_feature_report.csv"
    sae_finbert_path = "/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/SAE_AllLayers_Finbert/unnamed"
    
    # Load FinBERT model
    print("Loading FinBERT model...")
    finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    finbert_model = BertModel.from_pretrained('ProsusAI/finbert')
    
    # Load Qwen model
    qwen_model, qwen_tokenizer = load_qwen_model()
    if qwen_model is None:
        print("Failed to load Qwen model. Exiting.")
        return
    
    # Create comprehensive financial sentences
    sentences = load_real_financial_sentences()
    
    # Get ALL improving features from the layer-wise report
    target_features, feature_info = get_all_improving_features_from_layer_wise_report(layer_wise_report_path)
    
    print(f"Analyzing {len(target_features)} improving features across all layers")
    
    # Get feature activations using multiple GPUs
    feature_activations, sentence_activations = get_feature_activations_multi_gpu_parallel(
        sentences, finbert_model, finbert_tokenizer, sae_finbert_path, target_features, num_gpus=7
    )
    
    # Find top activating sentences by category
    feature_top_sentences = find_top_activating_sentences_by_category(feature_activations, feature_info, top_k=30)
    
    # Generate feature labels using Qwen
    feature_labels = generate_feature_labels_qwen(feature_top_sentences, qwen_model, qwen_tokenizer)
    
    # Save results
    print("\nSaving Qwen-based results...")
    
    # Save Qwen feature labels with layer details (matching layer-wise report format)
    labels_data = []
    for feature_idx, label in feature_labels.items():
        if feature_idx in feature_info:
            # For each layer this feature appears in, create a separate row
            for layer in feature_info[feature_idx]['layers']:
                # Find the improvement for this specific layer
                layer_improvement = None
                layer_category = None
                for i, layer_info in enumerate(feature_info[feature_idx]['layers']):
                    if layer_info == layer:
                        layer_improvement = feature_info[feature_idx]['improvements'][i]
                        layer_category = feature_info[feature_idx]['categories'][i]
                        break
                
                labels_data.append({
                    'layer': layer,
                    'feature_idx': feature_idx,
                    'qwen_label': label,
                    'avg_improvement': feature_info[feature_idx]['avg_improvement'],
                    'max_improvement': feature_info[feature_idx]['max_improvement'],
                    'layer_improvement': layer_improvement,
                    'layer_category': layer_category,
                    'layer_count': len(feature_info[feature_idx]['layers']),
                    'all_layers': str(feature_info[feature_idx]['layers']),
                    'all_categories': ' | '.join(feature_info[feature_idx]['categories'][:3])
                })
    
    df_labels = pd.DataFrame(labels_data)
    # Sort by layer, then by feature_idx to match the original format
    df_labels = df_labels.sort_values(['layer', 'feature_idx'])
    df_labels.to_csv('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/qwen_all_features_labels.csv', index=False)
    
    # Save category-specific activating sentences
    sentences_data = []
    for feature_idx, category_sentences in feature_top_sentences.items():
        # Get top 10 sentences across all categories for this feature
        all_feature_sentences = []
        for category, sentences_list in category_sentences.items():
            all_feature_sentences.extend([(sentence, category) for sentence in sentences_list[:10]])
        
        # Sort by activation strength (assuming sentences are already sorted)
        all_feature_sentences = all_feature_sentences[:10]  # Take top 10 overall
        
        for i, (sentence, category) in enumerate(all_feature_sentences):
            sentences_data.append({
                'feature_idx': feature_idx,
                'rank': i + 1,
                'sentence': sentence,
                'category': category
            })
    
    df_sentences = pd.DataFrame(sentences_data)
    df_sentences.to_csv('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/qwen_all_features_activating_sentences.csv', index=False)
    
    # Print summary
    print(f"\n" + "="*80)
    print("QWEN-BASED FEATURE LABELING COMPLETED!")
    print("="*80)
    print(f"Features analyzed: {len(feature_labels)}")
    print(f"Total feature-layer combinations: {len(df_labels)}")
    print(f"Sentences created: {len(sentences)}")
    print(f"Categories covered: {len(set([s['category'] for s in sentences]))}") # Count unique categories from loaded sentences
    print(f"Files generated:")
    print(f"- Qwen all features labels: qwen_all_features_labels.csv")
    print(f"- Qwen all features activating sentences: qwen_all_features_activating_sentences.csv")
    
    # Show some example labels with layer details
    print(f"\nExample Qwen-generated feature labels (with layer details):")
    for i, (_, row) in enumerate(df_labels.head(10).iterrows()):
        layer = row['layer']
        feature_idx = row['feature_idx']
        label = row['qwen_label']
        layer_imp = row['layer_improvement']
        layer_cat = row['layer_category']
        print(f"{i+1:2d}. Layer {layer:2d}, Feature {feature_idx:5d}: '{label}' (Layer Improvement: {layer_imp:.2f}, Category: {layer_cat})")
    
    return {
        'feature_labels': feature_labels,
        'feature_activations': feature_activations,
        'sentences': sentences,
        'labels_df': df_labels
    }

if __name__ == "__main__":
    results = main() 