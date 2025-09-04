#!/usr/bin/env python3
"""
Generate CSV from the summary report
"""

import json
import pandas as pd

# Load the summary report
with open('runs/llama2_7b_layer16_finance_autointerp/summary_report.json', 'r') as f:
    data = json.load(f)

# Extract data for CSV
csv_data = []
for latent_key, result in data['results_summary'].items():
    # Extract latent number from key like "layers.16_latent0"
    latent_number = int(latent_key.split('_')[-1].replace('latent', ''))
    
    # Get explanation and truncate to reasonable length
    explanation = result['explanation']
    words = explanation.split()
    if words:
        label = ' '.join(words[:3])  # Take first 3 words
        if len(label) > 30:  # Truncate if too long
            label = label[:27] + "..."
    else:
        label = "UNKNOWN"
    
    csv_data.append({
        'latent_number': latent_number,
        'label': label,
        'f1_score': result['f1_score']
    })

# Sort by latent number
csv_data.sort(key=lambda x: x['latent_number'])

# Create DataFrame and save CSV
df = pd.DataFrame(csv_data)
df.to_csv('runs/llama2_7b_layer16_finance_autointerp/latent_results.csv', index=False)

print("📊 CSV generated successfully!")
print("Columns: latent_number, label, f1_score")
print("\n📋 CSV Content:")
print("=" * 50)
print(df.to_string(index=False))
