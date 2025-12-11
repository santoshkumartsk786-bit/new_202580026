"""
Build FAISS Index for RAG System with AUTOMATIC SENTIMENT DETECTION
Works with ANY dataset - detects sentiment from review text using keyword analysis

Run this script locally BEFORE deploying to Streamlit Cloud
It will generate models/faiss_index.bin and models/review_metadata.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import re
import json

print("="*80)
print("BUILDING FAISS INDEX WITH AUTO SENTIMENT DETECTION")
print("="*80)

# ============================================================================
# SENTIMENT DETECTION FUNCTIONS
# ============================================================================

# Positive keywords
POSITIVE_WORDS = {
    'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'outstanding',
    'great', 'superb', 'perfect', 'awesome', 'incredible', 'magnificent',
    'love', 'loved', 'beautiful', 'best', 'masterpiece', 'stunning',
    'entertaining', 'enjoyable', 'fun', 'impressive', 'recommend', 'recommended',
    'powerful', 'moving', 'touching', 'hilarious', 'thrilling', 'exciting',
    'captivating', 'gripping', 'compelling', 'fascinating', 'delightful',
    'charming', 'heartwarming', 'uplifting', 'inspiring', 'refreshing'
}

# Negative keywords
NEGATIVE_WORDS = {
    'awful', 'terrible', 'horrible', 'bad', 'worst', 'poor', 'boring',
    'disappointing', 'disappointed', 'waste', 'wasted', 'pathetic', 'ridiculous',
    'stupid', 'dull', 'bland', 'mediocre', 'weak', 'fails', 'failed',
    'poorly', 'badly', 'hate', 'hated', 'annoying', 'irritating',
    'awful', 'garbage', 'trash', 'mess', 'disaster', 'unwatchable',
    'tedious', 'slow', 'dragging', 'confusing', 'confused', 'pointless',
    'unfunny', 'cringeworthy', 'cringe', 'overrated', 'pretentious'
}

def detect_sentiment_advanced(text):
    """
    Advanced sentiment detection with negation handling

    Args:
        text: Review text

    Returns:
        'positive' or 'negative'
    """
    if not isinstance(text, str):
        return 'neutral'

    text_lower = text.lower()

    # Negation words that flip sentiment
    negations = {'not', 'no', "n't", 'never', 'neither', 'nobody', 'nothing'}

    # Split into words
    words = re.findall(r'\b\w+\b', text_lower)

    pos_score = 0
    neg_score = 0

    for i, word in enumerate(words):
        # Check if previous word is negation
        is_negated = (i > 0 and words[i-1] in negations) or \
                     (i > 1 and words[i-2] in negations)

        if word in POSITIVE_WORDS:
            if is_negated:
                neg_score += 1  # "not good" counts as negative
            else:
                pos_score += 1

        if word in NEGATIVE_WORDS:
            if is_negated:
                pos_score += 1  # "not bad" counts as positive
            else:
                neg_score += 1

    # Decide sentiment
    if pos_score == 0 and neg_score == 0:
        return 'neutral'
    elif pos_score > neg_score:
        return 'positive'
    elif neg_score > pos_score:
        return 'negative'
    else:
        return 'positive'  # Default on tie

# ============================================================================
# MAIN BUILD PROCESS
# ============================================================================

# Create models directory
Path("models").mkdir(exist_ok=True)

# Step 1: Load dataset
print("\n[1/6] Loading dataset...")

# Try to find the CSV file
possible_files = [
    'imdb_sup.csv',
    'data/imdb_sup.csv',
    'imdb_reviews.csv',
    'data/imdb_reviews.csv'
]

csv_file = None
for file in possible_files:
    if os.path.exists(file):
        csv_file = file
        break

if csv_file is None:
    print("❌ ERROR: Could not find dataset file!")
    print("   Looked for: " + ", ".join(possible_files))
    print("\n   Please ensure your CSV file is in one of these locations:")
    print("   - imdb_sup.csv (in current directory)")
    print("   - data/imdb_sup.csv")
    exit(1)

print(f"✓ Found dataset: {csv_file}")
df = pd.read_csv(csv_file)

print(f"Original columns: {df.columns.tolist()}")

# Standardize column names (case-insensitive)
column_mapping = {}
for col in df.columns:
    col_lower = col.lower().strip()
    if col_lower in ['review', 'reviews', 'text']:
        column_mapping[col] = 'text'
    elif col_lower in ['sentiment', 'sentimemt']:
        column_mapping[col] = 'sentiment'
    elif col_lower == 'rating':
        column_mapping[col] = 'rating'

df = df.rename(columns=column_mapping)

# Verify we have text column
if 'text' not in df.columns:
    raise ValueError(f"Dataset must have a 'Review' or 'text' column. Found: {df.columns.tolist()}")

print(f"✓ Loaded {len(df):,} reviews")

# Step 2: Handle sentiment column
print("\n[2/6] Processing sentiment...")

sentiment_source = 'auto_detected'

if 'sentiment' in df.columns:
    print("✓ Sentiment column found")

    # Check if numeric (0/1) or text (positive/negative)
    if df['sentiment'].dtype in ['int64', 'float64']:
        print("  Converting numeric sentiment (0→negative, 1→positive)")
        df['sentiment'] = df['sentiment'].apply(lambda x: 'positive' if x == 1 else 'negative')
        sentiment_source = 'existing_column_numeric'
    else:
        # Text sentiment - standardize
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
        df['sentiment'] = df['sentiment'].map({
            'positive': 'positive',
            'pos': 'positive',
            'good': 'positive',
            'negative': 'negative',
            'neg': 'negative',
            'bad': 'negative'
        })
        sentiment_source = 'existing_column_text'

    # Handle any remaining unmapped values
    if df['sentiment'].isnull().any():
        unmapped_count = df['sentiment'].isnull().sum()
        print(f"  WARNING: {unmapped_count} unmapped sentiment values")
        print("  Detecting sentiment from text for these rows...")
        df.loc[df['sentiment'].isnull(), 'sentiment'] = df.loc[df['sentiment'].isnull(), 'text'].apply(detect_sentiment_advanced)
        sentiment_source = 'hybrid'

else:
    print("✗ No sentiment column found")
    print("  AUTO-DETECTING sentiment from review text...")

    # Apply sentiment detection
    df['sentiment'] = df['text'].apply(detect_sentiment_advanced)
    sentiment_source = 'auto_detected'

    # Remove neutral sentiments
    neutral_count = (df['sentiment'] == 'neutral').sum()
    if neutral_count > 0:
        print(f"  Removing {neutral_count} neutral reviews (no clear sentiment)")
        df = df[df['sentiment'] != 'neutral']

# Final sentiment counts
pos_count = (df['sentiment'] == 'positive').sum()
neg_count = (df['sentiment'] == 'negative').sum()

print(f"\n✓ Sentiment distribution:")
print(f"  Positive: {pos_count:,} ({pos_count/len(df)*100:.1f}%)")
print(f"  Negative: {neg_count:,} ({neg_count/len(df)*100:.1f}%)")

if pos_count == 0 or neg_count == 0:
    print("\n⚠️  WARNING: Only one sentiment type detected!")
    print("     This might indicate a problem with the data or detection.")
    response = input("     Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit()

# Add review IDs
df['review_id'] = df.index

# Step 3: Load embedding model
print("\n[3/6] Loading sentence transformer model...")
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Model loaded")

# Step 4: Generate embeddings
print("\n[4/6] Generating embeddings (this may take 5-10 minutes)...")
review_texts = df['text'].tolist()
embeddings = encoder.encode(
    review_texts,
    show_progress_bar=True,
    batch_size=64,
    convert_to_numpy=True
)
print(f"✓ Generated {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")

# Step 5: Build FAISS index
print("\n[5/6] Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
print(f"✓ FAISS index built with {index.ntotal:,} vectors")

# Step 6: Save everything
print("\n[6/6] Saving index and metadata...")

# Save FAISS index
faiss.write_index(index, "models/faiss_index.bin")
print("✓ Saved FAISS index to models/faiss_index.bin")

# Save metadata
metadata = {
    'review_ids': df['review_id'].tolist(),
    'texts': df['text'].tolist(),
    'sentiments': df['sentiment'].tolist(),
    'ratings': df['rating'].tolist() if 'rating' in df.columns else None
}

with open('models/review_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Saved metadata to models/review_metadata.pkl")

# Print file sizes
index_size = os.path.getsize('models/faiss_index.bin') / (1024**2)
meta_size = os.path.getsize('models/review_metadata.pkl') / (1024**2)

print(f"\n{'='*80}")
print("INDEX BUILD COMPLETE!")
print(f"{'='*80}")
print(f"Total reviews indexed: {len(df):,}")
print(f"  Positive reviews: {pos_count:,}")
print(f"  Negative reviews: {neg_count:,}")
print(f"\nFile sizes:")
print(f"  FAISS index: {index_size:.2f} MB")
print(f"  Metadata: {meta_size:.2f} MB")
print(f"  Total: {index_size + meta_size:.2f} MB")
print(f"\nSentiment detection method: {sentiment_source.replace('_', ' ').title()}")
print("\nNext steps:")
print("1. Verify sentiment distribution looks correct")
print("2. Commit models/faiss_index.bin and models/review_metadata.pkl to GitHub")
print("3. Deploy app.py to Streamlit Cloud")
print("="*80)

# Save sentiment detection report
report = {
    'total_reviews': len(df),
    'positive_reviews': int(pos_count),
    'negative_reviews': int(neg_count),
    'positive_percentage': float(pos_count / len(df) * 100),
    'negative_percentage': float(neg_count / len(df) * 100),
    'sentiment_source': sentiment_source,
    'index_size_mb': float(index_size),
    'metadata_size_mb': float(meta_size)
}

with open('models/build_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n✓ Build report saved to models/build_report.json")