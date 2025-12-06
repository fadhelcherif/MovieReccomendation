"""
Create a smaller sample dataset for GitHub deployment
This reduces the dataset size while keeping high-quality movies
"""
import pandas as pd

print("📂 Loading full datasets...")
df_full = pd.read_csv('movies_cleaned_full.csv', low_memory=False)
df_model = pd.read_csv('movies_preprocessed_model.csv', low_memory=False)

# Sample strategy: Keep top movies by popularity and rating
print("🎯 Sampling top movies...")

# Sample 20,000 movies instead of 127k (reduces from 70MB to ~11MB)
n_sample = 20000

# Prioritize high-rated and popular movies
if 'vote_average' in df_model.columns and 'vote_count' in df_model.columns:
    df_model['score'] = df_model['vote_average'] * df_model['vote_count'].apply(lambda x: min(x, 1000))
    df_model_sample = df_model.nlargest(n_sample, 'score').drop('score', axis=1)
else:
    df_model_sample = df_model.head(n_sample)

# Match the full dataset
titles_to_keep = set(df_model_sample['title'].values)
df_full_sample = df_full[df_full['title'].isin(titles_to_keep)]

print(f"✅ Sampled {len(df_model_sample):,} movies from {len(df_model):,}")
print(f"📊 Full dataset sample: {len(df_full_sample):,} movies")

# Save sampled datasets
df_full_sample.to_csv('movies_cleaned_full_sample.csv', index=False)
df_model_sample.to_csv('movies_preprocessed_model_sample.csv', index=False)

print("💾 Saved sample datasets:")
print("  - movies_cleaned_full_sample.csv")
print("  - movies_preprocessed_model_sample.csv")
print("\n🚀 Now update app.py to use these sample files for deployment!")
