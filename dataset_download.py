import pandas as pd
from config import RAW_DIR,RAW_FILE

# ── 1. Your existing project folder path ────────────────────────────────

RAW_DIR.mkdir(parents=True, exist_ok=True)
print(f"Folder ready: {RAW_DIR}")

# ── 2. Download all 3 GoEmotions raw files ──────────────────────────────
base_url = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/"

files = {
    "goemotions_1.csv": f"{base_url}goemotions_1.csv",
    "goemotions_2.csv": f"{base_url}goemotions_2.csv",
    "goemotions_3.csv": f"{base_url}goemotions_3.csv",
}

dataframes = []

for filename, url in files.items():
    save_path = RAW_DIR / filename
    print(f"Downloading {filename}...")
    
    df = pd.read_csv(url)
    df.to_csv(save_path, index=False)
    dataframes.append(df)
    
    print(f"Saved: {save_path} | Rows: {len(df)}")

# ── 3. Combine all 3 into one file ──────────────────────────────────────

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv(RAW_FILE, index=False)

# ── 4. Summary ───────────────────────────────────────────────────────────

print("\n" + "="*50)
print("DOWNLOAD SUMMARY")
print("="*50)
print(f"Total rows        : {len(combined_df):,}")
print(f"Unique comments   : {combined_df['id'].nunique():,}")
print(f"Total columns     : {len(combined_df.columns)}")
print(f"Combined file     : {RAW_FILE}")
print("="*50)
print("\nFIRST ROW PREVIEW:")
print(combined_df.head(1).T)
print("\nANNOTATORS PER COMMENT (distribution):")
print(combined_df.groupby('id')['rater_id'].count().value_counts().sort_index())



