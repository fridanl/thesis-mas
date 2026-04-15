import pandas as pd
 
# ── Paths ──────────────────────────────────────────────────────────────────────
MAIN_CSV   = "/home/rp-fril-mhpe/second/llama-3.1-8b-sarcasm.csv"
SELF_INTERACTION  = "/home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.1-8b-self-interaction_subsampled.csv"
OUTPUT_CSV = "/home/rp-fril-mhpe/second/llama-3.1-8b-sarcasm.csv"
 
# ── Load data ──────────────────────────────────────────────────────────────────
df        = pd.read_csv(MAIN_CSV)
interact    = pd.read_csv(SELF_INTERACTION)
 
print(f"Loaded main CSV:        {len(df):,} rows")
print(f'Input rows for self-interaction: {len(interact):,}')

# ── Define self-interaction mask ───────────────────────────────────────────────
MODEL = "llama-3.1-8b"
is_self = (df["model_receiver"] == MODEL) & (df["model_sender"] == MODEL)
 
print(f"\nSelf-interaction rows:  {is_self.sum():,}")
 
# ── Rows NOT involved in self-interaction → keep as-is ────────────────────────
df_other = df[~is_self].copy()
print(f"Size of df without self-interaction: {len(df_other)}")
 
# ── Self-interaction rows, split by match_type ────────────────────────────────
df_self = df[is_self].copy() 

# Restore original row order if there's a meaningful sort column
# df_filtered = df_filtered.sort_values("id").reset_index(drop=True)
 
print(f"\nFinal row count: {len(df_other):,}  (was {len(df):,})")
print(f"Rows removed:    {len(df) - len(df_self):,}")

if is_self.sum() == interact.shape[0]:
    df_other.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to:\n  {OUTPUT_CSV}")