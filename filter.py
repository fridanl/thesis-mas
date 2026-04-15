import pandas as pd
 
# ── Paths ──────────────────────────────────────────────────────────────────────
MAIN_CSV   = "/home/rp-fril-mhpe/second/gemma-3-4b-sarcasm.csv"
AGREE_IDS  = "/home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_self_interaction_agree_subsampled.csv"
BB_IDS     = "/home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_self_interaction_disagree_subsampled.csv"
OUTPUT_CSV = "/home/rp-fril-mhpe/second/gemma-3-4b-sarcasm_filtered.csv"
 
# ── Load data ──────────────────────────────────────────────────────────────────
df        = pd.read_csv(MAIN_CSV)
agree_ids = pd.read_csv(AGREE_IDS)["id"].unique()
bb_ids    = pd.read_csv(BB_IDS)["id"].unique()
 
print(f"Loaded main CSV:        {len(df):,} rows")
print(f"Agree subsample IDs:    {len(agree_ids):,} unique ids")
print(f"B:B subsample IDs:      {len(bb_ids):,} unique ids")
 
# ── Define self-interaction mask ───────────────────────────────────────────────
MODEL = "gemma-3-4b-sarcasm"
is_self = (df["model_receiver"] == MODEL) & (df["model_sender"] == MODEL)
 
print(f"\nSelf-interaction rows:  {is_self.sum():,}")
 
# ── Rows NOT involved in self-interaction → keep as-is ────────────────────────
df_other = df[~is_self].copy()
 
# ── Self-interaction rows, split by match_type ────────────────────────────────
df_self = df[is_self].copy()
 
# 1:1 and 0:0 → keep only if id in agree subsample file
mask_agree = df_self["match_type"].isin(["1:1", "0:0"])
df_agree_keep = df_self[mask_agree & df_self["id"].isin(agree_ids)]
 
# B:B → keep only if id in B:B subsample file
mask_bb = df_self["match_type"] == "B:B"
df_bb_keep = df_self[mask_bb & df_self["id"].isin(bb_ids)]
 
# Any other match_type within self-interaction → keep unchanged
mask_other_mt = ~mask_agree & ~mask_bb
df_self_other = df_self[mask_other_mt]
 
print(f"\nSelf-interaction breakdown:")
print(f"  1:1 / 0:0 rows before filter: {mask_agree.sum():,}")
print(f"  1:1 / 0:0 rows after filter:  {len(df_agree_keep):,}")
print(f"  B:B rows before filter:        {mask_bb.sum():,}")
print(f"  B:B rows after filter:         {len(df_bb_keep):,}")
print(f"  Other match_type (unchanged):  {len(df_self_other):,}")
 
# ── Reassemble ─────────────────────────────────────────────────────────────────
df_filtered = pd.concat(
    [df_other, df_agree_keep, df_bb_keep, df_self_other],
    ignore_index=True
)
 
# Restore original row order if there's a meaningful sort column
# df_filtered = df_filtered.sort_values("id").reset_index(drop=True)
 
print(f"\nFinal row count: {len(df_filtered):,}  (was {len(df):,})")
print(f"Rows removed:    {len(df) - len(df_filtered):,}")