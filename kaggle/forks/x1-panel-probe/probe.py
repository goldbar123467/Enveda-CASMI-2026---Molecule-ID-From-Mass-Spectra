# CPU probe: describe the labelled panel datasets used for offline evaluation. No training.
import glob, json, os
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
INP = "/kaggle/input"
for p in sorted(glob.glob(INP + "/*")):
    print("INPUT", p)
pats = ["**/labelled_candidate_scores.parquet", "**/all_features.parquet", "**/per_query_results.csv",
        "**/frozen_candidate_evidence.parquet", "**/features.parquet", "**/effective_query_spectra.parquet",
        "**/structure_splits.parquet", "**/split_summary.parquet", "**/validation_predictions.csv",
        "**/panel*.parquet", "**/*panel*.csv", "**/*benchmark*.parquet"]
seen = set()
for pat in pats:
    for f in sorted(glob.glob(INP + "/" + pat, recursive=True))[:3]:
        if f in seen: continue
        seen.add(f)
        print("=" * 120); print(f, os.path.getsize(f))
        try:
            df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
            print(df.shape); print(df.dtypes.to_string()); print(df.head(5).to_string()[:4000])
            for c in ("molecule_id", "query_id", "inchikey14", "label", "is_correct", "correct"):
                if c in df.columns: print("col", c, "nunique", df[c].nunique(), "mean" if df[c].dtype != object else "", df[c].mean() if df[c].dtype != object else "")
        except Exception as e:
            print("ERR", e)
for f in sorted(glob.glob(INP + "/**/*.parquet", recursive=True)):
    print("PARQUET", f, os.path.getsize(f))
for f in sorted(glob.glob(INP + "/**/README*", recursive=True))[:12]:
    print("=" * 120); print(f); print(open(f, errors="replace").read()[:3000])
