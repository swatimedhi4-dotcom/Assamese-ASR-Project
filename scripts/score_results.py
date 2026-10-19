"""Score a results.csv (columns: audio_path, reference, hypothesis) the same way every time.
Prints corpus-level WER/CER three ways (raw, normalized, normalized with ৰ/ৱ collapsed)
and a per-clip table. Usage: python score_results.py ../baseline_scripts/results.csv
"""
import sys
import pandas as pd
import jiwer
from normalize import normalize

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(path)
n0 = len(df)
df = df.dropna(subset=["reference", "hypothesis"]).copy()
print(f"rows in file: {n0}, usable (non-empty): {len(df)}")

variants = {
    "raw (no normalization)": lambda s: str(s).strip(),
    "normalized (strict)": lambda s: normalize(s),
    "normalized (r/w collapsed)": lambda s: normalize(s, True),
}
for name, fn in variants.items():
    r = df["reference"].map(fn); h = df["hypothesis"].map(fn)
    keep = r != ""
    r, h = r[keep].tolist(), h[keep].tolist()
    print(f"{name:28s} WER {jiwer.wer(r, h)*100:6.2f}%   CER {jiwer.cer(r, h)*100:6.2f}%")

df["ref_n"] = df["reference"].map(normalize); df["hyp_n"] = df["hypothesis"].map(normalize)
df["WER_%"] = [round(jiwer.wer(r, h) * 100, 1) for r, h in zip(df["ref_n"], df["hyp_n"])]
df["CER_%"] = [round(jiwer.cer(r, h) * 100, 1) for r, h in zip(df["ref_n"], df["hyp_n"])]
print(df[["audio_path", "WER_%", "CER_%"]].to_string(index=False))
