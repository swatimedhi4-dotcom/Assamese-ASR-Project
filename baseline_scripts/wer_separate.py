import pandas as pd
import jiwer

df = pd.read_csv("results.csv")
print(df.columns.tolist())
for _, row in df.iterrows():
    print("Audio number:", _ + 1)
    print("WER:", jiwer.wer(
        row["reference"], row["hypothesis"]
    ) * 100, "%")
    print("CER:", jiwer.cer(
        row["reference"], row["hypothesis"]
    ) * 100, "%")
