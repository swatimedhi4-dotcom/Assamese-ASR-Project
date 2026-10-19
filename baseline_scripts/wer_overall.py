import pandas as pd
import jiwer

df = pd.read_csv("results.csv")



references = df["reference"].tolist()
hypotheses = df["hypothesis"].tolist()

wer = jiwer.wer(references, hypotheses)
cer = jiwer.cer(references, hypotheses)


print("WER:", wer * 100, "%")
print("CER:", cer * 100, "%")