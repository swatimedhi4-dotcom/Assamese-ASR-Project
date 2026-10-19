"""Evaluate a CTC ASR model on your test set.
Usage (baseline, zero-shot MMS with Assamese adapter):
  python eval_asr.py --metadata test/metadata.csv --audio_dir test \
      --model facebook/mms-1b-all --lang asm --out results_before
Usage (after fine-tuning): --model <your_output_dir> --lang asm --skip_adapter_load, keep the rest identical.
metadata.csv needs: file_name, transcript_raw  (extra columns are kept for slicing).
"""
import argparse, json, os
import pandas as pd
import jiwer
from normalize import normalize


def metrics(refs, hyps, collapse):
    r = [normalize(x, collapse) for x in refs]
    h = [normalize(x, collapse) for x in hyps]
    w = jiwer.process_words(r, h)
    n = w.hits + w.substitutions + w.deletions
    return {
        "WER": w.wer, "CER": jiwer.cer(r, h),
        "sub_rate": w.substitutions / n, "del_rate": w.deletions / n,
        "ins_rate": w.insertions / n, "n_ref_words": n,
        "SER": sum(a != b for a, b in zip(r, h)) / len(r),
    }


def transcribe(args, paths):
    import torch, librosa
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(dev).eval()
    if args.lang:  # MMS adapter languages
        proc.tokenizer.set_target_lang(args.lang)
        if not args.skip_adapter_load:  # baseline: load the pretrained asm adapter from the hub
            model.load_adapter(args.lang)
    out = []
    for p in paths:
        wav, _ = librosa.load(p, sr=16000, mono=True)
        x = proc(wav, sampling_rate=16000, return_tensors="pt").to(dev)
        with torch.no_grad():
            ids = torch.argmax(model(**x).logits, dim=-1)[0]
        out.append(proc.decode(ids))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--model", default="facebook/mms-1b-all")
    ap.add_argument("--lang", default="asm")
    ap.add_argument("--skip_adapter_load", action="store_true",
                    help="use for your fine-tuned model dir (adapter weights are already inside)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.metadata)
    paths = [os.path.join(a.audio_dir, f) for f in df["file_name"]]
    df["prediction"] = transcribe(a, paths)
    res = {"strict": metrics(df["transcript_raw"], df["prediction"], False),
           "collapsed_ra_wa": metrics(df["transcript_raw"], df["prediction"], True)}
    df.to_csv(os.path.join(a.out, "predictions.csv"), index=False)  # keep: re-score later
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2, ensure_ascii=False))
