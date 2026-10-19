"""Turn your raw clips + a simple CSV into the dataset folder.

1. Put your raw audio in one folder (any common format).
2. Fill raw_metadata.csv (see raw_metadata_template.csv): one row per clip.
   Required columns: original_file, speaker_id, transcript_raw
   Optional (kept if present): scenario, age_group, gender, region, environment,
   noise_source, background_speech, device, consent, recording_date, take_number, ...
3. Run:
   python make_dataset.py --raw_dir raw_audio --raw_csv raw_metadata.csv --out data --split test
Output:  data/<split>/<utt_id>.wav (16 kHz mono) + data/<split>/metadata.csv
The folder works directly with eval_asr.py and with Hugging Face `audiofolder`.
"""
import argparse, os
import pandas as pd
import librosa
import soundfile as sf
from normalize import normalize

REQUIRED = ["original_file", "speaker_id", "transcript_raw"]


def given(v):
    return v is not None and not pd.isna(v) and str(v).strip() != ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    a = ap.parse_args()

    df = pd.read_csv(a.raw_csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"raw csv is missing columns: {missing}")

    out_dir = os.path.join(a.out, a.split)
    os.makedirs(out_dir, exist_ok=True)
    counters, rows, problems = {}, [], []

    for i, r in df.iterrows():
        src = os.path.join(a.raw_dir, str(r["original_file"]))
        if not os.path.exists(src):
            problems.append(f"row {i+2}: file not found: {src}")
            continue
        if not given(r["transcript_raw"]):
            problems.append(f"row {i+2}: empty transcript_raw (clip skipped)")
            continue
        wav, _ = librosa.load(src, sr=16000, mono=True)
        dur = len(wav) / 16000
        spk = str(r["speaker_id"])
        scen = str(r["scenario"]) if "scenario" in df.columns and given(r.get("scenario")) else "read"
        key = (spk, scen)
        counters[key] = counters.get(key, 0) + 1
        utt_id = str(r["utt_id"]) if "utt_id" in df.columns and given(r.get("utt_id")) \
            else f"{spk}_{scen}_{counters[key]:03d}"
        sf.write(os.path.join(out_dir, f"{utt_id}.wav"), wav, 16000, subtype="PCM_16")
        if dur < 0.5 or dur > 30:
            problems.append(f"{utt_id}: unusual duration {dur:.1f}s, check it")

        row = r.drop(labels=["original_file"]).to_dict()  # file names may contain real names
        row.update(file_name=f"{utt_id}.wav", utt_id=utt_id, scenario=scen,
                   sample_rate_hz=16000, duration_sec=round(dur, 2),
                   transcript_norm=normalize(r["transcript_raw"]), split=a.split)
        rows.append(row)

    meta = pd.DataFrame(rows)
    first = ["file_name", "utt_id", "speaker_id", "scenario", "transcript_raw", "transcript_norm"]
    meta = meta[first + [c for c in meta.columns if c not in first]]
    meta.to_csv(os.path.join(out_dir, "metadata.csv"), index=False, encoding="utf-8")

    print(f"clips written : {len(meta)}")
    print(f"speakers      : {meta['speaker_id'].nunique()}")
    print(f"total minutes : {meta['duration_sec'].sum()/60:.1f}")
    for p in problems:
        print("WARNING:", p)


if __name__ == "__main__":
    main()
