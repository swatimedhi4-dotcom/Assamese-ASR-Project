# Assamese ASR: fine-tuning and evaluation (work in progress)

Goal: improve automatic speech recognition for Assamese by fine-tuning a pretrained multilingual
model with CTC, and measure the improvement honestly on my own recordings.

## Status
| Step | State |
|---|---|
| Collect own test recordings (30 clips, mixed regions, ages, read + extempore) | Done |
| Baseline WER / CER with AI4Bharat IndicConformer on 13 usable clips | Done, see `baseline_scripts/results.csv` |
| Organise recordings into a dataset with metadata | In progress (15 of 30) |
| Fine-tune MMS adapter (CTC) on public data (FLEURS) | Planned, scripts written, not run yet |
| Re-evaluate on the same test set, compare before vs after | Planned |

## Approach
1. **Train** on public Assamese data (e.g. FLEURS). **Test** on my own recordings, which no model has seen.
2. Same text normalization everywhere (`scripts/normalize.py`) so before and after are comparable.
3. Metrics: WER and CER (corpus-level), plus substitution/deletion/insertion rates and per-clip scores.
4. Method: CTC fine-tuning of the MMS adapter. Only the small adapter is trained, which fits free GPUs (Kaggle).

## Repository
```
baseline_scripts/          my own first scripts + data: asr1.py (IndicConformer transcription), wer_overall.py, wer_separate.py, results.csv (clip names anonymised)
scripts/normalize.py       text normalization used for every score
scripts/score_results.py   scores results.csv (raw vs normalized WER/CER, per clip)
scripts/make_dataset.py    raw clips + CSV -> 16 kHz WAV folder + metadata.csv
scripts/eval_asr.py        run a CTC model on the test set and compute WER/CER
scripts/train_adapter.py   fine-tune the MMS adapter on FLEURS (planned run, untested)
docs/DATASET.md            dataset description and limitations
```

## Usage
```
pip install -r requirements.txt
python scripts/make_dataset.py --raw_dir raw_audio --raw_csv raw_metadata.csv --out data --split test
python scripts/eval_asr.py --metadata data/test/metadata.csv --audio_dir data/test --out results_before
```
(run from inside `scripts/` or add it to `PYTHONPATH`)

## Notes and limitations
- Audio is not included here: it contains voices of private individuals, including minors.
- The test set is small (30 clips), so scores are indicative, not precise.
- The training and evaluation scripts were prepared with help from an AI assistant; I am working through them to understand each step.

## Learning plan
Machine learning fundamentals (completed), neural networks and deep learning (in progress), then RNN / LSTM / BiLSTM.
