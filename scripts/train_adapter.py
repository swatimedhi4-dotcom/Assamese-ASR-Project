"""Fine-tune the MMS-1B adapter (CTC) for Assamese on FLEURS (as_in).
Only the small adapter + output layer are trained; the 1B base model stays frozen.
Run on a GPU (Kaggle / Colab):
  pip install transformers datasets jiwer librosa soundfile accelerate safetensors
  python train_adapter.py --out mms_asm_fleurs
Quick smoke test first:  python train_adapter.py --out tmp --max_train 64 --epochs 1
NOTE: if load_dataset("google/fleurs") fails on a new `datasets` version, run
  pip install "datasets<4"   and pass --trust_remote_code
"""
import argparse, json, os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
                          Wav2Vec2ForCTC, TrainingArguments, Trainer)
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
from safetensors.torch import save_file
from normalize import normalize

LANG = "asm"
BASE = "facebook/mms-1b-all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    a = ap.parse_args()

    kw = {"trust_remote_code": True} if a.trust_remote_code else {}
    train = load_dataset("google/fleurs", "as_in", split="train", **kw)
    dev = load_dataset("google/fleurs", "as_in", split="validation", **kw)
    if a.max_train:
        train = train.select(range(min(a.max_train, len(train))))
        dev = dev.select(range(min(a.max_train, len(dev))))
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    dev = dev.cast_column("audio", Audio(sampling_rate=16000))

    # same normalizer as everywhere else
    def add_text(b):
        b["target_text"] = normalize(b["raw_transcription"])
        return b
    train = train.map(add_text); dev = dev.map(add_text)

    # character vocabulary from the training text
    chars = set("".join(train["target_text"]))
    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = vocab.pop(" ") if " " in vocab else len(vocab)
    vocab["[UNK]"] = len(vocab); vocab["[PAD]"] = len(vocab)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({LANG: vocab}, f, ensure_ascii=False)

    tok = Wav2Vec2CTCTokenizer(os.path.join(a.out, "vocab.json"), unk_token="[UNK]",
                               pad_token="[PAD]", word_delimiter_token="|", target_lang=LANG)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)

    def prep(b):
        b["input_values"] = proc(b["audio"]["array"], sampling_rate=16000).input_values[0]
        b["input_length"] = len(b["input_values"])
        b["labels"] = proc(text=b["target_text"]).input_ids
        return b
    train = train.map(prep, remove_columns=train.column_names)
    dev = dev.map(prep, remove_columns=dev.column_names)

    @dataclass
    class Collator:
        processor: Wav2Vec2Processor
        def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]):
            inp = [{"input_values": f["input_values"]} for f in feats]
            lab = [{"input_ids": f["labels"]} for f in feats]
            batch = self.processor.pad(inp, padding=True, return_tensors="pt")
            labels = self.processor.pad(labels=lab, padding=True, return_tensors="pt")
            batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
            return batch

    def compute_metrics(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = proc.tokenizer.pad_token_id
        hyp = [normalize(t) for t in proc.batch_decode(ids)]
        ref = [normalize(t) for t in proc.batch_decode(pred.label_ids, group_tokens=False)]
        pairs = [(r, h) for r, h in zip(ref, hyp) if r]
        r, h = zip(*pairs)
        return {"wer": jiwer.wer(list(r), list(h)), "cer": jiwer.cer(list(r), list(h))}

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
        ctc_loss_reduction="mean", pad_token_id=proc.tokenizer.pad_token_id,
        vocab_size=len(proc.tokenizer), ignore_mismatched_sizes=True)
    model.init_adapter_layers()
    model.freeze_base_model()
    adapters = model._get_adapters()
    for p in adapters.values():
        p.requires_grad = True

    args = TrainingArguments(
        output_dir=a.out, group_by_length=True, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.accum, eval_strategy="steps", num_train_epochs=a.epochs,
        gradient_checkpointing=True, fp16=torch.cuda.is_available(), save_steps=200,
        eval_steps=100, logging_steps=50, learning_rate=a.lr, warmup_steps=100,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="wer",
        greater_is_better=False, report_to="none")
    trainer = Trainer(model=model, data_collator=Collator(proc), args=args,
                      compute_metrics=compute_metrics, train_dataset=train, eval_dataset=dev,
                      processing_class=proc.feature_extractor)
    trainer.train()

    trainer.save_model(a.out)                      # full model incl. trained adapter
    proc.save_pretrained(a.out)
    save_file(model._get_adapters(), os.path.join(a.out, WAV2VEC2_ADAPTER_SAFE_FILE.format(LANG)))
    print("saved to", a.out)


if __name__ == "__main__":
    main()
