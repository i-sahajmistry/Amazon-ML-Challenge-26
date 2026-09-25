"""Fine-tune multilingual-e5-small as a bi-encoder: (S2/S3 record) -> (its S1 entity), in-batch negatives.
Trained only on S1 folds 0-3 so folds 4-9 stay unseen for the GBM / validation."""
import os, sys, numpy as np, pandas as pd, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers
from common import load, s1_fold, embed_text, WORK

BASE = os.environ.get("E5", "intfloat/multilingual-e5-small")  # MIT licence; set E5=/local/path on offline nodes
OUT = f"{WORK}/e5_ft"
N_PER_S1 = int(os.environ.get("N_PER_S1", 2))  # pairs per S1 (distinct matches); each epoch-pass keeps S1 unique per batch mostly

s1 = load("train", 1); gt = load("train", "ground_truth")
other = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
gt = gt[s1_fold(gt.source1_entity_id.tolist()) < 4]
gt = gt[gt.matched_entity_ids != ""]
pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
pairs = pairs.sample(frac=1, random_state=0).groupby("source1_entity_id").head(N_PER_S1)
# order so that the k-th pass over S1 comes after the (k-1)-th -> few same-S1 collisions in a batch
pairs["k"] = pairs.groupby("source1_entity_id").cumcount()
pairs = pairs.sample(frac=1, random_state=1).sort_values("k", kind="stable")
s1t = pd.Series(embed_text(s1), index=s1.entity_id)
ot = pd.Series(embed_text(other), index=other.entity_id)
ds = Dataset.from_dict({"anchor": ot.loc[pairs.m].tolist(), "positive": s1t.loc[pairs.source1_entity_id].tolist()})
print("pairs", len(ds), ds[0], flush=True)

model = SentenceTransformer(BASE, device="cuda")
model.max_seq_length = 64
args = SentenceTransformerTrainingArguments(
    output_dir=f"{WORK}/e5_ckpt", num_train_epochs=1, per_device_train_batch_size=1024, learning_rate=1e-4,
    warmup_ratio=0.05, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=100,
    save_strategy="no", report_to=[], dataloader_num_workers=4, seed=0)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                     loss=losses.MultipleNegativesSymmetricRankingLoss(model, scale=30.0))
trainer.train()
model.save(OUT)
print("saved", OUT)
