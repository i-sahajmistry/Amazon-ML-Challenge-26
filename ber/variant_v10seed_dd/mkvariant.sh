#!/bin/bash
# sandbox for variant.sh: code of the clean branch, the original run's inputs symlinked read-only
set -e
R=$HOME/scratch/amlc_variant; O=$HOME/scratch/AmazonMLChallenge/work
[ -e $R ] && { echo "$R exists"; exit 1; }
mkdir -p $R/work/x $R/work/logs && cd $R && tar xf $HOME/scratch/final_a707292.tar
ln -s $HOME/scratch/AmazonMLChallenge/student_resource $R/student_resource
for f in pq e5_ft cand_train.parquet cand_test.parquet feats2_train.parquet feats2_test.parquet lexicon.json imitated_train.npy; do
    ln -s $O/$f $R/work/$f; done
for f in $O/x/tok_*.npz; do ln -s $f $R/work/x/$(basename $f); done
for f in ce_r10 ce_n10 test_q_v10pw.parquet val_q_v10pw.parquet oof5_train_v10.parquet p5_test_v10.parquet \
         llm_val_v10p.parquet llm_test_v10p.parquet; do ln -s $O/x/$f $R/work/x/$f; done
ln -s p5_test_v10.parquet $R/work/x/p5_test_v10pw.parquet
ln -s $O/archive_extra_v10/extra_train.parquet $R/work/x/extra_train.parquet
ln -s $O/archive_extra_v10/extra_test.parquet $R/work/x/extra_test.parquet
ls -la $R/work $R/work/x | head -40
