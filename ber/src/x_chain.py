"""Pipeline plans on the hold node (launch with run.sh so the GPU is set):  python x_chain.py [plan] [first_step]
Steps run in order; x_feats (CPU) runs alongside the CE steps (GPU). Each step logs to work/logs/<step>.log."""
import os, sys, subprocess
from common import WORK

L = f"{WORK}/logs"
PLANS = {
    "v5": [("x_ce_train", ["x_ce.py", "train"], {"CE_N": "12000000"}),
           ("x_ce_score_train", ["x_ce.py", "score", "train"], {}),
           ("x_ce_score_test", ["x_ce.py", "score", "test"], {}),
           ("x_stage_s1", ["x_stage.py", "s1"], {}),
           ("x_stage_cv", ["x_stage.py", "cv"], {}),
           ("x_stage_cv_noce", ["x_stage.py", "cv"], {"NOCE": "1"}),
           ("x_stage_test", ["x_stage.py", "test"], {})],
    # larger cross-encoder: e5-base (MIT, 278M), length-bucketed batches, scores only each record's top-3 pairs
    "base": [("x_ce_base_train", ["x_ce2.py", "train"], {"CE_BASE": "/scratch/scai/mtech/aib262144/models/e5-base", "CE_DIR": f"{WORK}/x/ce_base",
                                                          "CE_N": "8000000", "CE_LR": "3e-5"}),
             ("x_ce_base_score_train", ["x_ce2.py", "score", "train"], {"CE_DIR": f"{WORK}/x/ce_base", "TOPK": "3", "CE_TAG": "_base"}),
             ("x_ce_base_score_test", ["x_ce2.py", "score", "test"], {"CE_DIR": f"{WORK}/x/ce_base", "TOPK": "3", "CE_TAG": "_base"})],
    # v5 + word-edit LLR features (x_llr.py); artefacts tagged _llr
    "llr": [("x_llr_s1", ["x_stage_multi.py", "s1"], {"TAG": "_llr", "LLR": "1"}),
            ("x_llr_cv", ["x_stage_multi.py", "cv"], {"TAG": "_llr", "LLR": "1"}),
            ("x_llr_test", ["x_stage_multi.py", "test"], {"TAG": "_llr", "LLR": "1"})],
    # v6 = v5 + e5-base CE scores as extra features (both CEs); then test q + per-country sweep
    "v6": [("x_v6_s1", ["x_stage_multi.py", "s1"], {"TAG": "_base", "CE_TAGS": ",_base"}),
           ("x_v6_cv", ["x_stage_multi.py", "cv"], {"TAG": "_base", "CE_TAGS": ",_base"}),
           ("x_v6_thr", ["x_thr.py", "_base"], {})],
    # small CE on raw transliterated text (keeps punctuation / suffix spelling) - a second view for the ensemble
    "raw": [("x_ce_raw_train", ["x_ce3.py", "train"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_N": "12000000"}),
            ("x_ce_raw_score_train", ["x_ce3.py", "score", "train"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
            ("x_ce_raw_score_test", ["x_ce3.py", "score", "test"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"})],
    # v7 = all three CEs (small-norm, base-norm top-3, small-raw) as stage-1/2 features
    "v7": [("x_v7_s1", ["x_stage_multi.py", "s1"], {"TAG": "_all", "CE_TAGS": ",_base,_raw"}),
           ("x_v7_cv", ["x_stage_multi.py", "cv"], {"TAG": "_all", "CE_TAGS": ",_base,_raw"}),
           ("x_v7_thr", ["x_thr.py", "_all"], {})],
}
PLANS["rawv7"] = PLANS["raw"] + PLANS["v7"]   # raw CE, then the three-CE stages
# v8: country-agnostic and compliant. Mined dictionaries (no hand-written country lists), learned shortlist,
# small + raw-text cross-encoders (e5-base dropped: +0.00003 in v6), words.py features, copy-free stage 2 scored
# per stage-1 model, one threshold. Needs cand_*.parquet + e5_ft (train_embed.py, retrieve.py) first.
_V8 = {"SHORTLIST": "learned:0.001"}
_V8S = {**_V8, "TAG": "_v8", "CE_TAGS": ",_raw", "WORDS": "1"}
PLANS["v8"] = [("v8_mine_dicts", ["mine_dicts.py"], _V8),
               ("v8_prep_norm", ["prep_norm.py"], _V8),
               ("v8_shortlist", ["shortlist.py"], _V8),
               ("v8_stage1_v4", ["stage1_cv.py"], _V8),
               ("v8_x_feats_train", ["x_feats.py", "train"], _V8),
               ("v8_x_feats_test", ["x_feats.py", "test"], _V8),
               ("v8_words", ["words.py"], _V8),
               ("v8_ce_train", ["x_ce.py", "train"], {**_V8, "CE_N": "12000000"}),
               ("v8_ce_score_train", ["x_ce.py", "score", "train"], _V8),
               ("v8_ce_score_test", ["x_ce.py", "score", "test"], _V8),
               ("v8_ce_raw_train", ["x_ce3.py", "train"], {**_V8, "CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_N": "12000000"}),
               ("v8_ce_raw_score_train", ["x_ce3.py", "score", "train"], {**_V8, "CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
               ("v8_ce_raw_score_test", ["x_ce3.py", "score", "test"], {**_V8, "CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
               ("v8_stage1", ["x_stage_multi.py", "s1"], _V8S),
               ("v8_stage2_compare", ["x_nocopy.py"], _V8S),
               ("v8_stage2_test", ["x_nocopy.py", "test"], _V8S),
               ("v8_final", ["x_final.py", f"{os.path.dirname(WORK)}/output_v8", "_v8w", "0.70"], _V8S)]
ARGS = sys.argv[1:]
PLAN = ARGS.pop(0) if ARGS and ARGS[0] in PLANS else "v5"   # python x_chain.py [plan] [first_step]
STEPS = PLANS[PLAN]


def start(name, args, env):
    f = open(f"{L}/{name}.log", "w")
    print("start", name, flush=True)
    return subprocess.Popen([sys.executable, "-u"] + args, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **env})


first = ARGS[0] if ARGS else STEPS[0][0]
feats = None
if PLAN == "v5" and not all(os.path.exists(f"{WORK}/x/extra_{s}.parquet") for s in ("train", "test")):
    feats = subprocess.Popen(f"{sys.executable} -u x_feats.py train > {L}/x_feats_train.log 2>&1 && "
                             f"{sys.executable} -u x_feats.py test > {L}/x_feats_test.log 2>&1", shell=True)
go = False
for name, args, env in STEPS:
    go = go or name == first
    if not go:
        continue
    if name == "x_stage_s1" and feats is not None and feats.wait():
        sys.exit("x_feats failed")
    if start(name, args, env).wait():
        sys.exit(f"{name} failed")
    print("done", name, flush=True)
