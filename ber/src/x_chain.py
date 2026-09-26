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
# v8 = scalable blocking + no hand-written tables. v8a: FAISS HNSW retrieval, the learned per-country lexicon and the
# calibrated shortlist model (read its recall / size report, then pick SHORTLIST for v8b).
PLANS["v8a"] = [("v8_retrieve_train", ["retrieve.py", "train"], {"ANN": "hnsw"}),
                ("v8_retrieve_test", ["retrieve.py", "test"], {"ANN": "hnsw"}),
                ("v8_lexicon", ["lexicon.py"], {}),
                ("v8_shortlist", ["shortlist.py"], {})]
# v8b: features on the shortlisted pairs, the normalised-text CE retrained (the raw-text CE only re-scores; the e5-base
# CE is dropped: +0.00003 in v6), stage 1 with both CEs, stage 2 without distractor copies -> x/test_q_v8w
_SL = os.environ.get("SHORTLIST", "model:0.005")
PLANS["v8b"] = [
    ("v8_features", ["-c", "from match import features; features('train'); features('test')"], {"SHORTLIST": _SL}),
    ("v8_extra", ["-c", "import x_feats; x_feats.main('train'); x_feats.main('test')"], {}),
    ("v8_stage1v4", ["stage1_cv.py"], {}),              # v4 stage-1 p, used by the CE sanity checks
    ("v8_ce_train", ["x_ce.py", "train"], {"CE_N": "12000000"}),
    ("v8_ce_score_train", ["x_ce.py", "score", "train"], {}),
    ("v8_ce_score_test", ["x_ce.py", "score", "test"], {}),
    # the raw-text CE of the v8 run was the v7 one (trained on the exact-search top-5); this retrains it from scratch
    ("v8_raw_train", ["x_ce3.py", "train"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_N": "12000000"}),
    ("v8_raw_score_train", ["x_ce3.py", "score", "train"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
    ("v8_raw_score_test", ["x_ce3.py", "score", "test"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
    ("v8_s1", ["x_stage_multi.py", "s1"], {"TAG": "_v8", "CE_TAGS": ",_raw"}),
    ("v8_s2", ["x_nocopy.py", "test"], {"TAG": "_v8", "CE_TAGS": ",_raw"})]
# v9 experiments on the v8 features: error breakdown, shortlist with text similarities, and stages with 3 / 6 stage-1
# fold groups and a 5-seed stage 2 (compare the copy-free validation printed by x_nocopy with v8's)
_V9 = {"CE_TAGS": ",_raw"}
PLANS["v9"] = [("v9_verr", ["x_verr.py", "_v8"], {}),
               ("v9_shortlist2", ["x_shortlist2.py"], {}),
               ("v9_s1_k3", ["x_stage_multi.py", "s1"], {**_V9, "TAG": "_v9k3", "S1K": "3"}),
               ("v9_s2_k3", ["x_nocopy.py", "test"], {**_V9, "TAG": "_v9k3", "BAG": "5"}),
               ("v9_s1_k6", ["x_stage_multi.py", "s1"], {**_V9, "TAG": "_v9k6", "S1K": "6"}),
               ("v9_s2_k6", ["x_nocopy.py", "test"], {**_V9, "TAG": "_v9k6", "BAG": "5"})]
# v9m = v9fS end to end: Sahaj's v9 (HNSW retrieval, learned lexicon, name-edit features, text-aware shortlist at
# 0.001, three stage-1 models) merged with the v8-generalize branch (distractor-word model words.py, stage 2 scored
# once per stage-1 model, self-training of stage 1 for test countries without labels, LightGBM / XGBoost judge).
_SL9 = {"SHORTLIST": os.environ.get("SHORTLIST", "text:0.001")}   # Sahaj v9: reproduces its 12,760,925 test pairs exactly
_S9 = {"TAG": "_v9m", "CE_TAGS": ",_raw", "WORDS": "1", "S1K": "3", "SELFTRAIN": "1"}   # = v9fS
_OUT9 = os.path.dirname(WORK)
PLANS["v9m"] = PLANS["v8a"] + [
    ("v9m_features", ["-c", "from match import features; features('train'); features('test')"], _SL9),
    ("v9m_extra", ["-c", "import x_feats; x_feats.main('train'); x_feats.main('test')"], {}),
    ("v9m_stage1v4", ["stage1_cv.py"], {}),
    ("v9m_words", ["words.py"], {}),
    ("v9m_ce_train", ["x_ce.py", "train"], {"CE_N": "12000000"}),
    ("v9m_ce_score_train", ["x_ce.py", "score", "train"], {}),
    ("v9m_ce_score_test", ["x_ce.py", "score", "test"], {}),
    ("v9m_raw_train", ["x_ce3.py", "train"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_N": "12000000"}),
    ("v9m_raw_score_train", ["x_ce3.py", "score", "train"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
    ("v9m_raw_score_test", ["x_ce3.py", "score", "test"], {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
    ("v9m_s1", ["x_stage_multi.py", "s1"], _S9),
    ("v9m_s2", ["x_nocopy.py", "test"], _S9),
    ("v9m_final", ["x_final.py", f"{_OUT9}/output_v9mS", "_v9mw", "0.70"], _S9),
    ("v9m_judge", ["x_judge.py"], _S9),
    ("v9m_judge_test", ["x_judge.py", "test"], {**_S9, "JUDGE": "best"}),
    ("v9m_final_judge", ["x_final.py", f"{_OUT9}/output_v9mSJ", "_v9mJ", "auto"], _S9)]
# v9mX = v9fX, after v9m's features and cross-encoders exist: stages without stage-1 self-training (v9f), then both
# cross-encoders self-trained on that run's confident predictions for the test countries without training labels
# (x_ce_st.py), then stages 1-2 again with the re-scored test pairs (CE_TEST_SUFFIX=st) -> ../output_v9fX
_F9 = {"TAG": "_v9f", "CE_TAGS": ",_raw", "WORDS": "1", "S1K": "3"}
_X9 = {**_F9, "TAG": "_v9fX", "CE_TEST_SUFFIX": "st"}
PLANS["v9mX"] = [
    ("v9f_s1", ["x_stage_multi.py", "s1"], _F9),
    ("v9f_s2", ["x_nocopy.py", "test"], _F9),
    ("v9fX_ce_st", ["x_ce_st.py"], {"TAG": "_v9f", "CE_TEXT": "norm", "CE_DIR": f"{WORK}/x/ce", "CE_TAG": ""}),
    ("v9fX_ce_raw_st", ["x_ce_st.py"], {"TAG": "_v9f", "CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}),
    ("v9fX_s1", ["x_stage_multi.py", "s1"], _X9),
    ("v9fX_s2", ["x_nocopy.py", "test"], _X9),
    ("v9fX_final", ["x_final.py", f"{os.path.dirname(WORK)}/output_v9fX", "_v9fXw", "0.70"], _X9)]
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
