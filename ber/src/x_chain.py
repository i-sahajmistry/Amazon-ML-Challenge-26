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
# v9b = v9 build: the shortlist that also sees text similarities, then v8's CEs (trained on the retrieved top-5, so
# they only re-score), stage 1 with 3 fold groups and a 5-seed stage 2 -> x/test_q_v9w. Move the v8 feats2 / extra /
# ce / oof caches aside first (features() reuses an existing feats2). stage1v4 only refreshes the v4 p, so it runs last.
_SL9 = os.environ.get("SHORTLIST", "text:0.001")
_RAW = {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_raw", "CE_TAG": "_raw"}
PLANS["v9b"] = [
    ("v9_features", ["-c", "from match import features; features('train'); features('test')"], {"SHORTLIST": _SL9}),
    ("v9_extra", ["-c", "import x_feats; x_feats.main('train'); x_feats.main('test')"], {}),
    ("v9_ce_score_train", ["x_ce.py", "score", "train"], {}),
    ("v9_ce_score_test", ["x_ce.py", "score", "test"], {}),
    ("v9_raw_score_train", ["x_ce3.py", "score", "train"], _RAW),
    ("v9_raw_score_test", ["x_ce3.py", "score", "test"], _RAW),
    ("v9_s1", ["x_stage_multi.py", "s1"], {**_V9, "TAG": "_v9", "S1K": "3"}),
    ("v9_s2", ["x_nocopy.py", "test"], {**_V9, "TAG": "_v9", "BAG": "5"}),
    ("v9_stage1v4", ["stage1_cv.py"], {})]
# v9s = v9 with both CEs continued on confident test decisions of the countries without train labels (self-training,
# x_ce3.py train with SELF=x/test_q_v8w.parquet -> x/ce_ns, x/ce_rs), re-scored and restacked -> x/test_q_v9sw
_RS = {"CE_TEXT": "raw", "CE_DIR": f"{WORK}/x/ce_rs", "CE_TAG": "_rs"}
_NS = {"CE_DIR": f"{WORK}/x/ce_ns", "CE_TAG": "_ns"}
_ST = {"CE_TAGS": "_ns,_rs", "TAG": "_v9s"}
_SELF = {"SELF": f"{WORK}/x/test_q_v8w.parquet", "CE_N": "3000000", "SELF_N": "4000000", "CE_LR": "2e-5"}
PLANS["v9s"] = [("v9s_rs_fit", ["x_ce3.py", "train"], {**_RS, "CE_INIT": f"{WORK}/x/ce_raw", **_SELF}),
                ("v9s_ns_fit", ["x_ce3.py", "train"], {**_NS, "CE_INIT": f"{WORK}/x/ce", **_SELF}),
                ("v9s_rs_train", ["x_ce3.py", "score", "train"], _RS),
                ("v9s_rs_test", ["x_ce3.py", "score", "test"], _RS),
                ("v9s_ns_train", ["x_ce3.py", "score", "train"], _NS),
                ("v9s_ns_test", ["x_ce3.py", "score", "test"], _NS),
                ("v9s_s1", ["x_stage_multi.py", "s1"], {**_ST, "S1K": "3"}),
                ("v9s_s2", ["x_nocopy.py", "test"], {**_ST, "BAG": "5"}),
                ("v9s_s2p", ["x_nocopy.py", "test"], {**_ST, "BAG": "5", "PEERS": "1"})]   # -> x/test_q_v9spw
# v9p = v9 + house-number peer context in stage 2 -> x/test_q_v9pw
PLANS["v9p"] = [("v9_s2p", ["x_nocopy.py", "test"], {**_V9, "TAG": "_v9", "BAG": "5", "PEERS": "1"})]
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
