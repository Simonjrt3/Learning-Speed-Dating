"""Collect every number the Model 3 write-up uses into out/numbers.json.

Confidence intervals resample whole waves (21 events) with replacement, so they respect
the fact that people, and both rows of a date, sit inside one wave.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json; sys.path.insert(0, str(HERE))
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import lib

O = Path(str(OUT))
df = lib.load()
waves = df["wave"].astype(float).to_numpy()
order = np.concatenate([te for _, te in lib.wave_folds(df)])   # row order of pooled wave preds
w_pooled = waves[order]
uw = np.unique(waves)
rng = np.random.default_rng(0)
BOOT = [rng.choice(uw, size=len(uw), replace=True) for _ in range(2000)]


def boot_auc(p, y, p2=None):
    vals = []
    for s in BOOT:
        idx = np.concatenate([np.flatnonzero(w_pooled == w) for w in s])
        a = roc_auc_score(y[idx], p[idx])
        if p2 is not None:
            a -= roc_auc_score(y[idx], p2[idx])
        vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load_pred(path):
    a = np.load(path)
    return a[0], a[1].astype(int)


N = {}
# ---- ablation steps
for step in ["original", "no_leak", "recode", "pc_parents", "bdeu"]:
    for split in ["their", "wave"]:
        d = json.load(open(O / "ablation" / f"{step}_{split}.json"))
        f = d["folds"]
        N[f"{step}_{split}"] = dict(
            auc=d["pooled_auc"] if split == "wave" else f[0]["auc"],
            log_loss=d["pooled_log_loss"] if split == "wave" else f[0]["log_loss"],
            f1=float(np.mean([r["f1"] for r in f])), fold_auc=[r["auc"] for r in f],
            unseen=float(np.mean([r["unseen"] for r in f])), frac01=float(np.mean([r["frac01"] for r in f])),
            n_configs=[r["n_configs"] for r in f], parents=[r["parents"] for r in f],
            match_children=[[e[1] for e in r["edges"] if e[0] == "match"] for r in f],
            dropped=[r.get("dropped", 0) for r in f], threshold=[r["threshold"] for r in f],
            sel=[r["sel"] for r in f])
# ---- repaired structures
for tag in ["nb_k10_e5", "tan_k10_e5", "pc_nb_k10_e5", "nb_k5_e5", "nb_k20_e5", "nb_k10_e1", "nb_k10_e20"]:
    for split in ["their", "wave"]:
        f = O / "variants" / f"{tag}_{split}.json"
        if f.exists():
            d = json.load(open(f))
            N[f"{tag}_{split}"] = dict(auc=d["pooled_auc"] if split == "wave" else d["fold_auc"][0],
                                       log_loss=d["pooled_log_loss"] if split == "wave" else d["folds"][0]["log_loss"],
                                       f1=float(np.mean(d["fold_f1"])), fold_auc=d["fold_auc"],
                                       sel=[r["sel"] for r in d["folds"]])
ref = json.load(open(O / "ablation" / "references.json"))
for k, v in ref.items():
    N[k] = v
# ---- wave-bootstrap intervals
p_nb, y = load_pred(O / "variants" / "nb_k10_e5_wave_p.npy")
p_or, y2 = load_pred(O / "ablation" / "original_wave_p.npy")
p_bd, y3 = load_pred(O / "ablation" / "bdeu_wave_p.npy")
p_lr, y4 = load_pred(O / "ablation" / "ref_lr_all_wave_p.npy")
assert (y == y2).all() and (y == y3).all() and (y == y4).all()
assert (y == lib.P._encode_target(df["match"]).to_numpy()[order]).all()
N["ci"] = dict(nb_wave=boot_auc(p_nb, y), original_wave=boot_auc(p_or, y), bdeu_wave=boot_auc(p_bd, y),
               lr_wave=boot_auc(p_lr, y), nb_minus_original=boot_auc(p_nb, y, p_or),
               nb_minus_lr=boot_auc(p_nb, y, p_lr))
# ---- data facts
tr, te = lib.their_split(df)
iid, pid = df["_iid"].to_numpy(), df["_pid"].to_numpy()
pairs = set(zip(iid[tr], pid[tr]))
N["facts"] = dict(
    n=len(df), n_train=len(tr), n_test=len(te), match_rate=float(lib.P._encode_target(df["match"]).mean()),
    field_values=int(df["field"].nunique()), field_cd_values=int(df["_field_cd"].nunique()),
    enii_missing=float(df["expected_num_interested_in_me"].isna().mean()),
    test_participant_in_train=float(np.mean([p in set(iid[tr]) for p in iid[te]])),
    test_mirror_in_train=float(np.mean([(b, a) in pairs for a, b in zip(iid[te], pid[te])])),
    sincere_o_rank=N["original_their"]["sel"][0].index("sincere_o") + 1,
    n_waves=int(len(uw)), wave_match_min=float(df.assign(m=lib.P._encode_target(df["match"])).groupby("wave").m.mean().min()),
    wave_match_max=float(df.assign(m=lib.P._encode_target(df["match"])).groupby("wave").m.mean().max()),
    wave_test_sizes=[int(len(b)) for _, b in lib.wave_folds(df)])
hs = O / "hashseed.json"
if hs.exists():
    N["hashseed"] = json.load(open(hs))
json.dump(N, open(O / "numbers.json", "w"), indent=1)
for k, v in N.items():
    if isinstance(v, dict) and "auc" in v:
        print(f"{k:22s} auc {v['auc']:.4f}  logloss {v['log_loss']:.4f}")
print(json.dumps(N["ci"], indent=0)); print(json.dumps(N["facts"]))
