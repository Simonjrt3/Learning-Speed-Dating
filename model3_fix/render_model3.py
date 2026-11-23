"""Fill section_model3_template.tex from out/numbers.json (no hand-typed numbers)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json, re; sys.path.insert(0, str(HERE))
from dataclasses import replace
from pathlib import Path
import numpy as np
import lib

O = Path(str(OUT))
N = json.load(open(O / "numbers.json"))
F = N["facts"]
a3 = lambda x: f"{x:.3f}"
a2 = lambda x: f"{x:.2f}"
pct = lambda x: f"{100 * x:.0f}\\%"
sg = lambda f, x: f(x) if x >= 0 else "$-$" + f(-x)
ci = lambda c, f=a3: f"{sg(f, c[0])}--{sg(f, c[1])}"
p_or, y_or = np.load(O / "ablation" / "original_wave_p.npy")
p_rt, y_rt = np.load(O / "ablation" / "original_their_p.npy")
nb_sel = N["nb_k10_e5_wave"]["sel"]
always = [v for v in nb_sel[0] if all(v in s for s in nb_sel)]
kids = [len(set(c) | set(p)) for c, p in zip(N["bdeu_wave"]["match_children"], N["bdeu_wave"]["parents"])]
df = lib.load()
n_lr = len(lib.candidates(df, replace(lib.ORIGINAL, leak=False, recode=True)))
fold = N["nb_k10_e5_wave"]["fold_auc"]
tex = lambda v: v.replace("_", "\\_")
V = dict(
    orig_their_auc=a3(N["original_their"]["auc"]), orig_their_f1=a3(N["original_their"]["f1"]),
    orig_their_ll=a2(N["original_their"]["log_loss"]),
    sincere_rank={1: "first", 2: "second", 3: "third"}[F["sincere_o_rank"]],
    noleak_their_auc=a3(N["no_leak_their"]["auc"]), noleak_their_f1=a3(N["no_leak_their"]["f1"]),
    field_values=f"{F['field_values']}", enii_missing=pct(F["enii_missing"]),
    orig_configs=f"{N['original_their']['n_configs'][0]:,}".replace(",", "{,}"),
    n_train=f"{F['n_train']:,}".replace(",", "{,}"), n_test=f"{F['n_test']:,}".replace(",", "{,}"),
    n_rows=f"{F['n']:,}".replace(",", "{,}"),
    orig_frac01=pct(N["original_their"]["frac01"]), p0_obs=pct(y_rt[p_rt == 0].mean()),
    const_their_ll=a3(N["constant_their"]["log_loss"]), const_wave_ll=a3(N["constant_wave"]["log_loss"]),
    mirror=pct(F["test_mirror_in_train"]), orig_seen=pct(1 - N["original_their"]["unseen"]),
    orig_wave_auc=a3(N["original_wave"]["auc"]), orig_wave_ci=ci(N["ci"]["original_wave"], a2),
    bdeu_wave_auc=a3(N["bdeu_wave"]["auc"]), bdeu_wave_ll=a3(N["bdeu_wave"]["log_loss"]),
    pc_kids_range=f"between {min(kids)} and {max(kids)}",
    nb_wave_auc=a3(N["nb_k10_e5_wave"]["auc"]), nb_wave_ci=ci(N["ci"]["nb_wave"], a2),
    nb_fold_min=a2(min(fold)), nb_fold_max=a2(max(fold)), n_lr_vars=str(n_lr),
    lr_wave_auc=a3(N["lr_all_wave"]["auc"]), tan_wave_auc=a3(N["tan_k10_e5_wave"]["auc"]),
    pcnb_wave_auc=a3(N["pc_nb_k10_e5_wave"]["auc"]),
    nb_always=", ".join(f"\\texttt{{{tex(v)}}}" for v in always),
    base_rate=pct(F["match_rate"]),
    wave_p05=pct(np.mean(p_or == 0.5)), wave_p0_obs=pct(y_or[p_or == 0].mean()),
    wave_p1_obs=pct(y_or[p_or == 1].mean()),
    orig_their_auc_short=a2(N["original_their"]["auc"]), nb_their_auc_short=a2(N["nb_k10_e5_their"]["auc"]),
    orig_wave_auc_short=a2(N["original_wave"]["auc"]), nb_wave_auc_short=a2(N["nb_k10_e5_wave"]["auc"]),
    nb_minus_orig=a2(N["nb_k10_e5_wave"]["auc"] - N["original_wave"]["auc"]),
    nb_minus_orig_ci=ci(N["ci"]["nb_minus_original"], a2),
)
MI = json.load(open(O / "mi_field.json"))
V["field_null"] = f"{1000 * MI['field_null_mean']:.0f}"
V["field_mi"] = f"{1000 * MI['field']['mi']:.0f}"
assert MI["field_null_mean"] > max(v["mi"] for k, v in MI.items() if isinstance(v, dict) and k != "field")
WORDS = {i: w for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
crash = sum(d > 0 for d in N["original_wave"]["dropped"])
hs = N["hashseed"]
HF = json.load(open(O / "hashseed_fixed.json"))["summary"]
all_or_some = (f"all {WORDS[HF['seeds']]}" if HF["crashed"] == HF["seeds"]
               else f"{WORDS[HF['crashed']]} of the {WORDS[HF['seeds']]}")
V["fragile"] = (" Forcing five parents onto \\texttt{match} can close a directed loop, which pgmpy refuses, "
                "so the script stops. This already happens on the team's split once \\texttt{sincere\\_o} is "
                "excluded (PC's graph has a path from \\texttt{match} to \\texttt{pref\\_o\\_shared\\_interests}; "
                f"{all_or_some} hash seeds we tried), and on {WORDS[crash]} of the five new-event training sets "
                "with the committed variables. We drop one learned edge to continue. The committed version is "
                f"stable on the team's split: {WORDS[hs['seeds']]} hash seeds give the same network.")
for key, step in [("orig", "original"), ("noleak", "no_leak"), ("recode", "recode"), ("pcpar", "pc_parents"),
                  ("bdeu", "bdeu"), ("nb", "nb_k10_e5"), ("tan", "tan_k10_e5"), ("pcnb", "pc_nb_k10_e5"),
                  ("lr", "lr_all")]:
    for sp in ["their", "wave"]:
        V.setdefault(f"{key}_{sp}_auc", a3(N[f"{step}_{sp}"]["auc"]))
        V.setdefault(f"{key}_{sp}_ll", a3(N[f"{step}_{sp}"]["log_loss"]))
V["orig_their_ll"] = a2(N["original_their"]["log_loss"])
V["orig_their_ll3"] = a3(N["original_their"]["log_loss"])
V["lookup_only"] = pct(json.load(open(O / "lookup.json"))["same_only"])
import pgmpy, sklearn
V.update(pgmpy=pgmpy.__version__, sklearn=sklearn.__version__,
         orig_edges=str(len(json.load(open(O / "orig_f10.json"))["edges"])),
         field_bias=f"{1000 * (F['field_values'] - 1) / (2 * F['n_train']):.0f}",
         wave_sizes="{:,}--{:,}".format(min(F["wave_test_sizes"]), max(F["wave_test_sizes"])).replace(",", "{,}").replace("--", "--"),
         wave_min=pct(F["wave_match_min"]), wave_max=pct(F["wave_match_max"]), hs_seeds=WORDS[hs["seeds"]],
         nb_folds=", ".join(a3(x) for x in fold),
         nb_k5=a3(N["nb_k5_e5_wave"]["auc"]), nb_k20=a3(N["nb_k20_e5_wave"]["auc"]),
         nb_e1=a3(N["nb_k10_e1_wave"]["auc"]), nb_e20=a3(N["nb_k10_e20_wave"]["auc"]),
         kids=", ".join(map(str, kids)),
         tan_their=a3(N["tan_k10_e5_their"]["auc"]), tan_wave=a3(N["tan_k10_e5_wave"]["auc"]),
         pcnb_their=a3(N["pc_nb_k10_e5_their"]["auc"]), pcnb_wave=a3(N["pc_nb_k10_e5_wave"]["auc"]),
         nb_minus_lr=sg(a3, N["nb_k10_e5_wave"]["auc"] - N["lr_all_wave"]["auc"]),
         nb_minus_lr_ci=ci(N["ci"]["nb_minus_lr"]))
for src, dst in [("section_model3_template.tex", sys.argv[1]), ("appendix_model3_template.tex", sys.argv[2])]:
    t = Path(str(HERE) + "/" + src).read_text()
    missing = sorted(set(re.findall(r"<<(\w+)>>", t)) - set(V))
    assert not missing, (src, missing)
    Path(dst).write_text(re.sub(r"<<(\w+)>>", lambda m: V[m.group(1)], t))
json.dump(V, open(O / "rendered_values.json", "w"), indent=1)
print("rendered", len(V), "values; always-selected:", always, "kids", kids, "n_lr", n_lr)
