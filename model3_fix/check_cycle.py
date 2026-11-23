"""Does pgmpy refuse the committed script's forced edges on new-event fold 2 (as their code would)?"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, warnings; sys.path.insert(0, str(HERE))
from dataclasses import replace
import lib
from pgmpy.models import DiscreteBayesianNetwork
df = lib.load()
cfg = lib.ORIGINAL
for k in [2, 4]:
    tr_idx, te_idx = lib.wave_folds(df)[k]
    cols = lib.candidates(df, cfg); y = lib.P._encode_target(df["match"])
    tr, te = lib.encode(cfg, df.iloc[tr_idx][cols], df.iloc[te_idx][cols])
    sel = lib.mi_rank(tr, y.iloc[tr_idx], False)[:10]
    trn = tr[sel].assign(match=y.iloc[tr_idx].to_numpy())
    edges = lib.learn_structure(trn, replace(cfg, forced=False), [])
    m = DiscreteBayesianNetwork(); m.add_nodes_from(trn.columns); m.add_edges_from(edges)
    try:
        for p in sel[:5]:
            m.add_edge(p, "match")      # exactly what learn_and_fit_network does
        print("fold", k, "no error")
    except ValueError as e:
        print("fold", k, "pgmpy refuses:", str(e)[:90])
