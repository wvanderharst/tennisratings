"""Fast grid for the serve/return engine off cached setup. usage: fast_grid.py tour '<json list of configs>' [tag]"""
import paths as P
import sys, json, pickle, datetime, time
import numpy as np
from scipy.optimize import minimize
import confounder_harness as H
TOUR = sys.argv[1]
if TOUR == "wta":
    WD = P.ROOT + "/wta_data"
    H.birth = json.load(open(f"{WD}/wta_birth_ordinal_v2.json"))
import importlib, os
E = importlib.import_module(os.environ.get("ENGINE", "sr_engine2"))
C = np.load(f"cache_{TOUR}.npz"); rows = pickle.load(open(f"cache_{TOUR}_rows.pkl", "rb"))
corr, base, B0, when = C["corr"], C["base"], C["B0"], C["when"]; fm, vm, tr, te = C["fm"], C["vm"], C["tr"], C["te"]
ALL = when >= datetime.date(2005, 1, 1).toordinal()
cfe = json.load(open(P.OUT + "/confounder_fit_v21e.json"))
L = lambda z: np.logaddexp(0, -z)
def fitz(F, m):
    def g(w):
        zz = F[m] @ w; p = 1 / (1 + np.exp(-zz))
        return np.logaddexp(0, -zz).sum() + 5e-5 * (w[1:] @ w[1:]), -(F[m].T @ (1 - p)) + np.r_[0, 1e-4 * w[1:]]
    w0 = np.zeros(F.shape[1]); w0[0] = 1; return minimize(g, w0, jac=True, method="L-BFGS-B").x
def score(F):
    wv = fitz(F, fm); wt = fitz(F, tr); zt = F @ wt; wa = fitz(F, ALL); za = F @ wa
    return L(F[vm] @ wv).mean(), L(zt[te]).mean(), np.mean(zt[te] > 0), L(za[ALL]).mean()
if os.environ.get("ENGINE") == "clone_engine":
    BASE = dict(gscale=0.8, pt_lam=1.0) if TOUR == "atp" else dict(gscale=1.0, pt_lam=0.75)
else:
    BASE = dict(a_s=0.0025, a_r=0.0025, b_s=1.0, b_r=1.0, a_x=0.0, lvl_scale=2.0, games_k=3.5)
    if TOUR == "wta": BASE["g_w"] = 1.0
if len(sys.argv) > 3 and sys.argv[3] == "ref":
    for lab, F in (("production", B0), ("prod general alone", np.column_stack([base, corr]))):
        v, t_, acc, al = score(F); print(f"{lab:50s} VAL {v:.5f} test {t_:.5f} acc {acc*100:.2f} ALL {al:.5f}", flush=True)
for g in json.loads(sys.argv[2]):
    combo = g.pop("combo", 0); cfg = {**BASE, **g}; t0 = time.time()
    z, st = E.replay(rows, H.age_at, cfe["init_rank"], cfe["init_prior"], **cfg)
    v, t_, acc, al = score(np.column_stack([z, corr]))
    if os.environ.get("CHECK"): print("   corr with production base", np.corrcoef(z, base)[0, 1], "max abs diff", np.abs(z - base).max())
    print(f"{json.dumps(g):50s} VAL {v:.5f} test {t_:.5f} acc {acc*100:.2f} ALL {al:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    if g.get("diag") or combo == 2:
        bo5 = np.array([r["bo"] == 5 for r in rows], float); x = st["dsv"] + st["drt"]; lv = st["lev"] - np.mean(st["lev"])
        for lab, F in (("+ linear point gap, x bo5", np.column_stack([z, x, x * bo5, corr])),
                       ("+ serve-level interactions", np.column_stack([z, x, x * bo5, lv, lv * z, corr])),
                       ("+ closing/h2h-free raw logit", np.column_stack([z, st["zraw"], corr]))):
            v, t_, acc, al = score(F); print(f"   {lab:47s} VAL {v:.5f} test {t_:.5f} acc {acc*100:.2f} ALL {al:.5f}", flush=True)
    if combo == 3:
        v, t_, acc, al = score(np.column_stack([z, st["alt"], corr])); print(f"   + other prediction (same ratings)               VAL {v:.5f} test {t_:.5f} acc {acc*100:.2f} ALL {al:.5f}", flush=True)
    if combo == 1:
        v, t_, acc, al = score(np.column_stack([base, z, corr])); print(f"   + general rating                                VAL {v:.5f} test {t_:.5f} acc {acc*100:.2f} ALL {al:.5f}", flush=True)
