"""Do RESULT-based inputs add to the points-driven production model?  (user: Swiatek crushes weaker players but
struggles against better ones; the points update rewards the crushing.)

Candidates, all causal (each match uses only what happened before it), added next to the full production stack:
  elo_ta      -- Tennis-Abstract-style Elo, overall + surface 50/50 (K = 250/(n+5)^0.4), win/loss only
  elo_fast    -- same but K x2 and a floor (rolling form: recent results count more)
  vs_top      -- shrunk record vs top-20 opponents beyond the model's expectation (K=25 pseudo-matches),
                 only switched on when the OPPONENT is top-20 (by tour rank)
  both        -- elo_ta + vs_top
Chosen on the selection years; test reported once."""
import paths as P
import json, math, sys, os, datetime
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import confounder_harness as H, score_driven as S
from margin_model import parse_games
import court_speed_lib as CSL
OUT = P.OUT; WD = P.ROOT + "/wta_data"
TOUR = sys.argv[1]
if TOUR == "wta":
    H.DATA_PATHS = [f"{WD}/wta_tml.csv"]; H.birth = json.load(open(f"{WD}/wta_birth_ordinal_v2.json"))
    T = np.load(f"{OUT}/wta2_wta_tml.npz", allow_pickle=True)
else:
    T = np.load(f"{OUT}/cf_FULL_v21e_reference.npz", allow_pickle=True)
allrows = H.load()
rows = [r for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])]
n = len(rows); assert n == len(T["base"])
cfb = json.load(open(f"{OUT}/confounder_fit_v21b.json")); f = list(T["feats"]); sel = [x for x in cfb["selected"] if x in f]
base = T["base"]; X10 = T["X"][:, [f.index(x) for x in sel]]; when = T["when"]
if TOUR == "atp":     # production age cap
    ag = np.array([[H.age_at(r["a"], r["when"]) or 26.0, H.age_at(r["b"], r["when"]) or 26.0] for r in rows])
    c32 = np.minimum(np.maximum(0, ag - 32), 3); X10[:, sel.index("age_over32")] = c32[:, 0] - c32[:, 1]
cs = CSL.build(allrows, tag=f"court_{TOUR}"); acer = cs["acer_a"] - cs["acer_b"]; court = np.array([CSL.court(r) for r in rows]); has = cs["has_speed"]; sp = cs["speed"]
# Fixed dates (= 80% / 85% points of the data when the model was fitted), so daily data updates don't move the split
split = datetime.date(2017, 2, 27).toordinal() if TOUR == "atp" else datetime.date(2018, 10, 8).toordinal()
tr = when < split; te = ~tr
vcut = (datetime.date(2008, 4, 28) if TOUR == "atp" else datetime.date(2014, 7, 21)).toordinal(); fm, vm = tr & (when < vcut), tr & (when >= vcut)
cent = {c: (float(np.mean(sp[(court == c) & has & tr])) if ((court == c) & has & tr).any() else 0.0) for c in set(court)}
speed = np.where(has, sp - np.array([cent[c] for c in court]), 0.0) * acer
z0 = base + X10 @ np.array(cfb["beta"][:len(sel)])
dsum = defaultdict(float); dn = defaultdict(int); fd = np.zeros(n)
for i, r in enumerate(rows):
    a, b, bo = r["a"], r["b"], r["bo"]
    fd[i] = dsum[a] / (dn[a] + 50) - dsum[b] / (dn[b] + 50)
    q = H.inv_match(round(min(max(1 / (1 + math.exp(-z0[i])), 0.001), 0.999), 4), bo)
    if len(r["score"].split()) == 2 * (2 if bo == 3 else 3) - 1:
        dsum[a] += 1 - q; dsum[b] -= 1 - q; dn[a] += 1; dn[b] += 1
slam = np.array([r["lvl"] == "G" for r in rows], float)
t4 = np.array([(1.0 if (r["rank_a"] and r["rank_a"] <= 4) else 0.0) - (1.0 if (r["rank_b"] and r["rank_b"] <= 4) else 0.0) for r in rows])
cols = [base, X10, speed, fd, slam * t4]
if TOUR == "wta":
    import sr_lib
    cfe = json.load(open(f"{OUT}/confounder_fit_v21e.json"))
    z_sr, _ = sr_lib.replay(rows, json.load(open(f"{OUT}/wta_fit.json"))["sr_alpha"], H.age_at, cfe["init_rank"], cfe["init_prior"])
    cols.append(z_sr)
B0 = np.column_stack(cols)
def fitz(F, m):
    def g(w):
        zz = F[m] @ w; p = 1 / (1 + np.exp(-zz))
        return np.logaddexp(0, -zz).sum() + 5e-5 * (w[1:] @ w[1:]), -(F[m].T @ (1 - p)) + np.r_[0, 1e-4 * w[1:]]
    w0 = np.zeros(F.shape[1]); w0[0] = 1; return minimize(g, w0, jac=True, method="L-BFGS-B").x
L = lambda z: np.logaddexp(0, -z)
zprod = B0 @ fitz(B0, tr)                          # production prediction (for the vs-top residual's expectations)


L = lambda z: np.logaddexp(0, -z)
import importlib; sr_engine = importlib.import_module(os.environ.get('ENGINE', 'sr_engine'))
cfe = json.load(open(f"{OUT}/confounder_fit_v21e.json"))
corr = np.column_stack(cols[1:-1] if TOUR == "wta" else cols[1:])      # corrections without the main rating (and without WTA's old split add-on)
import pickle
np.savez(f"cache_{TOUR}.npz", B0=B0, base=base, corr=corr, when=when, fm=fm, vm=vm, tr=tr, te=te)
pickle.dump(rows, open(f"cache_{TOUR}_rows.pkl", "wb"))
print("cached", TOUR, n)
