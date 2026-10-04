"""Player-profile data: every match of every player (10+ rated matches), with the full model's pre-match win chance and
what the match did to the ratings.   python3 build_profiles.py atp|wta

Writes site_v8/prof_{tour}_index.json  ([name, bucket, rated matches, last date, current level] per player)
   and site_v8/prof_{tour}_{k}.json    (players in bucket k -> list of matches, newest last)
Match row: [date, tournament, level, surface, round, opponent, won, score, win_chance, level_before, d_level, d_serve,
            d_return, d_surface, opp_level, serve_pts_won_pct, return_pts_won_pct, rated]
Ratings in the model's units (level = (serve + return) / 2; surface = level on that surface = general + gap)."""
import paths as P
import json, sys, math, zlib
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import score_driven as S
import confounder_harness as H
import clone_engine as CE
from margin_model import parse_games

TOUR = sys.argv[1]
OUT = P.OUT; SCR = P.ROOT
if TOUR == "wta":
    H.DATA_PATHS = [f"{SCR}/wta_data/wta_tml.csv"]; H.birth = json.load(open(f"{SCR}/wta_data/wta_birth_ordinal_v2.json"))
CFE = json.load(open(f"{OUT}/confounder_fit_v21e.json")); CFG = json.load(open(f"{OUT}/sr_clone_fit.json"))[TOUR]
SURF = H.SURFACES; LAM = CFG.get("lam", 0.4); NB = 48
allrows = H.load()
rows = [r for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])]
EXT, RIDX, WHERE = CE.build_ext(allrows)

def state(p, surf, X):
    s, r_ = X["sg"][p], X["rg"][p]
    lv = (s + r_) / 2
    sl = (s + LAM * X["ss"][surf][p] + r_ + LAM * X["rs"][surf][p]) / 2 if surf in SURF else lv
    return lv, s, r_, sl
pre, post = {}, {}
def hook(i, r, X):
    pre[i] = {p: state(p, r["surf"], X) for p in (r["a"], r["b"])}
def hook_post(i, r, X):
    post[i] = {p: state(p, r["surf"], X) for p in (r["a"], r["b"])}
z, st = CE.replay_ckpt(f"profiles_{TOUR}", allrows, EXT, H.age_at, CFE["init_rank"], CFE["init_prior"], carry=dict(pre=pre, post=post),
                       hook=hook, hook_post=hook_post, **CFG); z = z[RIDX]

# full-model pre-match probability (same corrections refit as the odds export)
C = np.load(f"{SCR}/tennis/cache_{TOUR}.npz"); corr = C["corr"].copy(); tr = C["tr"]
dsum = defaultdict(float); dn = defaultdict(int); fd = np.zeros(len(rows))
for i, r in enumerate(rows):
    a, b, bo = r["a"], r["b"], r["bo"]
    fd[i] = dsum[a] / (dn[a] + 50) - dsum[b] / (dn[b] + 50)
    q = H.inv_match(round(min(max(1 / (1 + math.exp(-z[i])), 0.001), 0.999), 4), bo)
    if len(r["score"].split()) == 2 * (2 if bo == 3 else 3) - 1:
        dsum[a] += 1 - q; dsum[b] -= 1 - q; dn[a] += 1; dn[b] += 1
corr[:, 11] = fd
F = np.column_stack([z, corr])
def g(w):
    zz = F[tr] @ w; p = 1 / (1 + np.exp(-zz))
    return np.logaddexp(0, -zz).sum() + 5e-5 * (w[1:] @ w[1:]), -(F[tr].T @ (1 - p)) + np.r_[0, 1e-4 * w[1:]]
w0 = np.zeros(F.shape[1]); w0[0] = 1
zf = F @ minimize(g, w0, jac=True, method="L-BFGS-B").x
pa_all = np.full(len(EXT), np.nan); pa_all[RIDX] = 1 / (1 + np.exp(-zf))

R3 = lambda x: round(float(x), 3)
recs = defaultdict(list); nrated = defaultdict(int)
for r in allrows:
    i = WHERE.get(id(r))
    for me, op, won in ((r["a"], r["b"], 1), (r["b"], r["a"], 0)):
        base = [r["when"].isoformat(), r.get("tname") or "", r["lvl"] or "", r["surf"] or "", r["rnd"] or "", op, won, r["score"] or ""]
        if i is None or i not in pre or i not in post:
            recs[me].append(base + [None] * 9 + [0]); continue
        uo = EXT[i].get("upd_only", False)
        if not uo: nrated[me] += 1
        b0, b1, o0 = pre[i][me], post[i][me], pre[i][op]
        pw = None if np.isnan(pa_all[i]) else (pa_all[i] if won else 1 - pa_all[i])
        spct = rpct = None
        if r["pts"]:
            _, _, sw, sn, rw, rn = r["pts"]
            if won: spct, rpct = (sw / sn if sn else None), (rw / rn if rn else None)
            else: spct, rpct = ((rn - rw) / rn if rn else None), ((sn - sw) / sn if sn else None)
        recs[me].append(base + [R3(pw) if pw is not None else None, R3(b0[0]), R3(b1[0] - b0[0]), R3(b1[1] - b0[1]), R3(b1[2] - b0[2]), R3(b1[3] - b0[3]),
                                R3(o0[0]), R3(spct * 100) if spct is not None else None, R3(rpct * 100) if rpct is not None else None, 2 if uo else 1])
players = [p for p in recs if nrated[p] >= 10]
bucket = lambda p: zlib.crc32(p.encode()) % NB
index = []
for p in sorted(players, key=lambda p: -nrated[p]):
    lv = (st["sg"][p] + st["rg"][p]) / 2
    index.append([p, bucket(p), nrated[p], recs[p][-1][0], R3(lv)])
D = f"{SCR}/site_v8"
json.dump(dict(tour=TOUR, lam=LAM, players=index), open(f"{D}/prof_{TOUR}_index.json", "w"), separators=(",", ":"))
tot = 0
for k in range(NB):
    part = {p: recs[p] for p in players if bucket(p) == k}
    fn = f"{D}/prof_{TOUR}_{k}.json"; json.dump(part, open(fn, "w"), separators=(",", ":"))
    import os; tot += os.path.getsize(fn)
print(TOUR, len(players), "players,", sum(nrated[p] for p in players), "rated player-matches,", round(tot / 1e6, 1), "MB in", NB, "files")
P = "Jannik Sinner" if TOUR == "atp" else "Iga Swiatek"
for x in recs[P][-3:]: print("  ", x)
