"""Player-profile data: every match of every player (10+ rated matches), with the full model's pre-match win chance and
what the match did to the ratings.   python3 build_profiles.py atp|wta

Writes site_v8/prof_{tour}_index.json  ([name, bucket, rated matches, last date, current level] per player)
   and site_v8/prof_{tour}_{k}.json    (players in bucket k -> list of matches, newest last)
Match row: [date, tournament, level, surface, round, opponent, won, score, win_chance, level_before, d_level, d_serve,
            d_return, d_surface, opp_level, serve_pts_won_pct, return_pts_won_pct, rated, opp_rank, performance, pts_surprise, pts_slope,
            closing_after, clutch_after]
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
    return lv, s, r_, sl, X["c"][p]          # level, serve, return, surface level, closing (set-level finishing)
pre, post = {}, {}
def hook(i, r, X):
    pre[i] = {p: state(p, r["surf"], X) for p in (r["a"], r["b"])}
def hook_post(i, r, X):
    post[i] = {p: state(p, r["surf"], X) for p in (r["a"], r["b"])}
z, st = CE.replay_ckpt(f"profiles_{TOUR}", allrows, EXT, H.age_at, CFE["init_rank"], CFE["init_prior"], carry=dict(pre=pre, post=post),
                       hook=hook, hook_post=hook_post, **CFG); z = z[RIDX]

# full-model pre-match probability (same corrections refit as the odds export)
C = np.load(f"{SCR}/tennis/cache_{TOUR}.npz"); corr = C["corr"].copy(); tr = C["tr"]
dsum = defaultdict(float); dn = defaultdict(int); fd = np.zeros(len(rows)); clutch_after = {}
for i, r in enumerate(rows):
    a, b, bo = r["a"], r["b"], r["bo"]
    fd[i] = dsum[a] / (dn[a] + 50) - dsum[b] / (dn[b] + 50)
    q = H.inv_match(round(min(max(1 / (1 + math.exp(-z[i])), 0.001), 0.999), 4), bo)
    if len(r["score"].split()) == 2 * (2 if bo == 3 else 3) - 1:
        dsum[a] += 1 - q; dsum[b] -= 1 - q; dn[a] += 1; dn[b] += 1
    clutch_after[id(r)] = (dsum[a] / (dn[a] + 50), dsum[b] / (dn[b] + 50))   # deciding-set clutch after the match (as in the odds)
corr[:, 11] = fd
F = np.column_stack([z, corr])
def g(w):
    zz = F[tr] @ w; p = 1 / (1 + np.exp(-zz))
    return np.logaddexp(0, -zz).sum() + 5e-5 * (w[1:] @ w[1:]), -(F[tr].T @ (1 - p)) + np.r_[0, 1e-4 * w[1:]]
w0 = np.zeros(F.shape[1]); w0[0] = 1
zf = F @ minimize(g, w0, jac=True, method="L-BFGS-B").x
pa_all = np.full(len(EXT), np.nan); pa_all[RIDX] = 1 / (1 + np.exp(-zf))

R3 = lambda x: round(float(x), 3); R4 = lambda x: round(float(x), 4)
# ---- match performance rating: the level at which the model would have expected exactly the points the player won
# (on serve + on return, against this opponent on this surface), i.e. the rating a zero update would have come from.
# Shift d on both serve and return of the player: expected points won = sn*sig(m + (s+d - r_opp)/PT_M) + rn*(1 - sig(m + (s_opp - r-d)/PT_M)).
PT_M = 2.132; S4 = st["S4"]
pidx = [i for i in range(len(EXT)) if not EXT[i].get("upd_only") and EXT[i]["pts"]]
perf = {}
if pidx:
    S4p = S4[pidx]; P = np.array([EXT[i]["pts"] for i in pidx], float)       # (won, tot, srv_won, srv_n, ret_won, ret_n), winner's view
    sig_ = lambda x: 1 / (1 + np.exp(-x))
    for side in (0, 1):                     # 0 = winner (a), 1 = loser (b)
        s_me, r_me, s_op, r_op = (S4p[:, 0], S4p[:, 1], S4p[:, 2], S4p[:, 3]) if side == 0 else (S4p[:, 2], S4p[:, 3], S4p[:, 0], S4p[:, 1])
        sw, sn, rw, rn = (P[:, 2], P[:, 3], P[:, 4], P[:, 5]) if side == 0 else (P[:, 5] - P[:, 4], P[:, 5], P[:, 3] - P[:, 2], P[:, 3])
        m = S4p[:, 4]; lo = np.full(len(pidx), -5.0); hi = np.full(len(pidx), 5.0)
        for _ in range(50):
            d = (lo + hi) / 2
            exp_ = sn * sig_(m + (s_me + d - r_op) / PT_M) + rn * (1 - sig_(m + (s_op - r_me - d) / PT_M))
            up = exp_ < sw + rw; lo = np.where(up, d, lo); hi = np.where(up, hi, d)
        d = (lo + hi) / 2
        # for combining matches (season performance): surprise at the actual rating (points won - expected) and its slope
        # in the shift, d expected / d d; sum(surprise) / sum(slope) is the shift that zeroes the summed update
        q1 = sig_(m + (s_me - r_op) / PT_M); q2 = sig_(m + (s_op - r_me) / PT_M)
        sur = sw + rw - (sn * q1 + rn * (1 - q2)); slo = (sn * q1 * (1 - q1) + rn * q2 * (1 - q2)) / PT_M
        for k, i in enumerate(pidx): perf[(i, side)] = (float(d[k]), float(sur[k]), float(slo[k]))
orank = lambda r, won: r["rank_b"] if won else r["rank_a"]     # opponent's official ranking at the time (None if unknown)
# how much the win chance moves per unit of level gap (logit scale), for the profile's performance rating
_x = np.array([pre[i][EXT[i]["a"]][0] - pre[i][EXT[i]["b"]][0] for i in RIDX]); _y = np.log(pa_all[RIDX] / (1 - pa_all[RIDX]))
SLOPE = float(np.cov(_x, _y)[0, 1] / np.var(_x, ddof=1))
recs = defaultdict(list); nrated = defaultdict(int); myrank = {}; cl_now = defaultdict(float)   # own official ranking per match row (best list only)
for r in allrows:
    i = WHERE.get(id(r))
    if id(r) in clutch_after: cl_now[r["a"]], cl_now[r["b"]] = clutch_after[id(r)]
    for me, op, won in ((r["a"], r["b"], 1), (r["b"], r["a"], 0)):
        base = [r["when"].isoformat(), r.get("tname") or "", r["lvl"] or "", r["surf"] or "", r["rnd"] or "", op, won, r["score"] or ""]
        if i is None or i not in pre or i not in post:
            recs[me].append(base + [None] * 9 + [0, orank(r, won), None, None, None, None, R4(cl_now[me])]); continue
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
                                R3(o0[0]), R3(spct * 100) if spct is not None else None, R3(rpct * 100) if rpct is not None else None, 2 if uo else 1, orank(r, won),
                                *((R3(b0[0] + perf[(i, 1 - won)][0]), round(perf[(i, 1 - won)][1], 2), round(perf[(i, 1 - won)][2], 2))
                                  if (i, 1 - won) in perf else (None, None, None)), R3(b1[4]), R4(cl_now[me])])     # [22] closing, [23] clutch after
        myrank[id(recs[me][-1])] = r["rank_a"] if won else r["rank_b"]
players = [p for p in recs if nrated[p] >= 10]
bucket = lambda p: zlib.crc32(p.encode()) % NB
index = []
for p in sorted(players, key=lambda p: -nrated[p]):
    lv = (st["sg"][p] + st["rg"][p]) / 2
    index.append([p, bucket(p), nrated[p], recs[p][-1][0], R3(lv)])
D = f"{SCR}/site_v8"
# best single-match performances (Best performances tab): the 100 highest match performance ratings
# tour-level main draws only: against Challenger/qualifying/Davis-Fed Cup mismatches a near-perfect score is almost expected
# and the performance shoots up without saying much; for the same reason both players must be rated at least 0.4 going in
tour_main = lambda r: r[2] not in ("C", "D", "Q", "S") and not (r[4] or "").startswith("Q")
# the point stats must fit the result (a few source rows have them swapped, e.g. a loser winning 70% of the points)
fits = lambda r: r[15] is not None and r[16] is not None and ((r[15] + r[16]) / 2 > 45 if r[6] else (r[15] + r[16]) / 2 < 55)
cand = sorted(((p, r) for p in recs for r in recs[p] if r[19] is not None and tour_main(r) and fits(r)), key=lambda pr: -pr[1][19])
row = lambda p, r: [p] + [r[k] for k in (0, 1, 2, 3, 4, 5, 6, 7, 19, 9, 14, 15, 16)] + [myrank.get(id(r)), r[18]]
top = lambda k: k is not None and k <= 20
json.dump(dict(all=[row(p, r) for p, r in cand if min(r[9], r[14]) >= 0.4][:100],              # both rated at least 0.4 going in
               top20=[row(p, r) for p, r in cand if top(myrank.get(id(r))) and top(r[18])][:100]),  # both in the official top 20
          open(f"{SCR}/site_v8/best_{TOUR}.json", "w"), separators=(",", ":"))
# rows: player, date, event, level, surface, round, opponent, won, score, perf, rating in, opp rating, serve%, return%, rank, opp rank
json.dump(dict(tour=TOUR, lam=LAM, slope=round(SLOPE, 4), players=index), open(f"{D}/prof_{TOUR}_index.json", "w"), separators=(",", ":"))
# career peaks of the serve and the return rating (after each rated match), for the profile page
srpk = {}
for r in allrows:
    i = WHERE.get(id(r))
    if i is None or i not in post or EXT[i].get("upd_only"): continue
    for p in (r["a"], r["b"]):
        _, sv, rt, _, _ = post[i][p]; x = srpk.get(p)
        if x is None: srpk[p] = x = [sv, r["when"], rt, r["when"]]
        if sv > x[0]: x[0], x[1] = sv, r["when"]
        if rt > x[2]: x[2], x[3] = rt, r["when"]
pset = set(players)
json.dump({p: [R3(x[0]), x[1].isoformat(), R3(x[2]), x[3].isoformat()] for p, x in srpk.items() if p in pset},
          open(f"{D}/srpeak_{TOUR}.json", "w"), separators=(",", ":"))
tot = 0
for k in range(NB):
    part = {p: recs[p] for p in players if bucket(p) == k}
    fn = f"{D}/prof_{TOUR}_{k}.json"; json.dump(part, open(fn, "w"), separators=(",", ":"))
    import os; tot += os.path.getsize(fn)
print(TOUR, len(players), "players,", sum(nrated[p] for p in players), "rated player-matches,", round(tot / 1e6, 1), "MB in", NB, "files")
P = "Jannik Sinner" if TOUR == "atp" else "Iga Swiatek"
for x in recs[P][-3:]: print("  ", x)
