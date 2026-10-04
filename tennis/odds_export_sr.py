"""Export for the Rankings and Match-odds tabs -- serve/return model (Oct 4 2026).

    python3 odds_export_sr.py atp|wta

Main rating: clone_engine.replay with the config in sr_clone_fit.json (production machinery, separate serve and
return ratings per player + per surface, prediction via service points -> holds -> sets (point tiebreak) -> match,
point edge shrunk by hm, early-career gain, no decay). Corrections (age, height x fast court, court speed x ace
skill, deciding-set clutch, top-4 at Slams, + match-context terms that are neutral in the odds tab) refit on the
training years on top of it. Writes odds_data_{tour}.json + Python reference probabilities for the JS parity test.
"""
import paths as P
import json, math, sys, pickle, datetime
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import score_driven as S
import confounder_harness as H
import clone_engine as CE
import sr_lib
import court_speed_lib as CSL
from margin_model import parse_games

TOUR = sys.argv[1]
OUT = P.OUT; SCR = P.ROOT
if TOUR == "wta":
    H.DATA_PATHS = [f"{SCR}/wta_data/wta_tml.csv"]; H.birth = json.load(open(f"{SCR}/wta_data/wta_birth_ordinal_v2.json"))
CFE = json.load(open(f"{OUT}/confounder_fit_v21e.json"))
CFG = json.load(open(f"{OUT}/sr_clone_fit.json"))[TOUR]
SURF = H.SURFACES; PT_M = 2.132; HM = CFG["hm"]; BW = 0.6
allrows = H.load()
rows = [r for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])]
n = len(rows)
EXT, RIDX, _ = CE.build_ext(allrows)
z, st = CE.replay(EXT, H.age_at, CFE["init_rank"], CFE["init_prior"], **CFG); z = z[RIDX]
sg, rg, ss, rs, ns, c, mu, last = (st[k] for k in ("sg", "rg", "ss", "rs", "ns", "c", "mu", "last"))

# ---------------- corrections refit on top of the new rating ----------------
C = np.load(f"{SCR}/tennis/cache_{TOUR}.npz"); corr = C["corr"].copy(); when = C["when"]; tr = C["tr"]; te = C["te"]
assert len(when) == n
# deciding-set clutch recomputed against the NEW model (column 11 of corr: X10 + speed + clutch + slam*top4)
dsum = defaultdict(float); dn = defaultdict(int); fd = np.zeros(n)
for i, r in enumerate(rows):
    a, b, bo = r["a"], r["b"], r["bo"]
    fd[i] = dsum[a] / (dn[a] + 50) - dsum[b] / (dn[b] + 50)
    q = H.inv_match(round(min(max(1 / (1 + math.exp(-z[i])), 0.001), 0.999), 4), bo)
    if len(r["score"].split()) == 2 * (2 if bo == 3 else 3) - 1:
        dsum[a] += 1 - q; dsum[b] -= 1 - q; dn[a] += 1; dn[b] += 1
corr[:, 11] = fd
cfb = json.load(open(f"{OUT}/confounder_fit_v21b.json")); T0 = np.load(f"{OUT}/" + ("cf_FULL_v21e_reference.npz" if TOUR == "atp" else "wta2_wta_tml.npz"), allow_pickle=True)
sel = [x for x in cfb["selected"] if x in list(T0["feats"])]
NAMES = sel + ["speedw_x_ace_gap", "deciding_set_clutch", "slam_x_atp_top4"]
assert corr.shape[1] == len(NAMES), (corr.shape, NAMES)
F = np.column_stack([z, corr])
def fitz(F, m):
    def g(w):
        zz = F[m] @ w; p = 1 / (1 + np.exp(-zz))
        return np.logaddexp(0, -zz).sum() + 5e-5 * (w[1:] @ w[1:]), -(F[m].T @ (1 - p)) + np.r_[0, 1e-4 * w[1:]]
    w0 = np.zeros(F.shape[1]); w0[0] = 1; return minimize(g, w0, jac=True, method="L-BFGS-B").x
w = fitz(F, tr); zt = F @ w
print(f"{TOUR.upper()} held-out from {datetime.date.fromordinal(int(when[te].min()))}: {te.sum()} matches, logloss {np.mean(np.logaddexp(0, -zt[te])):.5f}, accuracy {np.mean(zt[te] > 0)*100:.2f}%")
SLOPE = float(w[0]); B = dict(zip(NAMES, map(float, w[1:])))
print("slope", round(SLOPE, 4), {k: round(v, 4) for k, v in B.items()})

# ---------------- player state at the export date ----------------
NOW = rows[-1]["when"]
def dec(p):
    idle = max(0, (NOW - last[p]).days - H.IDLE_GRACE_DAYS)
    return H.IDLE_GB ** idle if idle > 0 else 1.0
mc = defaultdict(int); h2h = defaultdict(lambda: defaultdict(int))
for r in rows: mc[r["a"]] += 1; mc[r["b"]] += 1; h2h[frozenset((r["a"], r["b"]))][r["a"]] += 1
last_rank = {}; ioc = {}; hts = {}
for r in allrows:
    for p, s in ((r["a"], "a"), (r["b"], "b")):
        if r["ioc_" + s]: ioc[p] = r["ioc_" + s]
        h = r["ht_" + s]
        if h and 150 < h < 220: hts[p] = h
        if r["rank_" + s]: last_rank[p] = (r["rank_" + s], r["when"])
HT_MEAN = float(np.mean([h for r in allrows for h in (r["ht_a"], r["ht_b"]) if h and 150 < h < 220]))
clutch = {p: dsum[p] / (dn[p] + 50) for p in dsum}
cs_b = CSL.build(allrows); ace = {p: math.log(v) for p, v in cs_b["ace_skill"].items()}

# classic Elo for comparison (Tennis Abstract style: K = 250/(n+5)^0.4, win/loss only; surface Elo per surface)
elo = defaultdict(lambda: 1500.0); elo_s = {x: defaultdict(lambda: 1500.0) for x in SURF}
en = defaultdict(int); ens = {x: defaultdict(int) for x in SURF}
for r in rows:
    a, b, sf = r["a"], r["b"], r["surf"]
    for E, N in [(elo, en)] + ([(elo_s[sf], ens[sf])] if sf in SURF else []):
        pa = 1 / (1 + 10 ** ((E[b] - E[a]) / 400)); ka = 250 / (N[a] + 5) ** 0.4; kb = 250 / (N[b] + 5) ** 0.4
        E[a] += ka * (1 - pa); E[b] -= kb * (1 - pa); N[a] += 1; N[b] += 1
active = [p for p in last if mc[p] >= 30 and (NOW - last[p]).days <= 365]
lvl = lambda p: (sg[p] + rg[p]) / 2 * dec(p)
active.sort(key=lambda p: -lvl(p)); active = active[:450]
R4 = lambda x: round(float(x), 4)
P = []
for p in active:
    d = dec(p); ag = H.age_at(p, NOW); rk = last_rank.get(p)
    P.append(dict(n=p, g=R4((sg[p] + rg[p]) / 2 * d), sv=R4(sg[p] * d), rt=R4(rg[p] * d),
                  ssv=[R4(ss[x][p] * d) for x in SURF], srt=[R4(rs[x][p] * d) for x in SURF], ns=[ns[x][p] for x in SURF],
                  c=R4(c[p] * d), cl=R4(clutch.get(p, 0.0)), age=round(ag, 2) if ag is not None else None,
                  rk=rk[0] if rk and (NOW - rk[1]).days <= 400 else None, ht=hts.get(p), ioc=ioc.get(p, ""),
                  ace=R4(ace.get(p, 0.0)), m=mc[p], last=last[p].isoformat(),
                  elo=round(elo[p], 1), eloS=[round(elo_s[x][p], 1) for x in SURF]))
idx = {x["n"]: i for i, x in enumerate(P)}
H2H = []
for k, v in h2h.items():
    if len(k) != 2: continue
    x, y = tuple(k)
    if x in idx and y in idx: H2H.append([idx[x], idx[y], v.get(x, 0), v.get(y, 0)])

# rank-fill pools at the export date: per surface, sorted general serve / surface serve / general return / surface return
fill_pool = {}
for x in SURF:
    pool = [p for p, d_ in last.items() if ns[x][p] >= 1 and (NOW - d_).days <= 365]
    if len(pool) >= 30:
        fill_pool[x] = [sorted(R4(D[p] * dec(p)) for p in pool) for D in (sg, ss[x], rg, rs[x])]

# court-speed presets (both tours, from ace data)
court = np.array([CSL.court(r) for r in rows]); has = cs_b["has_speed"]; sp = cs_b["speed"]
cent = {c_: (float(np.mean(sp[(court == c_) & has & tr])) if ((court == c_) & has & tr).any() else 0.0) for c_ in set(court)}
sdw = np.where(has, sp - np.array([cent[c_] for c_ in court]), 0.0)
agg = defaultdict(list)
for r, h_, v in zip(rows, has, sdw):
    if h_ and r["level"] == "tour" and r["when"].year >= NOW.year - 2 and r["surf"] in SURF:
        agg[(r["tname"].replace(" Masters", ""), r["surf"], bool(r["indoor"]), r["lvl"])].append(v)
LVMAP = {"G": "G", "M": "M", "F": "F", "500": "500"} if TOUR == "atp" else {"G": "G", "M": "M", "PM": "M", "P5": "M", "1000": "M", "T1": "M", "F": "F", "500": "500", "P": "500"}
presets = []
for (tn, sf, ind, lv), vs in agg.items():
    if len(vs) >= 40 and lv in LVMAP:
        presets.append(dict(t=tn, s=sf, i=ind, l=LVMAP[lv], v=round(float(np.mean(vs)), 3), n=len(vs)))
presets.sort(key=lambda x: ({"G": 0, "M": 1, "F": 2}.get(x["l"], 3), x["v"]))
speed_range = [round(float(np.percentile(sdw[has], 2)), 2), round(float(np.percentile(sdw[has], 98)), 2)]

AGE_CAP = 3 if TOUR == "atp" else None
RF = json.load(open(f"{OUT}/ret_fit_{TOUR}.json"))
RET = dict(r_slam=RF["r_slam"], b_slam=max(0.0, RF["b_slam"]), r_other=RF["r_other"], b_other=max(0.0, RF["b_other"]))
const = dict(tour=TOUR, asof=NOW.isoformat(), model="serve_return", ret=RET, gap=CFG.get("gap", False), lam=CFG.get("lam", 0.4), BLEND_W=BW, MIN_SURF=1, fill=True, age_cap=AGE_CAP,
             CAL_I=H.CAL_I, CAL_S=H.CAL_S, slope=SLOPE, h2h_tbl=H.h2h_tbl, HT_MEAN=HT_MEAN, hm=HM, pt_m=PT_M,
             mu={k: round(float(v), 6) for k, v in mu.items()},
             beta={k: B.get(k, 0.0) for k in ["big_x_age27", "height_x_fast", "age_over32", "speedw_x_ace_gap", "deciding_set_clutch", "slam_x_atp_top4"]},
             speed_range=speed_range)

# ---------------- python reference probabilities (same chain as the JS) ----------------
def qmap(x, gs, ss_):
    q = np.searchsorted(gs, x) / len(gs)
    return float(np.interp(q, (np.arange(len(ss_)) + 0.5) / len(ss_), ss_))
GAP = CFG.get("gap", False); LAM = CFG.get("lam", 0.4)
def eff(Q, si, use):
    s_, r_ = Q["sv"], Q["rt"]
    if GAP: return s_ + LAM * Q["ssv"][si], r_ + LAM * Q["srt"][si]
    if Q["ns"][si] >= 1: return BW * s_ + (1 - BW) * Q["ssv"][si], BW * r_ + (1 - BW) * Q["srt"][si]
    fp = fill_pool.get(SURF[si])
    if fp is None: return s_, r_
    return BW * s_ + (1 - BW) * qmap(s_, fp[0], fp[1]), BW * r_ + (1 - BW) * qmap(r_, fp[2], fp[3])
def prob(x, y, surf, bo, indoor=False, speed=0.0, level="A"):
    X, Y = P[idx[x]], P[idx[y]]; si = SURF.index(surf)
    use = X["ns"][si] >= 1 and Y["ns"][si] >= 1
    if not GAP and not use and surf not in fill_pool: sa, ra, sb, rb = X["sv"], X["rt"], Y["sv"], Y["rt"]
    else: (sa, ra), (sb, rb) = eff(X, si, use), eff(Y, si, use)
    m = const["mu"].get(surf, 0.55)
    ha = min(max(S.sigmoid(m + HM * (sa - rb) / PT_M), 0.02), 0.98); hb = min(max(S.sigmoid(m + HM * (sb - ra) / PT_M), 0.02), 0.98)
    qs = 0.5 * (sr_lib.set_prob(round(ha, 3), round(hb, 3)) + 1 - sr_lib.set_prob(round(hb, 3), round(ha, 3))); qs = min(max(qs, 1e-6), 1 - 1e-6)
    qa = min(max(S.sigmoid(S.logit(qs) + X["c"] - Y["c"]), 1e-6), 1 - 1e-6)
    zz = H.CAL_I + H.CAL_S * S.logit(min(max(S.match_from_set_prob(qa, bo), 1e-6), 1 - 1e-6))
    hh = h2h.get(frozenset((x, y)))
    if hh:
        aw, bw = hh.get(x, 0), hh.get(y, 0); nn = aw + bw
        if nn: zz += H.h2h_slope(nn) * S.logit(min(max((aw + 0.5) / (nn + 1.0), 1e-6), 1 - 1e-6))
    zz *= SLOPE
    fast = 1.0 if (surf == "Grass" or indoor) else 0.0; big = 1.0 if level in ("G", "M") else 0.0
    def side(Q):
        ag = Q["age"] if Q["age"] is not None else 26.0; ht = Q["ht"] or HT_MEAN
        return dict(big_x_age27=max(0, ag - 27) * big, height_x_fast=(ht - HT_MEAN) / 10 * fast,
                    age_over32=(min(max(0, ag - 32), AGE_CAP) if AGE_CAP else max(0, ag - 32)),
                    speedw_x_ace_gap=speed * Q["ace"], deciding_set_clutch=Q["cl"],
                    slam_x_atp_top4=1.0 if (level == "G" and Q["rk"] and Q["rk"] <= 4) else 0.0)
    fx, fy = side(X), side(Y)
    zz += sum(const["beta"][k] * (fx[k] - fy[k]) for k in fx)
    p = 1 / (1 + math.exp(-zz))                      # chance to win if the match is completed
    rr, bb = (RET["r_slam"], RET["b_slam"]) if level == "G" else (RET["r_other"], RET["b_other"])
    return (1 - rr) * p + rr / (1 + math.exp(-bb * zz))   # chance to advance: retirement / walkover risk included
tests = []
top = [x["n"] for x in P[:8]]
for i in range(len(top)):
    for j in range(i + 1, min(i + 3, len(top))):
        for surf, bo, ind, spd, lv in [("Hard", 3, False, 0.0, "A"), ("Clay", 5, False, -0.2, "G"), ("Grass", 3, False, 0.15, "M"), ("Hard", 3, True, 0.1, "A")]:
            tests.append([top[i], top[j], surf, bo, ind, spd, lv, round(prob(top[i], top[j], surf, bo, ind, spd, lv), 6)])
for k_, sf_ in enumerate(SURF):
    for zn in [x["n"] for x in P if x["ns"][k_] == 0][:3]:
        tests.append([top[0], zn, sf_, 3, False, 0.0, "A", round(prob(top[0], zn, sf_, 3), 6)])
json.dump(dict(const=const, players=P, h2h=H2H, presets=presets, tests=tests, fill_pool=fill_pool),
          open(f"{OUT}/odds_data_{TOUR}_sr.json", "w"), separators=(",", ":"))
print(TOUR, "as of", NOW, len(P), "players,", len(H2H), "h2h,", len(presets), "presets, mu", const["mu"])
for x in P[:12]: print(f"  {x['n']:24s} level {x['g']:+.3f} serve {x['sv']:+.3f} return {x['rt']:+.3f} c={x['c']:+.3f} rk={x['rk']}")
for t in tests[:4]: print("  test", t)
