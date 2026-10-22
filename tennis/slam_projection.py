"""Projected future Grand Slam titles for the active players.   python3 slam_projection.py atp|wta  (after build_profiles.py)

Two models, both learned from year-end snapshots of every active top-150 player (rating relative to the field = rating
minus the average of No. 5-15):

 (a) regression: lead, 12-month trend, Slams won so far, age, years on tour (women: also how crowded the top is around
     the player) -> Slams won after the snapshot (Poisson; logistic for the chance of at least one more). Trained on
     snapshots up to 12 years ago (the outcome is then known for nearly every career).
 (b) year-by-year simulation: next year's 4 Slams are won at a chance set by the CURRENT lead only; then the lead moves
     by a drift that depends on age, the lead itself and the 1-year and 3-year trend (plus the spread seen in the data),
     and the player stays active with a chance set by age and level. Age therefore weighs on the later years, not on the
     coming year. Simulated 3000 times per player.

Held-out tests (trained before the test years, outcome = all Slams won since): men 2009-2014 regression 400 / simulation
487 / average 424, 2013-2016 205 / 235 / 203; women 2009-2014 608 / 460, 2013-2016 238 / 218 (deviance, lower = better).
Used: men the average of both, women the simulation. Every model is too low for the Big 3 / Serena era, when careers got
longer than ever before; the final fits include all complete data, which contains that era.
Output site_v8/slamproj_{tour}.json: player -> [Slams won, expected further Slams, chance of at least one more, expected
in the next 12 months (current lead only), chance of one in the next 12 months, expected after that]."""
import paths as P
import json, glob, math, datetime, sys
import numpy as np
from scipy.optimize import minimize

TOUR = sys.argv[1]
SITE = f"{P.ROOT}/site_v8"
BIRTH = f"{P.OUT}/player_birth_ordinal.json" if TOUR == "atp" else f"{P.ROOT}/wta_data/wta_birth_ordinal_v2.json"
FIRST_YEAR = 1985 if TOUR == "atp" else 1995
# field strength (women only): how crowded the top is around the player, and that times the player's own lead. On the
# held-out years it improved the women's projection (deviance 630 -> 608); for the men it made it worse (400 -> 449:
# in 2009-2014 the Big 3 crowded each other and still all kept winning), so it is left out there.
SPEC = (["lead", "lead+^2", "age", "age+^2", "agexlead", "trend", "slams", "yrs"] if TOUR == "atp"
        else ["lead", "age", "age+^2", "agexlead", "trend", "slams", "crowd", "crowdxlead"])
D = datetime.date.fromisoformat

rows = {}
for f in sorted(glob.glob(f"{SITE}/prof_{TOUR}_[0-9]*.json")): rows.update(json.load(open(f)))
birth = json.load(open(BIRTH))
H, SL, FIRST = {}, {}, {}
for p, L in rows.items():
    H[p] = [(D(r[0]), r[9] + r[10]) for r in L if r[9] is not None and r[10] is not None]
    SL[p] = [D(r[0]) for r in L if r[2] == "G" and r[4] == "F" and r[6]]
    FIRST[p] = D(L[0][0])


def rating_at(p, d):
    h = H[p]; lo, hi = 0, len(h)
    while lo < hi:
        m = (lo + hi) // 2
        if h[m][0] < d: lo = m + 1
        else: hi = m
    return (h[lo - 1][1], h[lo - 1][0]) if lo else (None, None)


def snapshot(d, limit=150):
    act = []
    for p in H:
        v, last = rating_at(p, d)
        if v is not None and (d - last).days <= 365 and p in birth: act.append((v, p))
    act.sort(reverse=True)
    if len(act) < 30: return []
    ref = float(np.mean([v for v, _ in act[4:15]]))
    out = []
    for k, (v, p) in enumerate(act[:limit]):
        v12, _ = rating_at(p, d - datetime.timedelta(days=365))
        others = [w for w, q in act[:20] if q != p]
        crowd = math.log(sum(math.exp(3.3 * (w - v)) for w in others))   # strength of the top 20 relative to the player
        out.append(dict(p=p, crowd=crowd, lead=v - ref, rank=k + 1, trend=(v - v12) if v12 is not None else 0.0,
                        age=(d.toordinal() - birth[p]) / 365.25, yrs=(d - FIRST[p]).days / 365.25,
                        won=sum(1 for x in SL[p] if x < d), fut=sum(1 for x in SL[p] if x >= d)))
    return out


def features(S):
    f = lambda k: np.array([s[k] for s in S], float)
    lead, a = f("lead"), f("age") - 25
    cols = {"lead": lead, "lead+^2": np.maximum(lead, 0) ** 2, "age": a, "age+^2": np.maximum(a, 0) ** 2, "agexlead": a * lead,
            "trend": f("trend"), "slams": np.log1p(f("won")), "yrs": np.minimum(f("yrs"), 15) / 10,
            "crowd": f("crowd"), "crowdxlead": f("crowd") * np.maximum(lead, 0)}
    return np.column_stack([np.ones(len(S))] + [cols[k] for k in SPEC])


def fit_logit(X, y, l2=1e-2):
    """P(at least one more Slam): fitted on its own (careers are far more uneven than a Poisson count implies)."""
    def nll(b):
        z = X @ b
        return np.logaddexp(0, z).sum() - y @ z + l2 * (b[1:] @ b[1:]), X.T @ (1 / (1 + np.exp(-z)) - y) + np.r_[0, 2 * l2 * b[1:]]
    return minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def fit(X, y, l2=1e-2):
    def nll(b):
        eta = np.clip(X @ b, -20, 5); mu = np.exp(eta)
        return (mu - y * eta).sum() + l2 * (b[1:] @ b[1:]), X.T @ (mu - y) + np.r_[0, 2 * l2 * b[1:]]
    b0 = np.zeros(X.shape[1]); b0[0] = math.log(max(y.mean(), 1e-3))
    return minimize(nll, b0, jac=True, method="L-BFGS-B").x


last = max(h[-1][0] for h in H.values() if h)
SNAP = {Y: snapshot(datetime.date(Y, 12, 31)) for Y in range(FIRST_YEAR - 3, last.year)}
NOW_D = last + datetime.timedelta(days=1)
now = snapshot(NOW_D)

# ---- (a) regression ----
train = [s for Y in range(FIRST_YEAR, last.year - 11) for s in SNAP[Y]]
Xtr = features(train); ytr = np.array([s["fut"] for s in train], float)
b = fit(Xtr, ytr); bl = fit_logit(Xtr, (ytr > 0).astype(float))
def regress(S):
    X = features(S); m = np.exp(np.clip(X @ b, -20, 5))
    # a count spread more unevenly than a Poisson with the same mean always has P(>=1) <= 1 - exp(-mean)
    return m, np.minimum(1 / (1 + np.exp(-(X @ bl))), 1 - np.exp(-m))

# ---- (b) year-by-year simulation ----
REF = {}
def field_ref(d):
    if d not in REF:
        act = sorted((v for p in H for v, last_ in [rating_at(p, d)] if v is not None and (d - last_).days <= 365), reverse=True)
        REF[d] = float(np.mean(act[4:15])) if len(act) >= 30 else None
    return REF[d]
def lead_at(p, d):
    v, last_ = rating_at(p, d); r = field_ref(d)
    return None if v is None or r is None or (d - last_).days > 365 else v - r
def state(s, d):
    l1 = lead_at(s["p"], d - datetime.timedelta(days=365)); l3 = lead_at(s["p"], d - datetime.timedelta(days=3 * 365))
    return dict(lead=s["lead"], tr1=(s["lead"] - l1) if l1 is not None else 0.0, tr3=((s["lead"] - l3) / 3) if l3 is not None else 0.0, age=s["age"])
def fx_drift(t): a = t["age"] - 25; return [1, a, max(a, 0) ** 2, max(t["age"] - 30, 0) ** 2, t["lead"], t["tr1"], t["tr3"]]
def fx_surv(t): a = t["age"] - 25; return [1, a, max(a, 0) ** 2, t["lead"]]
def fx_slam(t): return [1, t["lead"], max(t["lead"], 0) ** 2, t["tr1"], t["tr3"]]
# trend in the Slam chance: tested on 2009-2025 (trained to 2008): women slightly better (log-lik -276.6 -> -275.4; a
# 3-year rise means more Slams, a sudden 1-year jump slightly fewer), men unchanged (coefficients ~0)
def fit_logit_w(X, y, w):
    def nll(c):
        z = X @ c
        return (w * (np.logaddexp(0, z) - y * z)).sum() + 1e-3 * c[1:] @ c[1:], X.T @ (w * (1 / (1 + np.exp(-z)) - y)) + np.r_[0, 2e-3 * c[1:]]
    return minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
Xd, yd, Xs, ys, Xb, yb = [], [], [], [], [], []
for Y in range(FIRST_YEAR, last.year - 1):                      # transitions Y -> Y+1, both complete years
    d1 = datetime.date(Y + 1, 12, 31)
    for s_ in SNAP[Y]:
        t = state(s_, datetime.date(Y, 12, 31))
        Xb.append(fx_slam(t)); yb.append(sum(1 for x in SL[s_["p"]] if x.year == Y + 1) / 4)
        l1 = lead_at(s_["p"], d1)
        Xs.append(fx_surv(t)); ys.append(1.0 if l1 is not None else 0.0)
        if l1 is not None: Xd.append(fx_drift(t)); yd.append(l1 - t["lead"])
Xd, yd = np.array(Xd), np.array(yd); BD = np.linalg.lstsq(Xd, yd, rcond=None)[0]; SD = float(np.std(yd - Xd @ BD))
BS = fit_logit_w(np.array(Xs), np.array(ys), np.ones(len(ys))); BB = fit_logit_w(np.array(Xb), np.array(yb), np.full(len(yb), 4.0))
def simulate(t, n=3000, years=20, seed=0):
    rng = np.random.default_rng(seed)
    lead = np.full(n, t["lead"]); tr1 = np.full(n, t["tr1"])
    l3, l1 = t["lead"] - 3 * t["tr3"], t["lead"] - t["tr1"]
    hist = [np.full(n, l3), np.full(n, (l3 + l1) / 2), np.full(n, l1), lead.copy()]     # leads 3, 2, 1 and 0 years back
    alive = np.ones(n); tot = np.zeros(n); age = t["age"]
    for k in range(years):
        tr3 = (lead - hist[-4]) / 3
        p = 1 / (1 + np.exp(-(BB[0] + BB[1] * lead + BB[2] * np.maximum(lead, 0) ** 2 + BB[3] * tr1 + BB[4] * tr3)))
        tot += alive * rng.binomial(4, p)                        # the coming year's 4 Slams at the current level and trend
        a = age - 25
        sv = 1 / (1 + np.exp(-(BS[0] + BS[1] * a + BS[2] * max(a, 0) ** 2 + BS[3] * lead)))
        new = lead + BD[0] + BD[1] * a + BD[2] * max(a, 0) ** 2 + BD[3] * max(age - 30, 0) ** 2 + BD[4] * lead + BD[5] * tr1 + BD[6] * tr3 + rng.normal(0, SD, n)
        tr1 = new - lead; hist.append(new.copy()); lead = new
        alive = alive * (rng.random(n) < sv); age += 1
        if not alive.any(): break
    return float(tot.mean()), float((tot > 0).mean())

def next_year(t):
    """The coming 12 months: 4 Slams at the chance set by the current lead and its 1- and 3-year trend (no age)."""
    p = 1 / (1 + math.exp(-(BB[0] + BB[1] * t["lead"] + BB[2] * max(t["lead"], 0) ** 2 + BB[3] * t["tr1"] + BB[4] * t["tr3"])))
    return 4 * p, 1 - (1 - p) ** 4

# share of the next year's Slams won by the year-end top 150 (the rest: players outside it)
_won = sum(1 for Y in range(FIRST_YEAR, last.year - 1) for s_ in SNAP[Y] for x in SL[s_["p"]] if x.year == Y + 1)
_all = 4 * len(range(FIRST_YEAR, last.year - 1))
SHARE = _won / _all

def project(S, d, n):
    """per player: (expected further Slams, P(1+ more), expected in the next 12 months, P(1+ in the next 12 months)).
    The next 12 months are shared out over the field: the per-Slam chances are scaled so the top 150 together win
    4 x SHARE Slams (each player computed on their own would not add up to the 4 Slams there are)."""
    m_r, q_r = regress(S) if S else (np.zeros(0), np.zeros(0))
    ts = [state(s_, d) for s_ in S]; raw = [next_year(t)[0] for t in ts]
    scale = 4 * SHARE / sum(raw) if raw and sum(raw) > 0 else 1.0
    res = []
    for i, s_ in enumerate(S):
        t = ts[i]; m_s, q_s = simulate(t, n=n, seed=i)
        p_ = min(raw[i] * scale / 4, 0.95); n1, q1 = 4 * p_, 1 - (1 - p_) ** 4
        m, q = ((m_r[i] + m_s) / 2, (q_r[i] + q_s) / 2) if TOUR == "atp" else (m_s, q_s)
        res.append((max(m, n1), max(q, q1), n1, q1))
    return res

# players: [Slams won, expected further (total), P(1+ more), expected next 12 months, P(1+ next 12 months), expected after that]
out = {s_["p"]: [s_["won"], round(m, 2), round(q, 3), round(n1, 2), round(q1, 3), round(m - n1, 2)]
       for s_, (m, q, n1, q1) in zip(now, project(now, NOW_D, 3000))}
for s_ in snapshot(NOW_D, limit=100000):                         # outside the top 150: Slams won only
    out.setdefault(s_["p"], [s_["won"], None, None, None, None, None])
# the projection at every year end (top 150 at the time), for the player page: [year, won, expected further, P(1+ more)]
# (years inside the fitted data show what the model makes of that position, not a forecast made at the time)
hist = {}
for Y, S in SNAP.items():
    if Y < FIRST_YEAR or not S: continue
    for s_, (m, q, n1, q1) in zip(S, project(S, datetime.date(Y, 12, 31), 400)):
        hist.setdefault(s_["p"], []).append([Y, s_["won"], round(m, 2), round(q, 3), round(n1, 2)])
for s_ in now: hist.setdefault(s_["p"], []).append([last.year + 0.8, s_["won"], out[s_["p"]][1], out[s_["p"]][2], out[s_["p"]][3]])
json.dump(dict(asof=last.isoformat(), trained_to=last.year - 2, players=out, hist=hist), open(f"{SITE}/slamproj_{TOUR}.json", "w"), separators=(",", ":"))
top = sorted(((p, v) for p, v in out.items() if v[1] is not None), key=lambda kv: -kv[1][1])[:8]
print(f"{TOUR}: Slam projection ({'average of regression and simulation' if TOUR == 'atp' else 'simulation'}) for {len(now)} active players;",
      "; ".join(f"{p} {v[0]} won +{v[3]:.2f} next 12 months, +{v[5]:.1f} after" for p, v in top))
