"""Projected future Grand Slam titles for the active players.   python3 slam_projection.py atp|wta  (after build_profiles.py)

Snapshots at every year end: each active top-150 player's rating relative to the field (rating minus the average of
No. 5-15), rating trend over the past 12 months, Slams won so far, age and years on tour -> Slams won AFTER the
snapshot (Poisson regression). Trained on snapshots up to 12 years ago (the outcome is then known for nearly every
career); checked on held-out years: trained on 1985-2008 and tested on 2009-2014, the model is far better than a flat
rate (deviance men 400 vs 1827) but too low overall, because careers got longer (Big 3, Serena), which is why the
final fit includes the most recent complete snapshots. Model per tour: the form that tested best (men: with a
squared term for a large lead; women: without years on tour).
Output site_v8/slamproj_{tour}.json: player -> [Slams won, expected further Slams, chance of at least one more
(logistic regression on the same inputs)]."""
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
SNAP = {Y: snapshot(datetime.date(Y, 12, 31)) for Y in range(FIRST_YEAR, last.year)}
train = [s for Y in range(FIRST_YEAR, last.year - 11) for s in SNAP[Y]]
Xtr = features(train); ytr = np.array([s["fut"] for s in train], float)
b = fit(Xtr, ytr); bl = fit_logit(Xtr, (ytr > 0).astype(float))
now = snapshot(last + datetime.timedelta(days=1)); Xn = features(now)
mu = np.exp(np.clip(Xn @ b, -20, 5)); p1 = 1 / (1 + np.exp(-(Xn @ bl)))
# a count spread more unevenly than a Poisson with the same mean always has P(>=1) <= 1 - exp(-mean): keeps the two
# numbers consistent where the models extrapolate (e.g. players far older than nearly everyone in the training data)
p1 = np.minimum(p1, 1 - np.exp(-mu))
out = {s["p"]: [s["won"], round(float(m), 2), round(float(q), 3)] for s, m, q in zip(now, mu, p1)}
for s in snapshot(last + datetime.timedelta(days=1), limit=100000):      # outside the top 150: Slams won only
    out.setdefault(s["p"], [s["won"], None, None])
# the projection at every year end (top 150 at the time), for the player page: [year, won, expected further, P(1+ more)]
# (years up to trained_to are inside the training data: there the line shows what the model makes of that position)
hist = {}
for Y, S in SNAP.items():
    if not S: continue
    Xs = features(S); m_ = np.exp(np.clip(Xs @ b, -20, 5)); q_ = np.minimum(1 / (1 + np.exp(-(Xs @ bl))), 1 - np.exp(-m_))
    for s_, m1, q1 in zip(S, m_, q_): hist.setdefault(s_["p"], []).append([Y, s_["won"], round(float(m1), 2), round(float(q1), 3)])
for s_, m1, q1 in zip(now, mu, p1): hist.setdefault(s_["p"], []).append([last.year + 0.8, s_["won"], round(float(m1), 2), round(float(q1), 3)])
json.dump(dict(asof=last.isoformat(), trained_to=last.year - 12, players=out, hist=hist), open(f"{SITE}/slamproj_{TOUR}.json", "w"), separators=(",", ":"))
top = sorted(((p, v) for p, v in out.items() if v[1] is not None), key=lambda kv: -kv[1][1])[:8]
print(f"{TOUR}: Slam projection from {len(train)} snapshots ({FIRST_YEAR}-{last.year - 12}) for {len(now)} active players;",
      "; ".join(f"{p} {v[0]} won +{v[1]:.1f}" for p, v in top))
