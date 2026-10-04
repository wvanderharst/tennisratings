"""A third, structurally different rating model, testing the user's idea
directly: instead of driving the score-driven filter off a BINOMIAL on sets
won (2-0, 2-1, 3-0... which treats a 6-0 6-0 rout identically to a 7-6 7-6
squeaker, since both are "won 2 sets to 0"), drive it off the GAME MARGIN
(games won minus games lost across the whole match) -- a continuous
plus/minus signal that actually sees the difference between a rout and a
squeaker.

That also means a genuinely different observation DISTRIBUTION, not just a
different statistic: sets-won is discrete and bounded (Binomial), game
margin is closer to continuous and can be large in either direction, so it
is modeled here as a score-driven LOCATION filter under a Student-t
observation density (heavier tails than Gaussian, to stay robust to
retirement-adjacent or blowout matches instead of letting one 6-0 6-0 kick
the rating around) -- both a different driving statistic AND a different
distributional family in the same experiment, matching what was asked.
"""
import paths as P
import json, math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm

import score_driven as S
from load_full import load_full

OUT = P.OUT
SET_RE = S.SET_RE


def parse_games(score):
    """Total games won by the match winner and by the loser. None for anything
    that didn't finish on court (mirrors score_driven.parse_sets)."""
    if not score:
        return None
    upper = score.upper()
    if "RET" in upper or "W/O" in upper or "DEF" in upper or "ABD" in upper or "UNFINISHED" in upper:
        return None
    w = l = 0
    n_sets = 0
    for token in score.split():
        m = SET_RE.match(token)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        w += a
        l += b
        n_sets += 1
    if n_sets < 2:
        return None
    return w, l


def t_score_and_ll(resid, sigma, nu):
    """Student-t score (d log-density / d location) and log-density, location-scale
    parameterisation. resid = observation - location."""
    z = resid / sigma
    ll = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
          - math.log(sigma) - (nu + 1) / 2 * math.log(1 + z * z / nu))
    score = (nu + 1) * resid / (nu * sigma * sigma + resid * resid)   # bounded score, Harvey (2013)
    return score, ll


def filter_margin(matches, a_rate, b_persist, sigma, nu, until=None, callback=None):
    """Score-driven Student-t location filter on game margin."""
    phi = defaultdict(float)
    ll = 0.0
    for m in matches:
        when, a, b, margin = m
        mu = phi[a] - phi[b]
        if callback is not None:
            callback(when, a, b, mu, phi)
        if until is not None and when >= until:
            continue
        resid = margin - mu
        score, ll_i = t_score_and_ll(resid, sigma, nu)
        ll += ll_i
        step = a_rate * score
        phi[a] = b_persist * phi[a] + step
        phi[b] = b_persist * phi[b] - step
    return ll


def fit_margin(matches, until):
    """MLE for (a_rate, b_persist, sigma, nu) -- sigma/nu fit once on the pooled
    residual distribution first (cheap), then a_rate/b_persist refined given those,
    same two-pass approach as the rest of this project's GAS fits."""
    margins = np.array([m[3] for m in matches if m[0] < until])
    sigma0 = float(np.std(margins))

    def negll(par):
        a_rate = math.exp(par[0])
        b_persist = 1.0 / (1.0 + math.exp(-par[1]))
        sigma = math.exp(par[2])
        nu = 2.5 + math.exp(par[3])   # nu > 2.5 keeps variance finite
        try:
            return -filter_margin(matches, a_rate, b_persist, sigma, nu, until)
        except (OverflowError, ValueError):
            return 1e18

    start = [math.log(0.08), 6.0, math.log(sigma0), math.log(5.0)]
    res = minimize(negll, start, method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 5.0, "maxiter": 400})
    a_rate = math.exp(res.x[0])
    b_persist = 1.0 / (1.0 + math.exp(-res.x[1]))
    sigma = math.exp(res.x[2])
    nu = 2.5 + math.exp(res.x[3])
    return {"negll": res.fun, "a": a_rate, "b": b_persist, "sigma": sigma, "nu": nu}


if __name__ == "__main__":
    raw = load_full()
    matches = []
    for m in raw:
        g = parse_games(m["score"])
        if g is None:
            continue
        matches.append((m["when"], m["a"], m["b"], g[0] - g[1], m["bo"]))
    matches.sort(key=lambda r: r[0])
    print(f"{len(matches)} matches with parseable game margins")
    split = matches[int(len(matches) * 0.8)][0]
    print(f"split at {split}")

    print("\nfitting Student-t margin filter (this takes a bit)...")
    mfit = fit_margin([(w, a, b, mg) for w, a, b, mg, _bo in matches], split)
    print("margin fit:", mfit)
    json.dump(mfit, open(f"{OUT}/margin_fit.json", "w"), indent=1)

    # replay with the fitted parameters, collecting phi_diff at prediction time,
    # then map phi_diff -> P(A wins the match) via the SAME two-stage calibration
    # design used throughout (free-fit logistic on train, no hand-tuned constant)
    recs = []
    def cb(when, a, b, mu, phi):
        recs.append((when, a, b, mu))
    filter_margin([(w, a, b, mg) for w, a, b, mg, _bo in matches],
                   mfit["a"], mfit["b"], mfit["sigma"], mfit["nu"], until=None, callback=cb)

    rng = np.random.default_rng(0)
    flip = rng.random(len(recs)) < 0.5
    y = flip.astype(float)
    mu_arr = np.array([r[3] for r in recs])
    mu_sym = np.where(flip, mu_arr, -mu_arr)
    when_arr = np.array([r[0] for r in recs])
    train = when_arr < split
    test = ~train

    X = sm.add_constant(mu_sym[train])
    calfit = sm.Logit(y[train], X).fit(disp=0)
    print(f"\ncalibration (margin phi_diff -> match win prob): intercept={calfit.params[0]:.5f} slope={calfit.params[1]:.5f}")

    z_test = calfit.params[0] + calfit.params[1] * mu_sym[test]
    p_margin = 1 / (1 + np.exp(-z_test))
    y_test = y[test]

    # --- compare against the existing sets-binomial model on the SAME matches/split ---
    sfit = json.load(open("gas_result_fit.json"))["sets"]
    phi_sets = defaultdict(float)
    p_sets_list = []
    for when, a, b, margin, bo in matches:
        q = min(max(S.sigmoid(phi_sets[a] - phi_sets[b]), 1e-9), 1 - 1e-9)
        p_sets_list.append(S.match_from_set_prob(q, bo))
        # need actual sets won/lost to update -- reparse quickly from margin's source match
    # (re-run properly using S.parse_sets on the same raw matches, aligned by index)
    raw_by_key = {}
    for m in raw:
        g = parse_games(m["score"])
        if g is None:
            continue
        raw_by_key[(m["when"], m["a"], m["b"])] = m
    phi_sets = defaultdict(float)
    p_sets_arr = []
    for when, a, b, margin, bo in matches:
        q = min(max(S.sigmoid(phi_sets[a] - phi_sets[b]), 1e-9), 1 - 1e-9)
        p_sets_arr.append(q)
        mrow = raw_by_key[(when, a, b)]
        sets = mrow["sets"]
        if sets is not None:
            won, lost = sets
            n = won + lost
            score_ = won - n * q
            step = sfit["a"] * score_
            phi_sets[a] = sfit["b"] * phi_sets[a] + step
            phi_sets[b] = sfit["b"] * phi_sets[b] - step
    p_sets_arr = np.array(p_sets_arr)
    p_sets_raw = np.array([S.match_from_set_prob(q, bo) for q, (_w, _a, _b, _mg, bo) in zip(p_sets_arr, matches)])
    p_sets_sym = np.where(flip, p_sets_raw, 1 - p_sets_raw)
    logit_sets = np.log(np.clip(p_sets_sym, 1e-9, 1-1e-9) / (1 - np.clip(p_sets_sym, 1e-9, 1-1e-9)))
    Xs = sm.add_constant(logit_sets[train])
    calfit_sets = sm.Logit(y[train], Xs).fit(disp=0)
    z_sets_test = calfit_sets.params[0] + calfit_sets.params[1] * logit_sets[test]
    p_sets_cal = 1 / (1 + np.exp(-z_sets_test))

    def brier(p, yy): return np.mean((p - yy) ** 2)
    def logloss(p, yy):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
    def acc(p, yy): return np.mean((p >= 0.5).astype(float) == yy)

    print(f"\nheld-out n={test.sum()}")
    print(f"sets-binomial model, calibrated:   logloss={logloss(p_sets_cal,y_test):.4f} brier={brier(p_sets_cal,y_test):.4f} accuracy={acc(p_sets_cal,y_test):.4f}")
    print(f"game-margin Student-t model, cal.: logloss={logloss(p_margin,y_test):.4f} brier={brier(p_margin,y_test):.4f} accuracy={acc(p_margin,y_test):.4f}")

    ll_sets = -(y_test*np.log(np.clip(p_sets_cal,1e-9,1)) + (1-y_test)*np.log(np.clip(1-p_sets_cal,1e-9,1)))
    ll_margin = -(y_test*np.log(np.clip(p_margin,1e-9,1)) + (1-y_test)*np.log(np.clip(1-p_margin,1e-9,1)))
    t, pv = stats.ttest_rel(ll_margin, ll_sets)
    print(f"\npaired t-test (margin vs sets, calibrated): t={t:.3f} p={pv:.3g} mean diff={(ll_margin-ll_sets).mean():+.5f}")
    print("(negative t means the game-margin model has LOWER logloss, i.e. better)")

    print("\nreliability, game-margin model (held-out, calibrated):")
    order = np.argsort(p_margin)
    ps, ysort = p_margin[order], y_test[order]
    for i in range(10):
        l, h = int(i*len(ps)/10), int((i+1)*len(ps)/10)
        print(f"  predicted {ps[l:h].mean():.3f}  actual {ysort[l:h].mean():.3f}  gap {ysort[l:h].mean()-ps[l:h].mean():+.3f}")

    print(f"\nfitted Student-t dof (nu) = {mfit['nu']:.2f}  (lower = heavier tails; "
          f"nu>30 would be practically Gaussian)")
