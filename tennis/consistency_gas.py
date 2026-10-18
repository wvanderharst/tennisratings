"""Consistency: each player's match-to-match swing, as a score-driven (GAS) model for the variance.

    python3 consistency_gas.py atp|wta          -> site_v8/cons_{tour}.json   (after build_profiles.py)
    python3 consistency_gas.py atp|wta --fit    re-estimate the parameters below (matches before 2016) and test after

Per player, match by match: e = performance rating - rating going in, n = PT_M / slope (the part of the spread that
comes from the number of points alone). e ~ Normal(0, h) with h = exp(f) + n, and f, the log of the player's own
variance, moves with the scaled score of that likelihood:
    f <- w + b (f - w) + a (e^2/h - 1) exp(f)/h + d log(1 + days to next match / 30)
New players start at f0 (fitted separately from the long-run level w). Fitted out of sample: the swing is a slow, personal trait (b ~ 0.999).
Students-t tails did not help (nu -> infinity), layoffs barely matter.

In the match odds the two players' variances widen or narrow the odds: z -> z / sqrt(1 + K (Va + Vb - 2 V0) / (2 V0)),
V0 = the average variance of the active players.
K = 1 would be the full theoretical effect; out of sample the best weight is ~0 (men -0.07, women +0.17) and K = 1
makes the odds worse (log loss 0.6160 -> 0.6174, men 2016+), so a small K = 0.25 is used (no measurable cost)."""
import paths as P
import glob, json, math, sys, datetime
import numpy as np

TOUR = sys.argv[1]
PT_M = 2.132
PARAMS = {"atp": dict(w=-1.72286, a=0.01396, b=0.99939, d=-0.00404, f0=-1.72286),
          "wta": dict(w=-1.73503, a=0.01360, b=0.99979, d=-0.00236, f0=-1.73503)}
K = 0.25
SITE = f"{P.ROOT}/site_v8"
ordn = lambda s: datetime.date.fromisoformat(s).toordinal()


def load():
    rows = {}
    for f in sorted(glob.glob(f"{SITE}/prof_{TOUR}_[0-9]*.json")): rows.update(json.load(open(f)))
    names = sorted(rows); T = max(len(rows[p]) for p in names)
    E = np.full((len(names), T), np.nan); N = np.full_like(E, np.nan); G = np.zeros_like(E); Y = np.zeros(E.shape, int)
    for i, p in enumerate(names):
        L = rows[p]
        for k, r in enumerate(L):
            Y[i, k] = int(r[0][:4])
            if k + 1 < len(L): G[i, k] = max(0, ordn(L[k + 1][0]) - ordn(r[0]))
            if r[19] is not None and r[9] is not None and r[21] and r[21] > 0:   # performance, rating in, points slope
                E[i, k] = r[19] - r[9]; N[i, k] = PT_M / r[21]
    return rows, names, E, N, G, Y


def run(par, E, N, G):
    """Filter all players at once (one column = each player's k-th match). Returns per-match log-lik and f before
    every match, and f after each player's last match."""
    w, a, b, d = par["w"], par["a"], par["b"], par["d"]
    Pn, T = E.shape; f = np.full(Pn, par.get("f0", w)); ll = np.zeros((Pn, T)); F = np.zeros((Pn, T))
    for k in range(T):
        F[:, k] = f
        e, n = E[:, k], N[:, k]; ok = ~np.isnan(e)
        h = np.exp(f) + np.where(ok, n, 0); e2 = np.where(ok, e * e, 0)
        ll[:, k] = np.where(ok, -0.5 * (np.log(2 * np.pi * h) + e2 / h), 0)
        s = np.where(ok, (e2 / h - 1) * np.exp(f) / h, 0)
        f = np.clip(w + b * (f - w) + a * s + d * np.log1p(G[:, k] / 30), -6, 2)
    return ll, F, f


def fit(E, N, G, Y):
    from scipy.optimize import minimize
    ok = ~np.isnan(E); tr, te = ok & (Y < 2016), ok & (Y >= 2016)
    unpack = lambda q: dict(w=q[0], a=q[1], b=q[2], d=q[3], f0=q[4])
    c = minimize(lambda q: -run(unpack([q[0], 0, 0, 0, q[0]]), E, N, G)[0][tr].mean(), [-1.5], method="Nelder-Mead").x[0]
    q = minimize(lambda q: -run(unpack(q), E, N, G)[0][tr].mean(), [PARAMS[TOUR][k] for k in ("w", "a", "b", "d", "f0")], method="Nelder-Mead",
                 options=dict(maxiter=900, xatol=1e-5, fatol=1e-8)).x
    for lab, par in (("constant", unpack([c, 0, 0, 0, c])), ("GAS", unpack(q))):
        print(f"  {lab:8s} {dict((k, round(v, 5)) for k, v in par.items())}  test log-lik/match {run(par, E, N, G)[0][te].mean():.5f}")


def main():
    rows, names, E, N, G, Y = load()
    if "--fit" in sys.argv:
        fit(E, N, G, Y); return
    par = PARAMS[TOUR]
    _, F, f_end = run(par, E, N, G)
    v0 = math.exp(par["w"])
    last_all = max(r[0] for L in rows.values() for r in L)
    cut = (datetime.date.fromisoformat(last_all) - datetime.timedelta(days=365)).isoformat()
    out, active = {}, []
    for i, p in enumerate(names):
        L = rows[p]; n = int((~np.isnan(E[i, :len(L)])).sum())
        if n < 10: continue
        hist = []                                   # swing at the end of each season (after its last match)
        for k, r in enumerate(L):
            if k + 1 == len(L) or L[k + 1][0][:4] != r[0][:4]:
                fk = F[i, k + 1] if k + 1 < len(L) else f_end[i]
                hist.append([int(r[0][:4]), round(math.exp(fk / 2), 3)])
        sd = round(math.exp(f_end[i] / 2), 3)
        out[p] = [sd, n, None, hist]
        if L[-1][0] >= cut and n >= 30: active.append((sd, p))
    active.sort()
    for j, (sd, p) in enumerate(active):            # share of active players with a larger swing
        out[p][2] = round(1 - (j + 0.5) / len(active), 3)
    # reference for the odds and the display: the average variance of the active players (so on average the
    # consistency term in the odds is zero; exp(w) is the model's long-run level, above most established players)
    vref = float(np.mean([sd * sd for sd, p in active])) if active else v0
    json.dump(dict(k=K, v0=round(vref, 5), sd0=round(math.sqrt(vref), 3), n_active=len(active), players=out),
              open(f"{SITE}/cons_{TOUR}.json", "w"), separators=(",", ":"))
    print(f"{TOUR}: consistency for {len(out)} players ({len(active)} active); typical swing of active players {math.sqrt(vref):.3f};",
          "most consistent:", ", ".join(f"{p} {sd:.2f}" for sd, p in active[:3]), "| most erratic:", ", ".join(f"{p} {sd:.2f}" for sd, p in active[-3:]))


if __name__ == "__main__":
    main()
