"""Serve/return split model (see serve_return_split.py for the test). Two ratings per player on the point-logit
scale; service games -> sets (point-level tiebreak) -> match. Used in production (v21g) as one extra input:
its own match logit, next to the main rating model."""
import math
from collections import defaultdict
from functools import lru_cache
import numpy as np
import score_driven as S
from margin_model import parse_games


def hold(p):
    q = 1 - p
    return p**4 * (1 + 4*q + 10*q*q) + 20 * p**3 * q**3 * p * p / (1 - 2*p*q)


@lru_cache(maxsize=None)
def set_prob(pa, pb):
    """A's chance to win a set; pa / pb = point-win prob of A / B on their own serve. A serves first."""
    ha, hb = hold(pa), hold(pb)
    @lru_cache(maxsize=None)
    def tb(x, y):
        if x >= 7 and x - y >= 2: return 1.0
        if y >= 7 and y - x >= 2: return 0.0
        if x == y and x >= 6:
            w2 = pa * (1 - pb); l2 = (1 - pa) * pb
            return w2 / (w2 + l2) if w2 + l2 > 0 else 0.5
        k = x + y; a_serves = (k % 4 == 0) or (k % 4 == 3)
        p = pa if a_serves else 1 - pb
        return p * tb(x + 1, y) + (1 - p) * tb(x, y + 1)
    @lru_cache(maxsize=None)
    def g(x, y):
        if x >= 6 and x - y >= 2: return 1.0
        if y >= 6 and y - x >= 2: return 0.0
        if x == 7 or y == 7: return 1.0 if x > y else 0.0
        if x == 6 and y == 6: return tb(0, 0)
        p = ha if (x + y) % 2 == 0 else 1 - hb
        return p * g(x + 1, y) + (1 - p) * g(x, y + 1)
    return g(0, 0)


def match_logit(pa, pb, bo):
    pa = min(max(pa, 0.02), 0.98); pb = min(max(pb, 0.02), 0.98)
    s = 0.5 * (set_prob(round(pa, 3), round(pb, 3)) + 1 - set_prob(round(pb, 3), round(pa, 3)))
    qm = min(max(S.match_from_set_prob(min(max(s, 1e-6), 1 - 1e-6), bo), 1e-6), 1 - 1e-6)
    return S.logit(qm)


def replay(rows, alpha, age_at, init_rank, prior):
    """rows: the model's scored rows in order. Returns per-row pre-match split logit (winner's view) and the end state."""
    sv = defaultdict(float); rt = defaultdict(float); last = {}; seen = set()
    mu = defaultdict(lambda: 0.25)
    out = np.zeros(len(rows)); dsv = np.zeros(len(rows)); drt = np.zeros(len(rows))
    for i, r in enumerate(rows):
        a, b, when, surf, bo = r["a"], r["b"], r["when"], r["surf"] or "Hard", r["bo"]
        for p in (a, b):
            if p not in seen:
                seen.add(p); v = prior.get(r["level"], 0.0)
                rk = r["rank_a"] if p == a else r["rank_b"]
                if rk: v = min(max(init_rank[0] - init_rank[1] * (math.log(rk) - math.log(100.0)), -1.5), 1.0)
                sv[p] = rt[p] = v / 2.132 / 2
            if p in last:
                idle = max(0, (when - last[p]).days - 30)
                if idle: f = 0.9995 ** idle; sv[p] *= f; rt[p] *= f
        last[a] = last[b] = when
        m = mu[surf]
        pa = S.sigmoid(m + sv[a] - rt[b]); pb = S.sigmoid(m + sv[b] - rt[a])
        out[i] = match_logit(pa, pb, bo); dsv[i] = sv[a] - sv[b]; drt[i] = rt[a] - rt[b]
        pa = min(max(pa, 0.02), 0.98); pb = min(max(pb, 0.02), 0.98)
        aa, ab = age_at(a, when), age_at(b, when)
        ka = alpha * (1.4 if (aa is not None and aa < 24) else 1.0); kb = alpha * (1.4 if (ab is not None and ab < 24) else 1.0)
        pt = r["pts"]
        if pt:
            won, tot, sw, sn, rw, rn = pt
            ea = sw - sn * pa; eb = (rn - rw) - rn * pb
            sv[a] += ka * ea; rt[b] -= kb * ea; sv[b] += kb * eb; rt[a] -= ka * eb
            mu[surf] += 0.0005 * ((sw + rn - rw) / (sn + rn) - S.sigmoid(m))
        else:
            gw, gl = parse_games(r["score"])
            pg = 0.5 * (hold(pa) + 1 - hold(pb)); e = (gw - (gw + gl) * pg) * 1.2
            sv[a] += ka * e / 2; rt[a] += ka * e / 2; sv[b] -= kb * e / 2; rt[b] -= kb * e / 2
    return out, dict(sv=sv, rt=rt, mu=dict(mu), last=last, dsv=dsv, drt=drt)
