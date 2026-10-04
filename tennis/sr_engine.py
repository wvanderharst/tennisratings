"""Serve/return rating ENGINE -- Koopman & Lit (2019, IJF) attack/defence score-driven model, adapted to tennis,
with the production rating machinery built in.

States per player (point-logit scale): serve s and return r, each = general part + surface part
  (effective = BLEND_W * general + (1-BLEND_W) * surface, once the player has a match on that surface).
Observation: points won on A's serve ~ Binomial(n_A, sigmoid(mu_surf + s_A - r_B)), and the same for B's serve
  (the tennis analogue of Koopman-Lit's bivariate Poisson goals with intensities exp(delta + alpha_i - beta_j)).
Score-driven update with unit scaling (as in Koopman-Lit): s_A += a_s * (won - n p), r_B -= a_r * (won - n p);
  persistence b per match (their B matrix), separate a, b for serve and return.
Same housekeeping as the production rating: debut from ranking / level prior, idle decay (0.9995 per day past 30),
  age drift (+0.05/yr <24, -0.01*(age-30)/yr >30, in game-logit units -> split over serve and return),
  young boost 1.4 (<24), games fallback for matches without serve stats, closing (set-level) rating.
Prediction: point probs -> hold -> sets with point-level tiebreak (+ closing on the set logit) -> match logit."""
import math
from collections import defaultdict
import numpy as np
import score_driven as S
from margin_model import parse_games
from sr_lib import hold, set_prob

SURF = ("Hard", "Clay", "Grass")
IDLE_ON = [True]


def replay(rows, age_at, init_rank, prior, a_s=0.006, a_r=0.006, b_s=0.99948, b_r=0.99948, blend=0.6,
           age_drift=(0.05, 0.01, 24, 30), young=1.4, closing=(0.005, 0.999), games_k=1.2, a_x=0.0, idle=True):
    """a_x: CROSS update -- a player's serve surplus also moves her own return rating (and her return surplus her serve
    rating). a_x = 0: fully separate serve/return; a_x = a_s = a_r: equivalent to one overall rating. idle: layoff decay."""
    IDLE_ON[0] = idle
    sg = defaultdict(float); rg = defaultdict(float)
    ss = {x: defaultdict(float) for x in SURF}; rs = {x: defaultdict(float) for x in SURF}; ns = {x: defaultdict(int) for x in SURF}
    c = defaultdict(float); last = {}; seen = set(); mu = defaultdict(lambda: 0.25)
    out = np.zeros(len(rows)); dsv = np.zeros(len(rows)); drt = np.zeros(len(rows))
    dy, do_, ay, ao = age_drift
    def eff(p, surf):
        if surf in ss and ns[surf][p] >= 1:
            return blend * sg[p] + (1 - blend) * ss[surf][p], blend * rg[p] + (1 - blend) * rs[surf][p]
        return sg[p], rg[p]
    for i, r in enumerate(rows):
        a, b, when, surf, bo = r["a"], r["b"], r["when"], r["surf"], r["bo"]
        for p in (a, b):
            if p not in seen:
                seen.add(p); v = prior.get(r["level"], 0.0)
                rk = r["rank_a"] if p == a else r["rank_b"]
                if rk: v = min(max(init_rank[0] - init_rank[1] * (math.log(rk) - math.log(100.0)), -1.5), 1.0)
                v = v / 2.132 / 2
                sg[p] = rg[p] = v
                for x in SURF: ss[x][p] = rs[x][p] = v
            elif p in last:
                days = (when - last[p]).days
                idle = max(0, days - 30)
                if idle_days := (idle if IDLE_ON[0] else 0):
                    f = 0.9995 ** idle_days; sg[p] *= f; rg[p] *= f; c[p] *= f
                    for x in SURF: ss[x][p] *= f; rs[x][p] *= f
                ag = age_at(p, when); yrs = min(days / 365.25, 1.0)
                if ag is not None and yrs > 0:
                    dr = (dy if ag < ay else (-do_ * (ag - ao) if ag > ao else 0.0)) * yrs / 2.132 / 2
                    sg[p] += dr; rg[p] += dr
                    for x in SURF: ss[x][p] += dr; rs[x][p] += dr
        last[a] = last[b] = when
        m = mu[surf or "Hard"]
        sa, ra = eff(a, surf); sb, rb = eff(b, surf)
        pa = min(max(S.sigmoid(m + sa - rb), 0.02), 0.98); pb = min(max(S.sigmoid(m + sb - ra), 0.02), 0.98)
        sp = 0.5 * (set_prob(round(pa, 3), round(pb, 3)) + 1 - set_prob(round(pb, 3), round(pa, 3)))
        sp = min(max(sp, 1e-6), 1 - 1e-6)
        qs = min(max(S.sigmoid(S.logit(sp) + c[a] - c[b]), 1e-6), 1 - 1e-6)
        out[i] = S.logit(min(max(S.match_from_set_prob(qs, bo), 1e-6), 1 - 1e-6))
        dsv[i] = sa - sb; drt[i] = ra - rb
        # ---- updates ----
        aa, ab = age_at(a, when), age_at(b, when)
        ma = young if (aa is not None and aa < 24) else 1.0; mb = young if (ab is not None and ab < 24) else 1.0
        pt = r["pts"]
        if pt:
            won, tot, sw, sn, rw, rn = pt
            ea = sw - sn * pa                     # a's serve points beyond expectation (b's return)
            eb = (rn - rw) - rn * pb              # b's serve points beyond expectation (a's return)
            mu[surf or "Hard"] += 0.0005 * ((sw + rn - rw) / (sn + rn) - S.sigmoid(m))
        else:
            gw, gl = parse_games(r["score"])
            pg = 0.5 * (hold(pa) + 1 - hold(pb)); e = (gw - (gw + gl) * pg) * games_k
            ea = e / 2; eb = -e / 2               # no split info: credit serve and return equally
        def upd(Sd, Rd, key):
            # own serve performance ea (a) / eb (b); own return performance -eb (a) / -ea (b)
            Sd[a] = b_s * Sd[a] + ma * (a_s * ea - a_x * eb); Rd[a] = b_r * Rd[a] + ma * (-a_r * eb + a_x * ea)
            Sd[b] = b_s * Sd[b] + mb * (a_s * eb - a_x * ea); Rd[b] = b_r * Rd[b] + mb * (-a_r * ea + a_x * eb)
        upd(sg, rg, None)
        if surf in ss:
            upd(ss[surf], rs[surf], surf); ns[surf][a] += 1; ns[surf][b] += 1
        sets = S.parse_sets(r["score"])
        if sets and closing:
            n_set = sets[0] + sets[1]; e_c = sets[0] - n_set * qs
            c[a] = closing[1] * c[a] + closing[0] * e_c; c[b] = closing[1] * c[b] - closing[0] * e_c
    return out, dict(sg=sg, rg=rg, ss=ss, rs=rs, ns=ns, c=c, mu=dict(mu), last=last, dsv=dsv, drt=drt)
