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
import confounder_harness as H
from margin_model import parse_games
from sr_lib import hold, set_prob

SURF = ("Hard", "Clay", "Grass")
IDLE_ON = [True]


def replay(rows, age_at, init_rank, prior, a_s=0.006, a_r=0.006, b_s=0.99948, b_r=0.99948, blend=0.6,
           age_drift=(0.05, 0.01, 24, 30), young=1.4, closing=(0.005, 0.999), games_k=1.2, a_x=0.0, idle=True,
           fill=True, h2h=True, g_w=0.0, cal_close=True, lvl_scale=1.0, blend_s=None, blend_r=None, surf_k=1.0,
           venue_k=0.0, venue_b=1.0, gain_early=1.0, gain_settle=15.0):
    """a_x: CROSS update -- a player's serve surplus also moves her own return rating (and her return surplus her serve
    rating). a_x = 0: fully separate serve/return; a_x = a_s = a_r: equivalent to one overall rating. idle: layoff decay."""
    IDLE_ON[0] = idle
    bs_ = blend if blend_s is None else blend_s; br_ = blend if blend_r is None else blend_r
    sg = defaultdict(float); rg = defaultdict(float)
    ss = {x: defaultdict(float) for x in SURF}; rs = {x: defaultdict(float) for x in SURF}; ns = {x: defaultdict(int) for x in SURF}
    c = defaultdict(float); last = {}; seen = set(); mu = defaultdict(lambda: 0.25)
    hh = defaultdict(lambda: defaultdict(int)); ftab = {}; voff = defaultdict(float); nm = defaultdict(int)
    out = np.zeros(len(rows)); dsv = np.zeros(len(rows)); drt = np.zeros(len(rows)); lev = np.zeros(len(rows)); zraw = np.zeros(len(rows))
    dy, do_, ay, ao = age_drift
    def qmap(x, gsort, ssort):
        q = np.searchsorted(gsort, x) / len(gsort)
        return float(np.interp(q, (np.arange(len(ssort)) + 0.5) / len(ssort), ssort))
    def eff(p, surf, when):
        if surf in ss and ns[surf][p] >= 1:
            return bs_ * sg[p] + (1 - bs_) * ss[surf][p], br_ * rg[p] + (1 - br_) * rs[surf][p]
        if fill and surf in ss:
            ym = (when.year, when.month)
            if ftab.get("ym") != ym:                     # monthly pools: active players with a match on the surface
                ftab.clear(); ftab["ym"] = ym
                for x in SURF:
                    pool = [q for q, d in last.items() if ns[x][q] >= 1 and (when - d).days <= 365]
                    if len(pool) >= 30:
                        ftab[x] = tuple(np.sort(np.array([D[q] for q in pool])) for D in (sg, ss[x], rg, rs[x]))
            if surf in ftab:
                g1, s1, g2, s2 = ftab[surf]
                return bs_ * sg[p] + (1 - bs_) * qmap(sg[p], g1, s1), br_ * rg[p] + (1 - br_) * qmap(rg[p], g2, s2)
        return sg[p], rg[p]
    for i, r in enumerate(rows):
        a, b, when, surf, bo = r["a"], r["b"], r["when"], r["surf"], r["bo"]
        for p in (a, b):
            if p not in seen:
                seen.add(p); v = prior.get(r["level"], 0.0)
                rk = r["rank_a"] if p == a else r["rank_b"]
                if rk: v = min(max(init_rank[0] - init_rank[1] * (math.log(rk) - math.log(100.0)), -1.5), 1.0)
                v = v / 2.132 / 2 * lvl_scale
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
                    dr = (dy if ag < ay else (-do_ * (ag - ao) if ag > ao else 0.0)) * yrs / 2.132 / 2 * lvl_scale
                    sg[p] += dr; rg[p] += dr
                    for x in SURF: ss[x][p] += dr; rs[x][p] += dr
        last[a] = last[b] = when
        vk = (r.get("tname"), surf, r["indoor"]); m = mu[surf or "Hard"] + voff[vk]
        sa, ra = eff(a, surf, when); sb, rb = eff(b, surf, when)
        pa = min(max(S.sigmoid(m + sa - rb), 0.02), 0.98); pb = min(max(S.sigmoid(m + sb - ra), 0.02), 0.98)
        sp = 0.5 * (set_prob(round(pa, 3), round(pb, 3)) + 1 - set_prob(round(pb, 3), round(pa, 3)))
        sp = min(max(sp, 1e-6), 1 - 1e-6)
        qs = min(max(S.sigmoid(S.logit(sp) + c[a] - c[b]), 1e-6), 1 - 1e-6)
        out[i] = S.logit(min(max(S.match_from_set_prob(qs, bo), 1e-6), 1 - 1e-6))
        qraw = min(max(S.match_from_set_prob(sp, bo), 1e-6), 1 - 1e-6)
        if h2h:
            k = frozenset((a, b)); aw_, bw_ = hh[k][a], hh[k][b]
            if aw_ + bw_: out[i] += H.h2h_slope(aw_ + bw_) * S.logit((aw_ + 0.5) / (aw_ + bw_ + 1.0))
            hh[k][a] += 1
        dsv[i] = sa - sb; drt[i] = ra - rb; lev[i] = 2 * m + sa + sb - ra - rb; zraw[i] = S.logit(qraw)
        # ---- updates ----
        aa, ab = age_at(a, when), age_at(b, when)
        ma = young if (aa is not None and aa < 24) else 1.0; mb = young if (ab is not None and ab < 24) else 1.0
        if gain_early != 1.0:
            ma *= 1 + (gain_early - 1) * math.exp(-nm[a] / gain_settle); mb *= 1 + (gain_early - 1) * math.exp(-nm[b] / gain_settle)
        nm[a] += 1; nm[b] += 1
        pt = r["pts"]
        if pt:
            won, tot, sw, sn, rw, rn = pt
            ea = sw - sn * pa                     # a's serve points beyond expectation (b's return)
            eb = (rn - rw) - rn * pb              # b's serve points beyond expectation (a's return)
            mu[surf or "Hard"] += 0.0005 * ((sw + rn - rw) / (sn + rn) - S.sigmoid(m))
            if venue_k:                           # venue (court-speed) serve advantage, learned from past matches there
                voff[vk] = venue_b * voff[vk] + venue_k * (ea + eb) / (sn + rn)
            if g_w:                               # production trick (WTA): mix in the games surplus
                gw, gl = parse_games(r["score"])
                pg = 0.5 * (hold(pa) + 1 - hold(pb)); e = (gw - (gw + gl) * pg) * g_w
                ea += e / 2; eb -= e / 2
        else:
            gw, gl = parse_games(r["score"])
            pg = 0.5 * (hold(pa) + 1 - hold(pb)); e = (gw - (gw + gl) * pg) * games_k
            ea = e / 2; eb = -e / 2               # no split info: credit serve and return equally
        def upd(Sd, Rd, key):
            k_ = surf_k if key else 1.0
            # own serve performance ea (a) / eb (b); own return performance -eb (a) / -ea (b)
            Sd[a] = b_s * Sd[a] + k_ * ma * (a_s * ea - a_x * eb); Rd[a] = b_r * Rd[a] + k_ * ma * (-a_r * eb + a_x * ea)
            Sd[b] = b_s * Sd[b] + k_ * mb * (a_s * eb - a_x * ea); Rd[b] = b_r * Rd[b] + k_ * mb * (-a_r * ea + a_x * eb)
        upd(sg, rg, None)
        if surf in ss:
            upd(ss[surf], rs[surf], surf); ns[surf][a] += 1; ns[surf][b] += 1
        sets = S.parse_sets(r["score"])
        if sets and closing:
            n_set = sets[0] + sets[1]
            if cal_close:                         # production: closing surprise vs the calibrated set prob (no closing in it)
                qc = min(max(S.sigmoid(H.CAL_I + H.CAL_S * S.logit(qraw)), 1e-6), 1 - 1e-6)
                e_c = sets[0] - n_set * H.inv_match(round(qc, 4), bo)
            else:
                e_c = sets[0] - n_set * qs
            c[a] = closing[1] * c[a] + closing[0] * e_c; c[b] = closing[1] * c[b] - closing[0] * e_c
    return out, dict(sg=sg, rg=rg, ss=ss, rs=rs, ns=ns, c=c, mu=dict(mu), last=last, dsv=dsv, drt=drt, lev=lev, zraw=zraw)
