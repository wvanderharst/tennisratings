"""Production main rating rebuilt from scratch, with switches that morph it step by step into the serve/return model.

All ratings in production GAME-logit units. Point scale = rating / PT_M.
  pred   "game": production prediction  (q_game = sigmoid(gap) -> iid games -> set -> match, CAL, h2h)
         "hold": point model             (pa = sigmoid(mu + (s_a - r_b)/PT_M), pb likewise -> holds -> sets (point
                 tiebreak) -> match, CAL, h2h)
  resid  "pooled": production surprise   (all points won vs sigmoid(gap/PT_M))
         "serve":  per-serve surprise    (a's serve points vs pa, a's return points vs 1 - pb)
  split  False: one rating (s = r = phi);  True: separate serve and return ratings
         (single-rating update of size u becomes serve 2*u_serve, return 2*u_return, so the SUM moves the same)
"""
import math
from collections import defaultdict
import numpy as np
import score_driven as S
import confounder_harness as H
from margin_model import parse_games
import sr_lib

SURF = ("Hard", "Clay", "Grass")


def replay(rows, age_at, init_rank, init_prior, age_drift=(0.05, 0.01, 24, 30), gscale=0.8, pt_lam=1.0, pt_c=0.7, pt_m=2.132,
           pred="game", resid="pooled", split=False, k_s=1.0, k_r=1.0, blend_s=0.6, blend_r=0.6, gain_early=1.0, gain_settle=15.0,
           gb=None, ga=None, mu_k=0.0005, h2h=True, hm=1.0, res_hm=False, k_d=1.0, hook=None, surf_k=1.0, surf_early=1.0, surf_settle=10.0,
           gap=False, lam=0.4, gwt=(0.6, 0.3, 0.1), clamp=None, hook_post=None, k_ret=1.0, ret_side='both'):
    GA = H.ga_rate if ga is None else ga; GB = H.gb_persist if gb is None else gb
    # state: serve & return (identical when split=False), general + per surface
    sg = defaultdict(float); rg = defaultdict(float)
    ss = {x: defaultdict(float) for x in SURF}; rs = {x: defaultdict(float) for x in SURF}; ns = {x: defaultdict(int) for x in SURF}
    c = defaultdict(float); last = {}; seen = set(); hh = defaultdict(lambda: defaultdict(int)); ftab = {}
    mu = defaultdict(lambda: 0.55); nm = defaultdict(int)
    n = len(rows); S4 = np.zeros((n, 5)); out = np.zeros(n); alt = np.zeros(n); dsv = np.zeros(n); drt = np.zeros(n)
    dy, do_, ay, ao = age_drift

    def qmap(x, gsort, ssort):
        q = np.searchsorted(gsort, x) / len(gsort)
        return float(np.interp(q, (np.arange(len(ssort)) + 0.5) / len(ssort), ssort))

    for i, r in enumerate(rows):
        a, b, when, surf, bo = r["a"], r["b"], r["when"], r["surf"], r["bo"]
        for p in (a, b):
            if p not in seen:
                seen.add(p); v = init_prior.get(r["level"], 0.0)
                rk = r["rank_a"] if p == a else r["rank_b"]
                if rk: v = min(max(init_rank[0] - init_rank[1] * (math.log(rk) - math.log(100.0)), -1.5), 1.0)
                sg[p] = rg[p] = v
                for x in SURF: ss[x][p] = rs[x][p] = (0.0 if gap else v)   # gap mode: surface gaps start at 0
        for p in (a, b):
            if p in last:
                idle = max(0, (when - last[p]).days - H.IDLE_GRACE_DAYS)
                if idle > 0:
                    f = H.IDLE_GB ** idle; sg[p] *= f; rg[p] *= f; c[p] *= f
                    for x in SURF: ss[x][p] *= f; rs[x][p] *= f
        for p in (a, b):
            if p in last:
                yrs = min((when - last[p]).days / 365.25, 1.0); ag = age_at(p, when)
                if ag is not None and yrs > 0:
                    dr = (dy if ag < ay else (-do_ * (ag - ao) if ag > ao else 0.0)) * yrs
                    sg[p] = min(max(sg[p] + dr, -3), 3); rg[p] = min(max(rg[p] + dr, -3), 3)
                    if not gap:
                        for x in SURF: ss[x][p] = min(max(ss[x][p] + dr, -3), 3); rs[x][p] = min(max(rs[x][p] + dr, -3), 3)
        if hook is not None: hook(i, r, dict(sg=sg, rg=rg, ss=ss, rs=rs, ns=ns, c=c, last=last, mu=mu))
        last[a] = last[b] = when

        # ---- effective (blended, rank-filled) serve and return ratings ----
        use = gap or (surf in SURF and ns[surf][a] >= 1 and ns[surf][b] >= 1)
        if surf in SURF and not use:
            ym = (when.year, when.month)
            if ftab.get("ym") != ym:
                ftab.clear(); ftab["ym"] = ym
                for x in SURF:
                    pool = [q for q, d in last.items() if ns[x][q] >= 1 and (when - d).days <= 365]
                    if len(pool) >= 30:
                        ftab[x] = tuple(np.sort(np.array([D[q] for q in pool])) for D in (sg, ss[x], rg, rs[x]))
        def eff(p):
            if gap:      # surface skill = general + gap; the prediction uses general + lam * gap
                return (sg[p] + lam * ss[surf][p], rg[p] + lam * rs[surf][p]) if surf in SURF else (sg[p], rg[p])
            if surf not in SURF or (not use and surf not in ftab): return sg[p], rg[p]
            if ns[surf][p] >= 1: s_, r_ = ss[surf][p], rs[surf][p]
            elif surf in ftab:
                g1, s1, g2, s2 = ftab[surf]
                s_ = qmap(sg[p], g1, s1); r_ = qmap(rg[p], g2, s2) if split else s_
            else: return sg[p], rg[p]
            return blend_s * sg[p] + (1 - blend_s) * s_, blend_r * rg[p] + (1 - blend_r) * r_
        sa, ra = eff(a); sb, rb = eff(b)
        if not split:                              # one rating: serve and return identical by construction
            ra, rb = sa, sb
        gap = 0.5 * ((sa + ra) - (sb + rb))
        m = mu[surf or "Hard"]
        pa = min(max(S.sigmoid(m + (sa - rb) / pt_m), 0.02), 0.98); pb = min(max(S.sigmoid(m + (sb - ra) / pt_m), 0.02), 0.98)
        q_game = min(max(S.sigmoid(gap), 1e-6), 1 - 1e-6); qs_g = H.set_prob(round(q_game, 4))
        ha = min(max(S.sigmoid(m + hm * (sa - rb) / pt_m), 0.02), 0.98); hb = min(max(S.sigmoid(m + hm * (sb - ra) / pt_m), 0.02), 0.98)
        qs_h = 0.5 * (sr_lib.set_prob(round(ha, 3), round(hb, 3)) + 1 - sr_lib.set_prob(round(hb, 3), round(ha, 3)))
        qs_h = min(max(qs_h, 1e-6), 1 - 1e-6)
        q_set, q_oth = (qs_g, qs_h) if pred == "game" else (qs_h, qs_g)
        q_o = min(max(S.match_from_set_prob(min(max(S.sigmoid(S.logit(q_oth) + c[a] - c[b]), 1e-6), 1 - 1e-6), bo), 1e-6), 1 - 1e-6)
        alt[i] = H.CAL_I + H.CAL_S * S.logit(q_o)
        q_adj = min(max(S.sigmoid(S.logit(q_set) + c[a] - c[b]), 1e-6), 1 - 1e-6)
        q_match = min(max(S.match_from_set_prob(q_adj, bo), 1e-6), 1 - 1e-6)
        cal = H.CAL_I + H.CAL_S * S.logit(q_match)
        k = frozenset((a, b))
        if h2h:
            aw_, bw_ = hh[k][a], hh[k][b]
            if aw_ + bw_: cal += H.h2h_slope(aw_ + bw_) * S.logit(min(max((aw_ + 0.5) / (aw_ + bw_ + 1.0), 1e-6), 1 - 1e-6))
        out[i] = cal; dsv[i] = sa - sb; drt[i] = ra - rb; S4[i] = (sa, ra, sb, rb, m)
        uo = r.get('upd_only', False)
        if not uo: hh[k][a] += 1

        # ---- closing (production) ----
        if not uo:
            sets = S.parse_sets(r["score"]); n_set = sets[0] + sets[1]
            q_mr = min(max(S.match_from_set_prob(q_set, bo), 1e-6), 1 - 1e-6)
            q_mc = min(max(S.sigmoid(H.CAL_I + H.CAL_S * S.logit(q_mr)), 1e-6), 1 - 1e-6)
            e_c = sets[0] - n_set * H.inv_match(round(q_mc, 4), bo)
            c[a] = H.CLOSING_B2 * c[a] + H.CLOSING_A2 * e_c; c[b] = H.CLOSING_B2 * c[b] - H.CLOSING_A2 * e_c

        # ---- surprise: u_s = a's serve part, u_r = a's return part (b gets the mirror) ----
        if uo: gw = gl = ng = 0
        else:
            gw, gl = parse_games(r["score"]); ng = gw + gl
        if pred == "game": q_up = q_game
        else: q_up = min(max(0.5 * (sr_lib.hold(pa) + 1 - sr_lib.hold(pb)), 1e-6), 1 - 1e-6)
        sc_g = gw - ng * q_up                       # games surprise (production's base score)
        u_s = u_r = 0.5 * sc_g
        pt = r["pts"]
        if pt_lam and pt:
            won, tot, sw, sn, rw, rn = pt
            if resid == "pooled":
                qp = min(max(S.sigmoid(gap / pt_m), 1e-6), 1 - 1e-6); e = (won - tot * qp) * pt_c / pt_m
                ps_, pr_ = 0.5 * e, 0.5 * e
            else:
                ea_, eb_ = (ha, hb) if res_hm else (pa, pb)
                ps_ = (sw - sn * ea_) * pt_c / pt_m; pr_ = (rw - rn * (1 - eb_)) * pt_c / pt_m
            if mu_k: mu[surf or "Hard"] += mu_k * ((sw + rn - rw) / (sn + rn) - S.sigmoid(m))
            u_s = (1 - pt_lam) * u_s + pt_lam * ps_; u_r = (1 - pt_lam) * u_r + pt_lam * pr_
            if uo: u_s, u_r = ps_, pr_
        aa, ab = age_at(a, when), age_at(b, when)
        ma = H.YOUNG_MULT if (aa is not None and aa < H.YOUNG_AGE) else 1.0
        mb = H.YOUNG_MULT if (ab is not None and ab < H.YOUNG_AGE) else 1.0
        if gain_early != 1.0:
            ma *= 1 + (gain_early - 1) * math.exp(-nm[a] / gain_settle); mb *= 1 + (gain_early - 1) * math.exp(-nm[b] / gain_settle)
        nm[a] += 1; nm[b] += 1
        if uo and ret_side == 'winner': mb = 0.0      # retired match: only the player who did not retire learns
        if uo and not r['pts']: ma = mb = 0.0        # walkover / no stats: nothing to learn
        st = GA * gscale * (k_ret if uo else 1.0)
        if split:
            # level (s+r) learns at the full rate, style (s-r) at k_d of it
            sm, df = u_s + u_r, u_s - u_r
            da_s, da_r = k_s * (sm + k_d * df), k_r * (sm - k_d * df)
            db_s, db_r = k_s * (-sm + k_d * df), k_r * (-sm - k_d * df)
        else:
            da_s = da_r = u_s + u_r; db_s = db_r = -(u_s + u_r)
        def upd(Sd, Rd, ka=1.0, kb=1.0):
            Sd[a] = GB * Sd[a] + st * ma * ka * da_s; Rd[a] = GB * Rd[a] + st * ma * ka * da_r
            Sd[b] = GB * Sd[b] + st * mb * kb * db_s; Rd[b] = GB * Rd[b] + st * mb * kb * db_r
        upd(sg, rg)
        if gap and surf in SURF:
            # gaps learn from this surface's surprise, then are re-centred so their weighted average is 0
            # (=> general is always a weighted average of the three surface skills)
            for p, mm, ds_, dr_ in ((a, ma, da_s, da_r), (b, mb, db_s, db_r)):
                ss[surf][p] += st * mm * surf_k * ds_; rs[surf][p] += st * mm * surf_k * dr_
                for D in (ss, rs):
                    cm = sum(w_ * D[x][p] for w_, x in zip(gwt, SURF))
                    for x in SURF: D[x][p] -= cm
            ns[surf][a] += 1; ns[surf][b] += 1
        elif surf in SURF:
            # surface step: x surf_k for everyone, plus an early boost over a player's first matches ON THIS SURFACE
            ka = surf_k * (1 + (surf_early - 1) * math.exp(-ns[surf][a] / surf_settle))
            kb = surf_k * (1 + (surf_early - 1) * math.exp(-ns[surf][b] / surf_settle))
            upd(ss[surf], rs[surf], ka, kb); ns[surf][a] += 1; ns[surf][b] += 1
        if clamp and not gap:
            # consistency: general must lie between the player's lowest and highest PLAYED surface rating;
            # if it breaks out, level-shift either general ("general") or all surface ratings ("surfaces")
            for p in (a, b):
                for G_, D in ((sg, ss), (rg, rs)):
                    vals = [D[x][p] for x in SURF if ns[x][p] >= 1]
                    if not vals: continue
                    lo, hi = min(vals), max(vals); g_ = G_[p]
                    sh = g_ - hi if g_ > hi else (g_ - lo if g_ < lo else 0.0)
                    if sh:
                        if clamp == "general": G_[p] = g_ - sh
                        else:
                            for x in SURF: D[x][p] += sh
        if hook_post is not None: hook_post(i, r, dict(sg=sg, rg=rg, ss=ss, rs=rs, c=c))
    return out, dict(sg=sg, rg=rg, ss=ss, rs=rs, ns=ns, c=c, mu=dict(mu), last=last, dsv=dsv, drt=drt, alt=alt, S4=S4)


def build_ext(allrows):
    """Rows for the rating replay: every rated (completed) match, plus retirements and walkovers as update-only rows
    (a retirement with serve stats updates ONLY the player who did not retire, from the points played; walkovers
    update nothing but count for layoff dates). Returns (ext rows, index of rated rows in ext, original row id -> ext index)."""
    ext, rated_idx, where = [], [], {}
    for r in allrows:
        done = S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])
        if done:
            where[id(r)] = len(ext); rated_idx.append(len(ext)); ext.append(r)
        elif (r["ret"] or r["wo"]) and not r["dq"]:
            q = dict(r); q["upd_only"] = True; q["_oid"] = id(r); q["pts"] = r["pts"] if (r["ret"] and not r["wo"]) else None
            where[id(r)] = len(ext); ext.append(q)
    return ext, np.array(rated_idx), where
