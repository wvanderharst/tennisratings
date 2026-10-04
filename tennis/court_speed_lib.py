"""Court-speed state builder (same logic as court_speed.py), reusable by the live script."""
import math
from collections import defaultdict
import numpy as np
import score_driven as S
from margin_model import parse_games


def court(r): return ("Indoor " if r["indoor"] else "") + (r["surf"] or "Hard")


def ekey(r):
    tid = r["tid"]; k = tid.split("-", 1)[1] if "-" in tid else tid
    return (r["level"], k, court(r))


def build(allrows):
    n = sum(1 for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"]))
    LEAGUE0 = 0.07
    ace_skill = defaultdict(lambda: 1.0); ace_conc = defaultdict(lambda: 1.0)   # multiplicative, EW
    league = LEAGUE0
    ev_hist = defaultdict(list)       # event key -> [(year, log_ratio, weight_points)]
    surf_mean = defaultdict(float); surf_w = defaultdict(float)
    K_SHRINK = 1500.0                 # service points of pseudo-data at the surface mean
    
    speed = np.zeros(n); has_speed = np.zeros(n, bool)
    acer_a = np.zeros(n); acer_b = np.zeros(n)
    i_sc = 0
    for r in allrows:
        scored = S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])
        k = ekey(r); cr = court(r); yr = r["when"].year
        if scored:
            # ---- pre-match speed estimate for this event ----
            num = K_SHRINK * (surf_mean[cr] / max(surf_w[cr], 1e-9) if surf_w[cr] > 0 else 0.0); den = K_SHRINK
            for (y0, lr, w0) in ev_hist[k]:
                age = yr - y0
                ww = w0 * (0.6 ** age)                   # older editions count less
                num += lr * ww; den += ww
            speed[i_sc] = num / den; has_speed[i_sc] = len(ev_hist[k]) > 0
            acer_a[i_sc] = math.log(ace_skill[r["a"]]); acer_b[i_sc] = math.log(ace_skill[r["b"]])
            i_sc += 1
        # ---- update with this match's aces (any completed-stats row) ----
        if all(r[x] for x in ("svpt_a", "svpt_b")) and r["ace_a"] is not None and r["ace_b"] is not None and r["svpt_a"] > 10 and r["svpt_b"] > 10:
            ea = league * ace_skill[r["a"]] * ace_conc[r["b"]] * r["svpt_a"]
            eb = league * ace_skill[r["b"]] * ace_conc[r["a"]] * r["svpt_b"]
            act = r["ace_a"] + r["ace_b"]; exp_ = ea + eb
            if exp_ > 0.5:
                lr = math.log((act + 0.5) / (exp_ + 0.5))
                w = r["svpt_a"] + r["svpt_b"]
                ev_hist[k].append((yr, lr, w))
                surf_mean[cr] = 0.999 * surf_mean[cr] + lr * w * 0.001; surf_w[cr] = 0.999 * surf_w[cr] + w * 0.001
                # player skills, adjusted for the court's speed factor (exp(lr) of THIS match's court estimate is noisy; use event mean so far)
                cf_ = math.exp(max(min(lr, 1.0), -1.0))
                for srv, ret, a_, sv in ((r["a"], r["b"], r["ace_a"], r["svpt_a"]), (r["b"], r["a"], r["ace_b"], r["svpt_b"])):
                    expd = league * ace_skill[srv] * ace_conc[ret] * cf_
                    ratio = (a_ + 0.5) / (expd * sv + 0.5)
                    ace_skill[srv] *= ratio ** 0.03
                    ace_conc[ret] *= ratio ** 0.015
                league = 0.9995 * league + 0.0005 * (act / w)
    def speed_for(key, yr, cr):
        num = K_SHRINK * (surf_mean[cr] / max(surf_w[cr], 1e-9) if surf_w[cr] > 0 else 0.0); den = K_SHRINK
        for (y0, lr, w0) in ev_hist[key]:
            ww = w0 * (0.6 ** (yr - y0)); num += lr * ww; den += ww
        return num / den, len(ev_hist[key]) > 0
    return dict(speed=speed, has_speed=has_speed, speed_for=speed_for, ace_skill=ace_skill, acer_a=acer_a, acer_b=acer_b)
