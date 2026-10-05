"""Full-pipeline confounder harness (v18 baseline: shared surprise + young-age
boost + closing + h2h). One causal replay; for every scored match we log the
baseline calibrated logit (winner's perspective) plus a table of pre-match
confounder features, each as (winner-side minus loser-side). Fitting and
held-out scoring happen in confounder_fit.py, off the saved table, so the
model trajectory here is IDENTICAL to deployed v18 -- corrections enter only
the final prediction. (Stage 2, feeding corrections back into the surprise,
needs its own full replay per setting: see confounder_stage2.py.)

Retirements/walkovers are still excluded from scoring and rating updates, same
as always, but they ARE walked for rest/fatigue/injury bookkeeping.
"""
import paths as P
import csv, glob, json, math, os, re, sys
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
import score_driven as S
from load_uploads import UPLOAD_DIR, HIST_DIR
from margin_model import parse_games

OUT = P.OUT
SURFACES = ("Hard", "Clay", "Grass")
BLEND_W = 0.6
MIN_SURF_MATCHES = 8
IDLE_GRACE_DAYS = 30
IDLE_GB = 0.9995
YOUNG_AGE, YOUNG_MULT = 24.0, 1.4
gfit = json.load(open(f"{OUT}/games_rate_fit.json"))
ga_rate, gb_persist, g_scaling = gfit["a"], gfit["b"], gfit["scaling"]
bofit = json.load(open(f"{OUT}/bo_aware_fit.json"))
CAL_I, CAL_S = bofit["CAL_I"], bofit["CAL_S"]
CLOSING_A2, CLOSING_B2 = 0.005, 0.999
h2h_tbl = json.load(open(f"{OUT}/h2h_after_closing_fit_shared.json"))
birth = json.load(open(f"{OUT}/player_birth_ordinal.json"))

ROUND_ORD = {"Q1": -3, "Q2": -2, "Q3": -1, "Q4": -1, "RR": 0, "R128": 0, "R64": 1, "R32": 2,
             "R16": 3, "QF": 4, "SF": 5, "BR": 6, "F": 6}


def h2h_slope(n):
    if n <= 0: return 0.0
    if n == 1: return h2h_tbl["1"]
    if n == 2: return h2h_tbl["2"]
    if n <= 4: return h2h_tbl["3-4"]
    return h2h_tbl["5+"]


def _ddint():
    return defaultdict(int)


def _elo0():
    return 1500.0


HT_BEFORE = (2024, 1, 1)


def ht_mean(rows):
    """Mean player height, from the matches before 2024 only: that history never changes, so the value stays fixed as
    new matches arrive (it fills missing heights; a daily-moving mean would void the replay checkpoints, ckpt.py)."""
    import datetime
    cut = datetime.date(*HT_BEFORE)
    return float(np.mean([h for r in rows if r["when"] < cut for h in (r["ht_a"], r["ht_b"]) if h and 150 < h < 220]))


def age_at(p, when):
    b = birth.get(p)
    return None if b is None else (when.toordinal() - b) / 365.25


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _stats_swapped(r):
    """True when a row's serve stats are evidently attached to the wrong player: the winner took >55% of the
    games but <47% of the points. That never happens in clean years (0 cases 2015-18, 2020, 2022-24) but shows up in
    batches (ATP 2019: 30, 2025: 45; WTA 2021: 33, 2025: 43) -- a data-entry swap, so the stats are flipped back."""
    try:
        wsv, w1, w2 = int(float(r["w_svpt"])), int(float(r["w_1stWon"])), int(float(r["w_2ndWon"]))
        lsv, l1, l2 = int(float(r["l_svpt"])), int(float(r["l_1stWon"])), int(float(r["l_2ndWon"]))
    except (TypeError, ValueError, KeyError):
        return False
    gw = gl = 0
    for t in (r.get("score") or "").replace(",", " ").split():
        m = re.match(r"^(\d+)-(\d+)", t)
        if m: gw += int(m.group(1)); gl += int(m.group(2))
    pw = w1 + w2 + (lsv - l1 - l2); pl = l1 + l2 + (wsv - w1 - w2)
    return gw + gl > 0 and pw + pl > 0 and gw / (gw + gl) > 0.55 and pw / (pw + pl) < 0.47


def _swap_stats(r):
    r = dict(r)
    for k in list(r):
        if k.startswith("w_") and ("l_" + k[2:]) in r: r[k], r["l_" + k[2:]] = r["l_" + k[2:]], r[k]
    return r


def _points(r):
    """(points won by winner, total points) from serve stats, or None. Swapped-stat rows are flipped back first."""
    if _stats_swapped(r): r = _swap_stats(r)
    try:
        wsv, w1, w2 = int(float(r["w_svpt"])), int(float(r["w_1stWon"])), int(float(r["w_2ndWon"]))
        lsv, l1, l2 = int(float(r["l_svpt"])), int(float(r["l_1stWon"])), int(float(r["l_2ndWon"]))
    except (TypeError, ValueError, KeyError):
        return None
    tot = wsv + lsv
    if tot < 40 or w1 + w2 > wsv or l1 + l2 > lsv: return None
    return (w1 + w2 + (lsv - l1 - l2), tot, w1 + w2, wsv, lsv - l1 - l2, lsv)   # (won, total, srv_won, srv_n, ret_won, ret_n) for the winner


SURF_FILL = None    # None: fall back to general-only when either player has < MIN_SURF_MATCHES on the surface.
                    # "rank": impute the short player's surface rating from their general-rating RANK (quantile map
                    #   onto the surface-rating distribution of active players with enough surface matches);
                    # "regress": impute from a monthly regression of surface rating on general rating (same pool).
SURF_SCALE = 1.0        # multiplies surface-rating gaps in the blend (stretching surface ratings to general's spread)
SURF_FILL_MODE = "one"   # "one": impute only the short player(s), keep the other's real surface rating
SURF_SHRINK_K = None     # if set (with SURF_FILL): EVERY player's surface rating is pooled toward the imputed value,
                         #   s_eff = (n*s + K*s_imputed)/(n + K), n = matches on that surface; no hard threshold
DATA_PATHS = None   # set to a list of CSVs to run the model on another dataset (e.g. WTA)
CHALL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tml-chall")
CHALL_HIST = os.environ.get("TENNIS_CHALL", "1") == "1"   # ATP: also load the pre-2024 Challenger seasons (TENNIS_CHALL=0: off)


def load():
    if DATA_PATHS is not None:
        paths = list(DATA_PATHS)
    else:
        paths = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.csv")))
        for p in sorted(glob.glob(os.path.join(HIST_DIR, "*.csv"))):
            n = os.path.basename(p)
            if n[:4].isdigit() and int(n[:4]) < 2024:
                paths.append(p)
        if CHALL_HIST:      # Challenger seasons before 2024 (2024+ are in atp_uploads/)
            for p in sorted(glob.glob(os.path.join(CHALL_DIR, "*_challenger.csv"))):
                n = os.path.basename(p)
                if n[:4].isdigit() and int(n[:4]) < 2024:
                    paths.append(p)
    rows, seen = [], set()
    for path in paths:
        if "ATP_Database" in path or "qualifying_matches" in path:   # the WTA upload lives in the same folder
            continue
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f):
                rs_ = _swap_stats(r) if _stats_swapped(r) else r     # stats read from the corrected orientation
                when = S.parse_date(r.get("tourney_date"))
                if not when: continue
                a = (r.get("winner_name") or "").strip(); b = (r.get("loser_name") or "").strip()
                if not a or not b or a == b: continue
                lvl_raw = (r.get("tourney_level") or "").strip().upper()
                level = S.LEVEL_OF.get(lvl_raw, "tour")
                score = (r.get("score") or "").strip()
                key = (when, a, b, score)
                if key in seen: continue
                seen.add(key)
                surf = r.get("surface", "")
                if surf not in ("Hard", "Clay", "Grass", "Carpet"): surf = ""
                bo = to_int(r.get("best_of")) or 3
                up = score.upper()
                tid = r.get("tourney_id", "")
                rows.append(dict(
                    when=when, tid=tid, mid=f"{level[:2]}{tid}", level=level, lvl=lvl_raw,
                    a=a, b=b, surf=surf, bo=bo, score=score, rnd=(r.get("round") or "").strip(),
                    indoor=(r.get("indoor") or "").strip().upper() == "I",
                    minutes=to_int(r.get("minutes")),
                    ret="RET" in up, wo="W/O" in up, dq=("DEF" in up) or ("ABD" in up),
                    ent_a=(r.get("winner_entry") or "").strip().upper(), ent_b=(r.get("loser_entry") or "").strip().upper(),
                    ioc_a=(r.get("winner_ioc") or "").strip(), ioc_b=(r.get("loser_ioc") or "").strip(),
                    hand_a=(r.get("winner_hand") or "").strip().upper(), hand_b=(r.get("loser_hand") or "").strip().upper(),
                    ht_a=to_int(r.get("winner_ht")), ht_b=to_int(r.get("loser_ht")),
                    rank_a=to_int(r.get("winner_rank")), rank_b=to_int(r.get("loser_rank")),
                    seed_a=to_int(r.get("winner_seed")), seed_b=to_int(r.get("loser_seed")),
                    sv_a=to_int(rs_.get("w_SvGms")), bpf_a=to_int(rs_.get("w_bpFaced")), bps_a=to_int(rs_.get("w_bpSaved")),
                    sv_b=to_int(rs_.get("l_SvGms")), bpf_b=to_int(rs_.get("l_bpFaced")), bps_b=to_int(rs_.get("l_bpSaved")),
                    pts=_points(r),
                    ace_a=to_int(rs_.get("w_ace")), ace_b=to_int(rs_.get("l_ace")),
                    svpt_a=to_int(rs_.get("w_svpt")), svpt_b=to_int(rs_.get("l_svpt")),
                    tname=(r.get("tourney_name") or "").strip(),
                ))
    # quals (level 'qual') happen before main draw of same tourney_id -> order qual first
    lvl_ord = {"qual": 0, "tour": 1, "chall": 1}
    rows.sort(key=lambda r: (r["when"], r["tid"], lvl_ord.get(r["level"], 1), ROUND_ORD.get(r["rnd"], 2)))
    return rows


@lru_cache(maxsize=None)
def set_prob(q):
    @lru_cache(maxsize=None)
    def rec(x, y):
        if x >= 6 and x - y >= 2: return 1.0
        if y >= 6 and y - x >= 2: return 0.0
        if x == 6 and y == 6: return q
        return q * rec(x + 1, y) + (1 - q) * rec(x, y + 1)
    return rec(0, 0)


@lru_cache(maxsize=None)
def inv_match(qm, bo, lo=0.001, hi=0.999, it=40):
    for _ in range(it):
        mid = (lo + hi) / 2
        if S.match_from_set_prob(mid, bo) < qm: lo = mid
        else: hi = mid
    return (lo + hi) / 2


FEATS = [
    "age_over28", "age_over32", "age_under22",
    "rest_log", "rest_log_x_age27",
    "layoff90", "since_return_le3",
    "fat_sets", "fat_min", "fat_sets_x_age27",
    "inj_decay30", "prev_ret_loss", "pr_entry",
    "q_entry", "wc_entry", "ll_entry",
    "home",
    "logrank", "rank_missing",
    "lefty", "height", "height_x_fast",
    "streak", "bo5_x_age27", "big_x_age27", "seeded",
]


_ROWS = None


def main(init_prior=None, kappa=0.0, feed=None, out_name="confounder_table.npz", verbose=True, lam_match=0.0, set_mode=None, gscale=1.0, sbin=0.0, cap=None, tbw=0.0, dw=1.0, init_rank=None, age_drift=None, lvl_w=None, opp_n=None, idle_gb=None, gas_rho=0.0, gas_nu=0.0, g_scale_pow=None, pt_lam=0.0, pt_m=2.132, pt_c=1.0, pt_mode='pooled', pt_rho=0.0, pt_nu=0.0, pt_huber=0.0, pt_link='logit', pt_mu=0.64, young_mult=None, closing_on=True, blend_w=None, snap_slams=False, ckpt_tag=None):
    """init_prior: dict level('tour'/'chall'/'qual') -> starting phi_gen for a
    player's first-ever scored match (None = 0, as deployed).
    kappa/feed: feed kappa * sum(feed[f] * featdiff_f) into the GAME gap used
    for the update surprise (not the logged baseline prediction)."""
    global _ROWS
    if _ROWS is None:
        _ROWS = load()
    rows = _ROWS
    feed = feed or {}
    feed_idx = [(FEATS.index(k), v) for k, v in feed.items()]
    seen_player = set()

    # host country per event, inferred from wildcard entrants (modal ioc), before any match is played
    wc_ioc = defaultdict(Counter)
    for r in rows:
        for side in ("a", "b"):
            if r["ent_" + side] == "WC" and r["ioc_" + side]:
                wc_ioc[r["tid"]][r["ioc_" + side]] += 1
    host = {tid: c.most_common(1)[0][0] for tid, c in wc_ioc.items() if c and c.most_common(1)[0][1] >= 2}
    verbose and print(len(host), "events with inferred host country")

    HT_MEAN = ht_mean(rows)

    phi_gen = defaultdict(float)
    phi_surf = {s: defaultdict(float) for s in SURFACES}
    n_surf = {s: defaultdict(int) for s in SURFACES}
    c = defaultdict(float)
    h2h = defaultdict(_ddint)
    last_date = {}         # for idle decay (scored matches only, as in v18)
    last_any = {}          # any match incl. RET/W/O, for rest
    ent_rest = {}          # (mid, p) -> rest days entering event
    cumsets = defaultdict(int); cummin = defaultdict(int)
    last_inj = {}          # p -> date of last RET/W/O loss
    # ---- extras for Elo / h2h analysis (never feed back into the model) ----
    elo = defaultdict(_elo0); elo_s = {s_: defaultdict(_elo0) for s_ in SURFACES}
    elo_n = defaultdict(int); elo_ns = {s_: defaultdict(int) for s_ in SURFACES}
    pair_hist = defaultdict(list)   # sorted pair -> [(ord, surf, residual for pair[0])]
    ex = defaultdict(list)
    prev_was_ret_loss = defaultdict(bool)
    return_date = {}       # p -> date they came back from 90+ day layoff
    n_since_return = defaultdict(int)
    streak = defaultdict(int)

    out_when, out_base, out_bo, out_feats = [], [], [], []
    fill_tab = {}

    slam_ent = defaultdict(set)
    if snap_slams:
        for r in rows:
            if r["lvl"] == "G" and r["level"] == "tour":
                slam_ent[r["tid"]].update((r["a"], r["b"]))
    slam_snap = {}; slam_side = {}
    # ---- checkpoint (ckpt.py): resume from the saved state, save the state at the checkpoint date ----
    STATE = dict(seen_player=seen_player, phi_gen=phi_gen, phi_surf=phi_surf, n_surf=n_surf, c=c, h2h=h2h, last_date=last_date,
                 last_any=last_any, ent_rest=ent_rest, cumsets=cumsets, cummin=cummin, last_inj=last_inj, elo=elo, elo_s=elo_s,
                 elo_n=elo_n, elo_ns=elo_ns, pair_hist=pair_hist, ex=ex, prev_was_ret_loss=prev_was_ret_loss,
                 return_date=return_date, n_since_return=n_since_return, streak=streak, out_when=out_when, out_base=out_base,
                 out_bo=out_bo, out_feats=out_feats, fill_tab=fill_tab)
    start = 0; save_at = None
    if ckpt_tag and not snap_slams:
        import ckpt
        args = {k_: v_ for k_, v_ in locals().items() if k_ in main.__code__.co_varnames[:main.__code__.co_argcount]}
        ck = ckpt.Checkpoint(ckpt_tag, rows, (sorted(args.items()), SURF_FILL, MIN_SURF_MATCHES, SURF_SCALE, SURF_FILL_MODE,
                                              SURF_SHRINK_K, BLEND_W, IDLE_GB, HT_MEAN))
        saved = ck.load(); save_at = ckpt.first_at(rows, ck.cut)
        if saved:
            start = saved["i"]
            for k_, v_ in saved["state"].items():
                tgt = STATE[k_]
                if isinstance(tgt, set): tgt.update(v_)
                elif isinstance(tgt, list): tgt.extend(v_)
                elif k_ in ("phi_surf", "n_surf", "elo_s", "elo_ns"):
                    for s_ in SURFACES: tgt[s_].update(v_[s_])
                else: tgt.update(v_)
            if start == save_at: save_at = None
    for i_ in range(start, len(rows)):
        if i_ == save_at: ck.save(dict(i=i_, state=STATE))
        r = rows[i_]
        when, a, b, surf, bo, mid = r["when"], r["a"], r["b"], r["surf"], r["bo"], r["mid"]
        if snap_slams and r["lvl"] == "G" and r["level"] == "tour" and r["tid"] not in slam_snap:
            ent = slam_ent[r["tid"]]; ss_ = {}
            for p_ in ent:
                f_ = 1.0
                if p_ in last_date:
                    idl = max(0, (when - last_date[p_]).days - IDLE_GRACE_DAYS)
                    f_ = (idle_gb if idle_gb is not None else IDLE_GB) ** idl
                sk = surf if surf in SURFACES else None
                ss_[p_] = dict(phi=phi_gen[p_] * f_, phis=(phi_surf[sk][p_] * f_ if sk else 0.0), ns=(n_surf[sk][p_] if sk else 0), c=c[p_] * f_)
            hh = {}
            for p_ in ent:
                for q_, cnt in h2h.get(frozenset((p_,)), {}).items(): pass
            pairs = {}
            for k_, d_ in h2h.items():
                if len(k_) == 2 and k_ <= ent:
                    pairs[tuple(sorted(k_))] = dict(d_)
            slam_snap[r["tid"]] = dict(surf=surf, bo=bo, when=when.toordinal(), players=ss_, h2h=pairs)
        for p in (a, b):
            if (mid, p) not in ent_rest:
                lp = last_any.get(p)
                d = (when - lp).days if lp else None
                ent_rest[(mid, p)] = d
                if d is not None and d >= 90:
                    return_date[p] = when; n_since_return[p] = 0

        sets = S.parse_sets(r["score"]); g = parse_games(r["score"])
        scored = sets is not None and g is not None and not (r["ret"] or r["wo"] or r["dq"])

        if scored:
            if init_prior:
                for p in (a, b):
                    if p not in seen_player:
                        seen_player.add(p)
                        v = init_prior.get(r["level"], 0.0)
                        if init_rank is not None:
                            rk = r["rank_a"] if p == a else r["rank_b"]
                            if rk:
                                v = init_rank[0] - init_rank[1] * (math.log(rk) - math.log(100.0))
                                v = min(max(v, -1.5), 1.0)
                        phi_gen[p] = v
                        for s in SURFACES: phi_surf[s][p] = v
            for p in (a, b):
                if p in last_date:
                    idle = max(0, (when - last_date[p]).days - IDLE_GRACE_DAYS)
                    if idle > 0:
                        f = (idle_gb if idle_gb is not None else IDLE_GB) ** idle
                        phi_gen[p] *= f; c[p] *= f
                        for s in SURFACES: phi_surf[s][p] *= f
            if age_drift is not None:
                for p in (a, b):
                    if p in last_date:
                        yrs = min((when - last_date[p]).days / 365.25, 1.0)   # cap: a long layoff gets at most 1y of drift
                        ag = age_at(p, when)
                        if ag is not None and yrs > 0:
                            dy, do_, ay, ao = age_drift
                            dr = (dy if ag < ay else (-do_ * (ag - ao) if ag > ao else 0.0)) * yrs
                            phi_gen[p] = min(max(phi_gen[p] + dr, -3.0), 3.0)
                            for s_ in SURFACES: phi_surf[s_][p] = min(max(phi_surf[s_][p] + dr, -3.0), 3.0)
            last_date[a] = when; last_date[b] = when

            # ---- baseline v18 prediction ----
            q_gen = min(max(S.sigmoid(phi_gen[a] - phi_gen[b]), 1e-9), 1 - 1e-9)
            use_surf = surf in SURFACES and n_surf[surf][a] >= MIN_SURF_MATCHES and n_surf[surf][b] >= MIN_SURF_MATCHES
            BW = BLEND_W if blend_w is None else blend_w
            gap = BW * S.logit(q_gen) + (1 - BW) * SURF_SCALE * (phi_surf[surf][a] - phi_surf[surf][b]) if use_surf else S.logit(q_gen)
            if SURF_FILL and surf in SURFACES and (SURF_SHRINK_K or not use_surf):
                ym_ = (when.year, when.month)
                if fill_tab.get("ym") != ym_:           # refresh the pools once a month
                    fill_tab.clear(); fill_tab["ym"] = ym_
                    for s_ in SURFACES:
                        pool = [p_ for p_, d_ in last_date.items() if n_surf[s_][p_] >= MIN_SURF_MATCHES and (when - d_).days <= 365]
                        if len(pool) >= 30:
                            gs_ = np.array([phi_gen[p_] for p_ in pool]); ss_ = np.array([phi_surf[s_][p_] for p_ in pool])
                            fill_tab[s_] = (np.sort(gs_), np.sort(ss_), np.polyfit(gs_, ss_, 1))
                if surf in fill_tab:
                    gsort, ssort, coef = fill_tab[surf]
                    def fill_(p_):
                        if SURF_SHRINK_K:
                            qq = np.searchsorted(gsort, phi_gen[p_]) / len(gsort)
                            si_ = float(np.interp(qq, (np.arange(len(ssort)) + 0.5) / len(ssort), ssort))
                            n_ = n_surf[surf][p_]
                            return (n_ * phi_surf[surf][p_] + SURF_SHRINK_K * si_) / (n_ + SURF_SHRINK_K)
                        if n_surf[surf][p_] >= MIN_SURF_MATCHES: return phi_surf[surf][p_]
                        if SURF_FILL == "rank":
                            qq = np.searchsorted(gsort, phi_gen[p_]) / len(gsort)
                            return float(np.interp(qq, (np.arange(len(ssort)) + 0.5) / len(ssort), ssort))
                        return float(coef[0] * phi_gen[p_] + coef[1])
                    gap = BW * S.logit(q_gen) + (1 - BW) * SURF_SCALE * (fill_(a) - fill_(b))
            q_game = min(max(S.sigmoid(gap), 1e-6), 1 - 1e-6)
            q_set = set_prob(round(q_game, 4))
            q_setadj = min(max(S.sigmoid(S.logit(q_set) + c[a] - c[b]), 1e-6), 1 - 1e-6)
            q_match = min(max(S.match_from_set_prob(q_setadj, bo), 1e-6), 1 - 1e-6)
            cal = CAL_I + CAL_S * S.logit(q_match)
            cal_noh2h = cal
            wins = h2h.get(frozenset((a, b)))
            if wins:
                aw, bw = wins.get(a, 0), wins.get(b, 0); n = aw + bw
                if n > 0:
                    cal += h2h_slope(n) * S.logit(min(max((aw + 0.5) / (n + 1.0), 1e-6), 1 - 1e-6))

            # ---- features, per side ----
            def side(p, s):
                age = age_at(p, when)
                age_ok = age is not None
                ag = age if age_ok else 26.0
                rest = ent_rest[(mid, p)]
                rest_c = min(rest, 60) if rest is not None else 21
                rest_log = math.log1p(rest_c)
                a27 = max(0.0, ag - 27.0)
                ld = last_inj.get(p)
                inj = math.exp(-(when - ld).days / 30.0) if ld is not None else 0.0
                rank = r["rank_" + s]
                ht = r["ht_" + s]
                ht = ht if ht and 150 < ht < 220 else HT_MEAN
                ent = r["ent_" + s]
                fast = 1.0 if (surf == "Grass" or r["indoor"]) else 0.0
                big = 1.0 if r["lvl"] in ("G", "M") else 0.0
                return [
                    max(0.0, ag - 28.0), max(0.0, ag - 32.0), max(0.0, 22.0 - ag),
                    rest_log, rest_log * a27,
                    1.0 if (rest is not None and rest >= 90) else 0.0,
                    1.0 if (p in return_date and n_since_return[p] <= 3) else 0.0,
                    float(cumsets[(mid, p)]), cummin[(mid, p)] / 60.0, cumsets[(mid, p)] * a27,
                    inj, 1.0 if prev_was_ret_loss[p] else 0.0, 1.0 if ent == "PR" else 0.0,
                    1.0 if ent == "Q" else 0.0, 1.0 if ent == "WC" else 0.0, 1.0 if ent == "LL" else 0.0,
                    1.0 if (host.get(r["tid"]) and r["ioc_" + s] == host.get(r["tid"])) else 0.0,
                    math.log(rank) if rank else math.log(1500.0), 0.0 if rank else 1.0,
                    1.0 if r["hand_" + s] == "L" else 0.0, (ht - HT_MEAN) / 10.0, (ht - HT_MEAN) / 10.0 * fast,
                    float(min(streak[p], 15)), a27 * (1.0 if bo == 5 else 0.0), a27 * big,
                    1.0 if r["seed_" + s] else 0.0,
                ]
            fa, fb = side(a, "a"), side(b, "b")
            if snap_slams and r["lvl"] == "G":
                for p_, fv, sd_ in ((a, fa, "a"), (b, fb, "b")):
                    if (r["tid"], p_) not in slam_side:
                        slam_side[(r["tid"], p_)] = dict(f=fv, i=len(out_base), rank=r["rank_" + sd_])
            out_when.append(when.toordinal()); out_base.append(cal); out_bo.append(bo)
            out_feats.append([x - y for x, y in zip(fa, fb)])

            # ---- extras: Elo (538 tennis style) + h2h variants, a's perspective ----
            ea = elo[a]; eb = elo[b]
            ex["elo"].append((ea - eb) * math.log(10) / 400)
            if surf in SURFACES:
                ba_ = 0.5 * (ea + elo_s[surf][a]); bb_ = 0.5 * (eb + elo_s[surf][b])
            else:
                ba_, bb_ = ea, eb
            ex["elo_blend"].append((ba_ - bb_) * math.log(10) / 400)
            ex["cal_noh2h"].append(cal_noh2h)
            ex["cdiff"].append(c[a] - c[b])
            ex["phi_a"].append(phi_gen[a]); ex["phi_b"].append(phi_gen[b])
            ex["phigap"].append(gap)
            ex["n_a"].append(elo_n[a]); ex["n_b"].append(elo_n[b])
            ex["surf"].append(surf); ex["lvl"].append(r["lvl"]); ex["names"].append(a + "|" + b)
            key = tuple(sorted((a, b))); sign = 1.0 if key[0] == a else -1.0
            hist = pair_hist[key]
            wd = h2h.get(frozenset((a, b))) or {}
            aw_, bw_ = wd.get(a, 0), wd.get(b, 0)
            ex["h2h_n"].append(aw_ + bw_)
            ex["h2h_rate"].append(S.logit(min(max((aw_ + 0.5) / (aw_ + bw_ + 1.0), 1e-6), 1 - 1e-6)) if aw_ + bw_ else 0.0)
            ex["h2h_res"].append(sign * sum(h[2] for h in hist))
            ex["h2h_res_surf"].append(sign * sum(h[2] for h in hist if h[1] == surf))
            ex["h2h_n_surf"].append(sum(1 for h in hist if h[1] == surf))
            od = when.toordinal()
            ex["h2h_res_decay"].append(sign * sum(h[2] * math.exp(-(od - h[0]) / 730.0) for h in hist))
            ex["h2h_res_decay_surf"].append(sign * sum(h[2] * math.exp(-(od - h[0]) / 730.0) for h in hist if h[1] == surf))
            p_model = 1.0 / (1.0 + math.exp(-cal_noh2h))
            hist.append((od, surf, sign * (1.0 - p_model)))
            # Elo update (winner a)
            for E_, NA, NB, ra, rb in [(elo, elo_n, elo_n, ea, eb)] + ([(elo_s[surf], elo_ns[surf], elo_ns[surf], elo_s[surf][a], elo_s[surf][b])] if surf in SURFACES else []):
                pe = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                ka = 250.0 / (NA[a] + 5) ** 0.4; kb = 250.0 / (NB[b] + 5) ** 0.4
                E_[a] = ra + ka * (1 - pe); E_[b] = rb - kb * (1 - pe)
                NA[a] += 1; NB[b] += 1

            # ---- v18 update ----
            h2h[frozenset((a, b))][a] += 1
            n_set = sets[0] + sets[1]
            q_match_raw = min(max(S.match_from_set_prob(q_set, bo), 1e-6), 1 - 1e-6)
            q_match_cal = min(max(S.sigmoid(CAL_I + CAL_S * S.logit(q_match_raw)), 1e-6), 1 - 1e-6)
            q_set_cal = inv_match(round(q_match_cal, 4), bo)
            sc_c = sets[0] - n_set * q_set_cal
            if closing_on:
                c[a] = CLOSING_B2 * c[a] + CLOSING_A2 * sc_c
                c[b] = CLOSING_B2 * c[b] - CLOSING_A2 * sc_c
            gw, gl = g
            ng = gw + gl
            if kappa and feed_idx:
                fd = out_feats[-1]
                delta = sum(w * fd[j] for j, w in feed_idx)
                q_upd = min(max(S.sigmoid(gap + kappa * delta), 1e-6), 1 - 1e-6)
            else:
                q_upd = q_game
            sc = gw - ng * q_upd
            info = ng * q_upd * (1 - q_upd)
            gsp = g_scaling if g_scale_pow is None else g_scale_pow
            if gsp: sc /= max(info, 1e-6) ** gsp * (1.0 / max(ng * 0.25, 1e-6) ** gsp)   # info scaling, normalised to keep the average step
            # ---- alternative GAS distributions ----
            if pt_lam and r["pts"]:
                # POINT-level binomial: q_point = sigmoid(gap / m); score w.r.t. the game-scale gap is (pts - n*q)/m
                pw_, pn_, sw_, sn_, rw_, rn_ = r["pts"]
                gg = S.logit(q_upd) / pt_m                      # point-scale gap
                if pt_link == "probit":
                    # normal (probit) link, scale-matched to the logistic: sigmoid(x) ~ Phi(x / 1.702)
                    x_ = gg / 1.702
                    qp = min(max(0.5 * (1 + math.erf(x_ / math.sqrt(2))), 1e-6), 1 - 1e-6)
                    dens = math.exp(-x_ * x_ / 2) / math.sqrt(2 * math.pi) / 1.702      # d qp / d gg
                    sp = (pw_ - pn_ * qp) * dens / (qp * (1 - qp)) / pt_m
                elif pt_mode == "servret":
                    mu = S.logit(pt_mu)
                    q1 = min(max(S.sigmoid(mu + gg), 1e-6), 1 - 1e-6)     # winner holds serve points
                    q2 = min(max(S.sigmoid(-mu + gg), 1e-6), 1 - 1e-6)    # winner wins return points
                    sp = ((sw_ - sn_ * q1) + (rw_ - rn_ * q2)) / pt_m
                elif pt_mode == "logitnorm":
                    emp = S.logit((pw_ + 0.5) / (pn_ + 1.0))
                    sp = (emp - gg) * 38.0 / pt_m                        # 38 ~ typical n*q(1-q) so the average step matches
                else:
                    qp = min(max(S.sigmoid(gg), 1e-6), 1 - 1e-6)
                    sp = (pw_ - pn_ * qp) / pt_m
                if pt_rho:
                    sp /= (1.0 + (pn_ - 1) * pt_rho) / (1.0 + 150 * pt_rho)
                if pt_nu:
                    vv = pn_ * 0.25 / pt_m ** 2
                    sp = sp / (1.0 + sp * sp / (pt_nu * vv))
                if pt_huber:
                    sd = (pn_ * 0.25) ** 0.5 / pt_m
                    sp = max(-pt_huber * sd, min(pt_huber * sd, sp))
                sc = (1 - pt_lam) * sc + pt_lam * pt_c * sp
            if gas_rho:
                # beta-binomial (games within a match correlated): score shrinks with match length
                sc /= (1.0 + (ng - 1) * gas_rho) / (1.0 + 21 * gas_rho)
            if gas_nu:
                # Student-t style robust score: big surprises are down-weighted
                sc = sc / (1.0 + sc * sc / (gas_nu * max(info, 1e-6)))
            if lam_match:
                # match-RESULT surprise on top of the game-count surprise (winner is a)
                sc += lam_match * (1.0 - 1.0 / (1.0 + math.exp(-cal_noh2h)))
            aa, ab = age_at(a, when), age_at(b, when)
            YM = YOUNG_MULT if young_mult is None else young_mult
            ma = YM if (aa is not None and aa < YOUNG_AGE) else 1.0
            mb = YM if (ab is not None and ab < YOUNG_AGE) else 1.0
            if lvl_w is not None:
                lw = lvl_w.get(r["lvl"], lvl_w.get("other", 1.0)); ma *= lw; mb *= lw
            if opp_n is not None:
                ma *= min(1.0, elo_n[b] / opp_n); mb *= min(1.0, elo_n[a] / opp_n)
            per_set = None
            if set_mode:
                per_set = []
                for tok in r["score"].split():
                    mm = re.match(r"^(\d+)-(\d+)", tok)
                    if not mm: per_set = None; break
                    per_set.append((int(mm.group(1)), int(mm.group(2))))
                if per_set and (sum(x for x, _ in per_set), sum(y for _, y in per_set)) != (gw, gl):
                    per_set = None
            if per_set:
                # SET-BY-SET: re-predict before every set, update after it.
                first = True
                nsets = len(per_set)
                for si, (x, y) in enumerate(per_set):
                    dg = phi_gen[a] - phi_gen[b]
                    gp = BW * dg + (1 - BW) * (phi_surf[surf][a] - phi_surf[surf][b]) if use_surf else dg
                    qg = min(max(S.sigmoid(gp), 1e-6), 1 - 1e-6)
                    u = 0.0
                    if set_mode in ("games", "both"):
                        xe = x
                        if cap is not None and abs(x - y) > cap:        # blowout set: count margin only up to `cap`
                            xe = (x + y) / 2.0 + (cap / 2.0 if x > y else -cap / 2.0)
                        us = x - x + (xe - (x + y) * qg)
                        if tbw and {x, y} == {7, 6}:                     # tiebreak set: extra weight on who won the breaker
                            us += tbw * ((1.0 if x > y else 0.0) - qg)
                        if dw != 1.0 and si == nsets - 1 and nsets in (3, 5) and \
                                sum(1 for xx, yy in per_set[:-1] if xx > yy) == sum(1 for xx, yy in per_set[:-1] if yy > xx):
                            us *= dw                                     # deciding set
                        u += gscale * us
                    if set_mode in ("setbin", "both"):
                        u += sbin * ((1.0 if x > y else 0.0) - set_prob(round(qg, 4)))
                    pers = gb_persist if first else 1.0; first = False
                    phi_gen[a] = pers * phi_gen[a] + ga_rate * ma * u
                    phi_gen[b] = pers * phi_gen[b] - ga_rate * mb * u
                    if surf in SURFACES:
                        phi_surf[surf][a] = pers * phi_surf[surf][a] + ga_rate * ma * u
                        phi_surf[surf][b] = pers * phi_surf[surf][b] - ga_rate * mb * u
                if surf in SURFACES:
                    n_surf[surf][a] += 1; n_surf[surf][b] += 1
            else:
                sc_ = gscale * sc
                phi_gen[a] = gb_persist * phi_gen[a] + ga_rate * ma * sc_
                phi_gen[b] = gb_persist * phi_gen[b] - ga_rate * mb * sc_
                if surf in SURFACES:
                    phi_surf[surf][a] = gb_persist * phi_surf[surf][a] + ga_rate * ma * sc_
                    phi_surf[surf][b] = gb_persist * phi_surf[surf][b] - ga_rate * mb * sc_
                    n_surf[surf][a] += 1; n_surf[surf][b] += 1
            streak[a] += 1; streak[b] = 0

        # ---- bookkeeping for every row (incl. RET / W/O) ----
        if sets is not None:
            cumsets[(mid, a)] += sets[0] + sets[1]; cumsets[(mid, b)] += sets[0] + sets[1]
        if r["minutes"]:
            cummin[(mid, a)] += r["minutes"]; cummin[(mid, b)] += r["minutes"]
        prev_was_ret_loss[a] = False
        prev_was_ret_loss[b] = bool(r["ret"] or r["wo"])
        if r["ret"] or r["wo"]:
            last_inj[b] = when
        for p in (a, b):
            last_any[p] = when
            if p in return_date: n_since_return[p] += 1

    if snap_slams:
        import pickle
        pickle.dump(dict(snap=slam_snap, side={f"{k[0]}|{k[1]}": v for k, v in slam_side.items()}), open(f"{OUT}/slam_snapshots.pkl", "wb"))
    np.savez(f"{OUT}/{out_name}", when=np.array(out_when), base=np.array(out_base),
             bo=np.array(out_bo), X=np.array(out_feats, dtype=np.float64), feats=np.array(FEATS),
             **{"ex_" + k: np.array(v) for k, v in ex.items()})
    verbose and print("saved", len(out_base), "scored matches")


if __name__ == "__main__":
    main()
