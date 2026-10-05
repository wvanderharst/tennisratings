"""Chart trajectories for the serve/return model (Oct 4 2026).   python3 build_traj_sr.py atp|wta

Runs clone_engine.replay (sr_clone_fit.json config) and snapshots every selected player's PRE-match state through a hook,
in the same layout the chart already reads (player_rating_history_v23 / traj_data_v23):
  point = [date, level, level_match_surface, level_hard, level_clay, level_grass, stable_level, closing, age, tour_years,
           serve, return]
level = (serve + return) / 2, in the production game-logit units (so the chart's game->set->match chain still applies).
Surface views = 60% general + 40% that surface's own level; with no match on the surface, rank fill (serve and return
separately, monthly pool of active players who have played there). Field thresholds (top-10/20/50 of active players,
general and per surface) per month and per ISO week."""
import paths as P
import json, csv, sys, math, bisect, datetime
from collections import defaultdict, Counter
import numpy as np
import score_driven as S
import confounder_harness as H
import clone_engine as CE
from margin_model import parse_games

TOUR = sys.argv[1]
OUT = P.OUT; SCR = P.ROOT
if TOUR == "wta":
    H.DATA_PATHS = [f"{SCR}/wta_data/wta_tml.csv"]; H.birth = json.load(open(f"{SCR}/wta_data/wta_birth_ordinal_v2.json"))
CFE = json.load(open(f"{OUT}/confounder_fit_v21e.json")); CFG = json.load(open(f"{OUT}/sr_clone_fit.json"))[TOUR]
SURF = H.SURFACES; BW = 0.6
SLAM_NAME = {"Australian Championships": "Australian Open", "Australian Open": "Australian Open", "Australian Open-1": "Australian Open",
             "Australian Open-2": "Australian Open", "Roland Garros": "Roland Garros", "French Open": "Roland Garros", "US Open": "US Open",
             "Us Open": "US Open", "Wimbledon": "Wimbledon"}
allrows = H.load()
rows = [r for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])]
pos = {id(r): k for k, r in enumerate(allrows)}

# ---- who is plotted (same rules as before) ----
if TOUR == "atp":
    finals = json.load(open(f"{OUT}/slam_winner_ratings.json"))
    title_counts = Counter(r["champion"] for r in finals); finalist_counts = Counter()
    for r in finals: finalist_counts[r["champion"]] += 1; finalist_counts[r["runner_up"]] += 1
    career = Counter(); [career.update((r["a"], r["b"])) for r in allrows]
else:
    title_counts = Counter(); finalist_counts = Counter()
    with open(f"{SCR}/wta_data/wta_tml.csv", encoding="utf-8-sig", errors="replace") as f_:
        for r_ in csv.DictReader(f_):
            if (r_.get("tourney_level") or "").strip() == "G" and (r_.get("round") or "").strip() == "F":
                title_counts[r_["winner_name"].strip()] += 1
                finalist_counts[r_["winner_name"].strip()] += 1; finalist_counts[r_["loser_name"].strip()] += 1
    career = Counter()
    for r in allrows:
        if r["level"] == "tour" and r["rnd"] not in ("Q1", "Q2", "Q3", "Q4"): career.update((r["a"], r["b"]))
FINALISTS = set(finalist_counts)
# every tour-level title winner too (before 2024 the data has no Challengers, so e.g. a player who won one ATP title
# but spent most of his career at Challenger level can stay under 40 matches)
CHAMPS = {r["a"] for r in allrows if r["level"] == "tour" and r["rnd"] == "F" and r["lvl"] != "D"}
SELECTED = FINALISTS | CHAMPS | {p for p, k in career.items() if k >= 40}
DETAIL = FINALISTS | set(sorted(SELECTED - FINALISTS, key=lambda p: -career[p])[:max(170 - len(FINALISTS), 0)])
print(len(SELECTED), "players,", len(DETAIL), "with weekly detail")

ST_DECAY = 0.5 ** (1 / 270.0)
stab = {}; stab_t = {}
# all-time peaks: each player's highest pre-match level (after PEAK_MIN matches, so early-career noise doesn't count),
# from 1992 for the women (the chart's start; 1990-91 is warm-up)
PEAK_MIN = 30; PEAK_FROM = datetime.date(1968 if TOUR == "atp" else 1992, 1, 1)
peak = {}
mc = defaultdict(int); first = {}
fill = {}                       # monthly display pools: surface -> (sorted sg, sorted ss, sorted rg, sorted rs)
thr_m, thr_w = {}, {}
snaps = defaultdict(dict); snaps_f = defaultdict(dict); markers = []
state = {}; ptr = [0]; queries = defaultdict(list); CUR_I = [0]

def qmap(x, gs, ss_):
    n_ = len(gs); xx = bisect.bisect_left(gs, x) / n_ * n_ - 0.5
    if xx <= 0: return ss_[0]
    if xx >= n_ - 1: return ss_[-1]
    i_ = int(xx); f_ = xx - i_
    return ss_[i_] * (1 - f_) + ss_[i_ + 1] * f_

def refresh_fill(when, X):
    ym = (when.year, when.month)
    if fill.get("ym") == ym: return
    fill.clear(); fill["ym"] = ym
    for s_ in SURF:
        pool = [p for p, d in X["last"].items() if X["ns"][s_][p] >= 1 and (when - d).days <= 365]
        if len(pool) >= 30:
            fill[s_] = tuple(sorted(D[p] for p in pool) for D in (X["sg"], X["ss"][s_], X["rg"], X["rs"][s_]))

def lvl_surf(p, s_, X):
    """(serve, return) blended on surface s_ (60% general), rank-filled when never played there."""
    sv, rt = X["sg"][p], X["rg"][p]
    if s_ not in SURF: return sv, rt
    if CFG.get("gap"): return sv + CFG["lam"] * X["ss"][s_][p], rt + CFG["lam"] * X["rs"][s_][p]
    if X["ns"][s_][p] >= 1: return BW * sv + (1 - BW) * X["ss"][s_][p], BW * rt + (1 - BW) * X["rs"][s_][p]
    t = fill.get(s_)
    if not t: return sv, rt
    return BW * sv + (1 - BW) * qmap(sv, t[0], t[1]), BW * rt + (1 - BW) * qmap(rt, t[2], t[3])

def thresholds(when, X):
    act = [p for p, k in mc.items() if k >= 30 and p in X["last"] and (when - X["last"][p]).days <= 365]
    out = []
    g = sorted(((X["sg"][p] + X["rg"][p]) / 2 for p in act), reverse=True)
    at = lambda v, k: round(v[min(k, len(v)) - 1], 4) if v else 0.0
    out += [at(g, 10), at(g, 20), at(g, 50)]; s515 = [at(g, k) for k in range(5, 16)]
    for s_ in SURF:
        v = sorted((sum(lvl_surf(p, s_, X)) / 2 for p in act), reverse=True)
        out += [at(v, 10), at(v, 20), at(v, 50)]; s515 += [at(v, k) for k in range(5, 16)]
    return out + [round(x, 3) for x in s515]     # [0..11]: top-10/20/50 general + per surface; [12..55]: ranks 5-15, general/Hard/Clay/Grass

def stable(p, when, cur):
    t = stab_t.get(p)
    if t is None: stab[p] = cur
    else:
        d = (when - t).days
        if d > 0: a_ = ST_DECAY ** d; stab[p] = a_ * stab[p] + (1 - a_) * cur
    stab_t[p] = when; return stab[p]

def snap(p, r, X):
    when = r["when"]; lv = (X["sg"][p] + X["rg"][p]) / 2
    sm = sum(lvl_surf(p, r["surf"], X)) / 2
    sh, sc_, sgr = (sum(lvl_surf(p, s_, X)) / 2 for s_ in SURF)
    ag = H.age_at(p, when); ty = round((when - first[p]).days / 365.25, 3) if p in first else 0.0
    return [when.isoformat(), round(lv, 4), round(sm, 4), round(sh, 4), round(sc_, 4), round(sgr, 4), round(stable(p, when, lv), 4),
            round(X["c"][p], 4), round(ag, 3) if ag is not None else None, ty, round(X["sg"][p], 4), round(X["rg"][p], 4), 0.0]

def marker(r, X, p, opp, won):
    when = r["when"]; lv = (X["sg"][p] + X["rg"][p]) / 2
    ag = H.age_at(p, when)
    markers.append(dict(cl=0.0, player=p, date=when.isoformat(), slam=SLAM_NAME[r["tname"]], phi_gen=round(lv, 4),
                        bphi_match=round(sum(lvl_surf(p, r["surf"], X)) / 2, 4),
                        b_hard=round(sum(lvl_surf(p, "Hard", X)) / 2, 4), b_clay=round(sum(lvl_surf(p, "Clay", X)) / 2, 4),
                        b_grass=round(sum(lvl_surf(p, "Grass", X)) / 2, 4), sphi=round(stab.get(p, lv), 4), c=round(X["c"][p], 4),
                        age=round(ag, 3) if ag is not None else None, tour_years=round((when - first[p]).days / 365.25, 3) if p in first else 0.0,
                        opponent=opp, won=won, sv=round(X["sg"][p], 4), rt=round(X["rg"][p], 4)))
    queries[CUR_I[0]].append((markers[-1], "cl", p))

def is_final(r): return r["lvl"] == "G" and r["rnd"] == "F" and r["tname"] in SLAM_NAME

def hook(i, r, X):
    when, a, b = r["when"], r["a"], r["b"]; CUR_I[0] = i
    refresh_fill(when, X)
    k = pos[r.get("_oid", id(r))]
    for j in range(ptr[0], k):                   # unscored rows in between (RET/W-O): only Slam-final markers
        u = allrows[j]
        for p in (u["a"], u["b"]):
            if p in SELECTED and p not in first: first[p] = u["when"]
        if is_final(u):
            if u["a"] in SELECTED: marker(u, X, u["a"], u["b"], True)
            if u["b"] in SELECTED: marker(u, X, u["b"], u["a"], False)
    ptr[0] = k + 1
    for p in (a, b):
        if p in SELECTED and p not in first: first[p] = when
    ym = when.strftime("%Y-%m"); iw = when.isocalendar(); wk = "%04d-W%02d" % (iw[0], iw[1])
    if ym not in thr_m: thr_m[ym] = thresholds(when, X)
    need_w = a in DETAIL or b in DETAIL
    if need_w and wk not in thr_w: thr_w[wk] = thresholds(when, X)
    if is_final(r):
        if a in SELECTED: marker(r, X, a, b, True)
        if b in SELECTED: marker(r, X, b, a, False)
    for p in (a, b):
        if p in SELECTED:
            s_ = snap(p, r, X); snaps[p][ym] = s_; queries[i].append((s_, 12, p))
            if p in DETAIL: snaps_f[p][wk] = s_
    if when >= PEAK_FROM:
        for p in (a, b):
            if mc[p] >= PEAK_MIN:
                lv = (X["sg"][p] + X["rg"][p]) / 2
                if p not in peak or lv > peak[p][0]:
                    ag = H.age_at(p, when)
                    peak[p] = (lv, when.isoformat(), round(ag, 1) if ag is not None else None, mc[p], thr_m[ym][0], X["sg"][p], X["rg"][p])
    mc[a] += 1; mc[b] += 1

EXT, RIDX, _ = CE.build_ext(allrows)
carry = dict(stab=stab, stab_t=stab_t, mc=mc, first=first, fill=fill, thr_m=thr_m, thr_w=thr_w, snaps=snaps, snaps_f=snaps_f,
             markers=markers, state=state, ptr=ptr, queries=queries, CUR_I=CUR_I, peak=peak)
z, _ = CE.replay_ckpt(f"traj_{TOUR}", allrows, EXT, H.age_at, CFE["init_rank"], CFE["init_prior"], carry=carry,
                      deps=(sorted(SELECTED), sorted(DETAIL)), hook=hook, **CFG); z = z[RIDX]
# deciding-set clutch (same running definition as the odds export), filled in pre-match for every snapshot
dsum = defaultdict(float); dn = defaultdict(int)
for i, r in enumerate(rows):
    for obj, k, p in queries.get(i, ()): obj[k] = round(dsum[p] / (dn[p] + 50), 4)
    a, b, bo = r["a"], r["b"], r["bo"]
    q = H.inv_match(round(min(max(1 / (1 + math.exp(-z[i])), 0.001), 0.999), 4), bo)
    if len(r["score"].split()) == 2 * (2 if bo == 3 else 3) - 1:
        dsum[a] += 1 - q; dsum[b] -= 1 - q; dn[a] += 1; dn[b] += 1
names = sorted(snaps, key=lambda p: (-title_counts.get(p, 0), -finalist_counts.get(p, 0), -career.get(p, 0)))
players = [dict(name=p, titles=title_counts.get(p, 0), finals=finalist_counts.get(p, 0), matches=career.get(p, 0),
                pts=[snaps[p][k] for k in sorted(snaps[p])], ptsFine=[snaps_f[p][k] for k in sorted(snaps_f[p])]) for p in names]
data = dict(players=players, slam_markers=markers, thr_month=thr_m, thr_week=thr_w, cal=dict(i=H.CAL_I, s=H.CAL_S), model="serve_return")
fn = f"{OUT}/traj_data_{TOUR}_sr.json"
json.dump(data, open(fn, "w"), separators=(",", ":"))
import os
print("wrote", fn, round(os.path.getsize(fn) / 1e6, 1), "MB;", len(players), "players,", len(markers), "slam markers")

# ---- all-time peaks (top 100) -> work/peaks_{tour}.json (copied to the site by build_site_v8.py) ----
R3 = lambda x: round(float(x), 3)
top = sorted(peak.items(), key=lambda kv: -kv[1][0])[:100]
json.dump([dict(n=p, lv=R3(v[0]), d=v[1], age=v[2], m=v[3], vs10=R3(v[0] - v[4]), sv=R3(v[5]), rt=R3(v[6]),
                slams=title_counts.get(p, 0), matches=career.get(p, 0)) for p, v in top],
          open(f"{OUT}/peaks_{TOUR}.json", "w"), separators=(",", ":"))
print("peaks:", ", ".join(f"{p} {v[0]:.2f} ({v[1][:4]})" for p, v in top[:5]))
