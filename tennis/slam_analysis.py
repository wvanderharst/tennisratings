"""Grand Slam champions seen from the eve of the tournament.  python3 slam_analysis.py atp|wta
For every Slam: every main-draw entrant's state frozen before the event's first match (layoff decay to that date), the
rating chain of the model (surface level, holds/sets, closing; no h2h, no corrections), best of 5 men / 3 women.
Outputs per Slam: champion's rank among entrants by surface level, simulated title probabilities (seeded random draws),
and the champion's actual path: opponents' pre-event levels and ranks, chance of winning that exact path."""
import paths as P
import json, sys, math, pickle, collections
import numpy as np
import score_driven as S
import confounder_harness as H
import clone_engine as CE
import sr_lib
from margin_model import parse_games
T = sys.argv[1]
OUT = P.OUT; SCR = P.ROOT
if T == "wta":
    H.DATA_PATHS = [f"{SCR}/wta_data/wta_tml.csv"]; H.birth = json.load(open(f"{SCR}/wta_data/wta_birth_ordinal_v2.json"))
CFE = json.load(open(f"{OUT}/confounder_fit_v21e.json")); CFG = json.load(open(f"{OUT}/sr_clone_fit.json"))[T]
LAM, HM = CFG["lam"], CFG["hm"]; BO = 5 if T == "atp" else 3; START = 1968 if T == "atp" else 1992
allrows = H.load()
rows = [r for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])]
QR = ("Q1", "Q2", "Q3", "Q4")
ev = collections.defaultdict(list)
for r in allrows:
    if r["lvl"] == "G" and r["rnd"] not in QR and r["when"].year >= START: ev[r["tid"]].append(r)
first_id = {}
for tid, L in ev.items():
    rated = [r for r in rows if False]  # placeholder
for r in rows:
    if r["tid"] in ev and r["tid"] not in first_id and r["rnd"] not in QR: first_id[r["tid"]] = id(r)
snap = {}
def hook(i, r, X):
    tid = r["tid"]
    if first_id.get(tid) != id(r): return
    L = ev[tid]; when = r["when"]; surf = r["surf"]
    ents = {}; rank = {}
    for m in L:
        for p, rk in ((m["a"], m["rank_a"]), (m["b"], m["rank_b"])):
            ents[p] = 1
            if rk and p not in rank: rank[p] = rk
    st = {}
    for p in ents:
        if p in X["sg"] and (p in X["last"] or p in (r["a"], r["b"])):
            f = 1.0
            if p in X["last"] and p not in (r["a"], r["b"]):
                idle = max(0, (when - X["last"][p]).days - 30); f = 0.9995 ** idle
            gs = X["ss"][surf][p] if surf in X["ss"] else 0.0; gr = X["rs"][surf][p] if surf in X["rs"] else 0.0
            st[p] = (f * (X["sg"][p] + LAM * gs), f * (X["rg"][p] + LAM * gr), f * X["c"][p])
        else:                                   # debut at this Slam: debut value
            rk = rank.get(p); v = CFE["init_prior"].get("tour", -0.05)
            if rk: v = min(max(CFE["init_rank"][0] - CFE["init_rank"][1] * (math.log(rk) - math.log(100.0)), -1.5), 1.0)
            st[p] = (v, v, 0.0)
    snap[tid] = dict(when=when, surf=surf, st=st, rank=rank, mu=X["mu"][surf or "Hard"], name=r["tname"])
EXT, RIDX, _ = CE.build_ext(allrows)
CE.replay_ckpt(f"slams_{T}", allrows, EXT, H.age_at, CFE["init_rank"], CFE["init_prior"], carry=dict(snap=snap), hook=hook, **CFG)
RF = json.load(open(f"{OUT}/ret_fit_{T}.json")); RR, BB = RF["r_slam"], max(0.0, RF["b_slam"])

def pwin(A, B, mu):
    sa, ra, ca = A; sb, rb, cb = B
    ha = min(max(S.sigmoid(mu + HM * (sa - rb) / 2.132), .02), .98); hb = min(max(S.sigmoid(mu + HM * (sb - ra) / 2.132), .02), .98)
    qs = 0.5 * (sr_lib.set_prob(round(ha, 3), round(hb, 3)) + 1 - sr_lib.set_prob(round(hb, 3), round(ha, 3)))
    qs = min(max(S.sigmoid(S.logit(min(max(qs, 1e-6), 1 - 1e-6)) + ca - cb), 1e-6), 1 - 1e-6)
    return S.sigmoid(H.CAL_I + H.CAL_S * S.logit(min(max(S.match_from_set_prob(qs, BO), 1e-6), 1 - 1e-6)))

def seed_slots(N, ns):
    """slot index for seeds 1..ns: seed 1 top, seed 2 bottom, then each tier into the empty sections of the next size."""
    slots = [0, N - 1]; k = 2
    while len(slots) < ns:
        size = N // (2 ** k); new = []
        for sct in range(2 ** k):
            lo, hi = sct * size, (sct + 1) * size - 1
            if not any(lo <= s <= hi for s in slots): new.append(lo if sct % 2 else hi)
        slots += new; k += 1
    return slots[:ns]
rng = np.random.default_rng(7); NS = 4000
res = []
import ckpt
ITEMS = sorted(snap.items(), key=lambda kv: kv[1]["when"])
LOOP = ckpt.Loop(f"slamsim_{T}", allrows, (CFG, CFE, RF), [sp["when"] for _, sp in ITEMS])
if LOOP.saved:
    rng.bit_generator.state = LOOP.saved["rng"]; res = LOOP.saved["res"]
    for line in LOOP.saved["log"]: print(line)
LOG = list(LOOP.saved["log"]) if LOOP.saved else []
loop_state = lambda: dict(rng=rng.bit_generator.state, res=res, log=LOG)
for k_ev in range(LOOP.start, len(ITEMS)):
    LOOP.at(k_ev, loop_state)
    tid, sp = ITEMS[k_ev]
    L = ev[tid]; fin = [m for m in L if m["rnd"] == "F"]
    if not fin: continue
    champ = fin[0]["a"]; P = list(sp["st"]); n = len(P); idx = {p: i for i, p in enumerate(P)}
    N = 1 << (n - 1).bit_length()
    M = np.full((N, N), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            v = pwin(sp["st"][P[i]], sp["st"][P[j]], sp["mu"]); M[i, j] = v; M[j, i] = 1 - v
    Zl = np.log(np.clip(M, 1e-9, 1 - 1e-9) / np.clip(1 - M, 1e-9, 1)); M = (1 - RR) * M + RR / (1 + np.exp(-BB * Zl))   # retirement risk
    M[:n, n:] = 1.0; M[n:, :n] = 0.0
    lvl = np.array([(sp["st"][p][0] + sp["st"][p][1]) / 2 for p in P])
    order = np.argsort(-lvl); lrank = np.empty(n, int); lrank[order] = np.arange(1, n + 1)
    # seeded random draws
    ns = min(32 if sp["when"].year >= 2001 else 16, N // 4)
    ranked = sorted([p for p in P if p in sp["rank"]], key=lambda p: sp["rank"][p])[:ns]
    sl = seed_slots(N, len(ranked)); seeds = [idx[p] for p in ranked]
    others = np.array([i for i in range(N) if i not in set(seeds)])
    free = np.array([s for s in range(N) if s not in set(sl)])
    D = np.empty((NS, N), int)
    for t in range(NS):
        # within each seeding tier the slots are shuffled
        s_perm = list(sl); k0 = 2
        while k0 < len(sl):
            k1 = min(2 * k0, len(sl)); seg = s_perm[k0:k1]; rng.shuffle(seg); s_perm[k0:k1] = seg; k0 = k1
        D[t, s_perm] = seeds; D[t, free] = rng.permutation(others)
    B = D.copy()
    while B.shape[1] > 1:
        x, y = B[:, 0::2], B[:, 1::2]; pw = M[x, y]
        B = np.where(rng.random(pw.shape) < pw, x, y)
    title = np.bincount(B[:, 0], minlength=N)[:n] / NS
    ci = idx[champ]; trank = int((title > title[ci]).sum()) + 1; fav = int(np.argmax(title))
    path = sorted([m for m in L if m["a"] == champ], key=lambda m: H.ROUND_ORD.get(m["rnd"], 2))
    opp = [m["b"] for m in path]
    pp = [M[ci, idx[o]] for o in opp if o in idx]
    res.append(dict(tid=tid, slam=sp["name"], year=sp["when"].year, surf=sp["surf"], champ=champ, n=n,
                    lvl_rank=int(lrank[ci]), title_p=float(title[ci]), title_rank=trank, fav=P[fav], fav_p=float(title[fav]),
                    gap_to_best=float(lvl[order[0]] - lvl[ci]), champ_lvl=float(lvl[ci]), best_lvl=float(lvl[order[0]]), field_lvl=[round(float(v), 3) for v in lvl], opp=opp, opp_lvl_rank=[int(lrank[idx[o]]) for o in opp if o in idx],
                    opp_lvl=[float(lvl[idx[o]]) for o in opp if o in idx], path_p=float(np.prod(pp)), match_p=[float(x) for x in pp],
                    n_top10=int(sum(1 for o in opp if o in idx and lrank[idx[o]] <= 10)), rounds=[m["rnd"] for m in path]))
    LOG.append(f"{sp['when'].year} {sp['name'][:16]:16s} {champ:22s} lvl#{lrank[ci]:<3d} title {title[ci]*100:5.1f}% (#{trank}) fav {P[fav]} {title[fav]*100:4.1f}% path {np.prod(pp)*100:5.1f}%")
    print(LOG[-1], flush=True)
LOOP.at(len(ITEMS), loop_state)
json.dump(res, open(f"{OUT}/slam_eve_{T}.json", "w"), indent=0)
