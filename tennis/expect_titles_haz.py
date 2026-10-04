"""Actual vs predicted titles per player.  python3 expect_titles.py atp|wta
Every tour-level knockout event (no Davis/Fed Cup ties, no round-robin events or season finals, final played, 8+ entrants):
each entrant's state frozen on the eve of the event (as slam_analysis.py), title chances from seeded random draws
(seeds = official rank; byes go to the top seeds), the rating chain of the model (surface level, holds/sets, closing),
best-of as played. Output: per player Slam and all-event entries, titles and expected titles."""
import paths as P
import json, sys, math, collections
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
LAM, HM = CFG["lam"], CFG["hm"]
allrows = H.load()
rows = [r for r in allrows if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"])]
QR = ("Q1", "Q2", "Q3", "Q4")
ev = collections.defaultdict(list)
for r in allrows:
    if r["level"] == "tour" and r["rnd"] not in QR and r["lvl"] not in ("D", "F"): ev[r["tid"]].append(r)
ev = {k: L for k, L in ev.items() if any(m["rnd"] == "F" for m in L) and not any(m["rnd"] == "RR" for m in L)
      and len({p for m in L for p in (m["a"], m["b"])}) >= 8}
first_id = {}
for r in rows:
    if r["tid"] in ev and r["tid"] not in first_id and r["rnd"] not in QR: first_id[r["tid"]] = id(r)
snap = {}
def hook(i, r, X):
    tid = r["tid"]
    if first_id.get(tid) != id(r): return
    L = ev[tid]; when = r["when"]; surf = r["surf"]; rank = {}; ents = []
    for m in L:
        for p, rk in ((m["a"], m["rank_a"]), (m["b"], m["rank_b"])):
            if p not in ents: ents.append(p)
            if rk and p not in rank: rank[p] = rk
    st = {}
    for p in ents:
        if p in X["sg"] and (p in X["last"] or p in (r["a"], r["b"])):
            f = 1.0
            if p in X["last"] and p not in (r["a"], r["b"]): f = 0.9995 ** max(0, (when - X["last"][p]).days - 30)
            gs = X["ss"][surf][p] if surf in X["ss"] else 0.0; gr = X["rs"][surf][p] if surf in X["rs"] else 0.0
            st[p] = (f * (X["sg"][p] + LAM * gs), f * (X["rg"][p] + LAM * gr), f * X["c"][p])
        else:
            rk = rank.get(p); v = CFE["init_prior"].get("tour", -0.05)
            if rk: v = min(max(CFE["init_rank"][0] - CFE["init_rank"][1] * (math.log(rk) - math.log(100.0)), -1.5), 1.0)
            st[p] = (v, v, 0.0)
    bo = max(m["bo"] for m in L)
    snap[tid] = dict(when=when, st=st, rank=rank, mu=X["mu"][surf or "Hard"], bo=bo, slam=r["lvl"] == "G")
EXT, RIDX, _ = CE.build_ext(allrows)
CE.replay_ckpt(f"titles_{T}", allrows, EXT, H.age_at, CFE["init_rank"], CFE["init_prior"], carry=dict(snap=snap), hook=hook, **CFG)
RF = json.load(open(f"{OUT}/ret_fit_{T}.json"))

def pwin(A, B, mu, bo):
    sa, ra, ca = A; sb, rb, cb = B
    ha = min(max(S.sigmoid(mu + HM * (sa - rb) / 2.132), .02), .98); hb = min(max(S.sigmoid(mu + HM * (sb - ra) / 2.132), .02), .98)
    qs = 0.5 * (sr_lib.set_prob(round(ha, 3), round(hb, 3)) + 1 - sr_lib.set_prob(round(hb, 3), round(ha, 3)))
    qs = min(max(S.sigmoid(S.logit(min(max(qs, 1e-6), 1 - 1e-6)) + ca - cb), 1e-6), 1 - 1e-6)
    return S.sigmoid(H.CAL_I + H.CAL_S * S.logit(min(max(S.match_from_set_prob(qs, bo), 1e-6), 1 - 1e-6)))
def seed_slots(N, ns):
    slots = [0, N - 1]; k = 2
    while len(slots) < ns:
        size = N // (2 ** k)
        for sct in range(2 ** k):
            lo, hi = sct * size, (sct + 1) * size - 1
            if not any(lo <= s <= hi for s in slots): slots.append(lo if sct % 2 else hi)
        k += 1
    return slots[:ns]
rng = np.random.default_rng(11); RS = [None]
agg = collections.defaultdict(lambda: dict(se=0, st=0, sx=0.0, e=0, t=0, x=0.0))
done = 0; recs = []
import ckpt
ITEMS = sorted(snap.items(), key=lambda kv: kv[1]["when"])
LOOP = ckpt.Loop(f"titlesim_{T}", allrows, (CFG, CFE, RF), [sp["when"] for _, sp in ITEMS])
if LOOP.saved:
    rng.bit_generator.state = LOOP.saved["rng"]; agg.update(LOOP.saved["agg"]); recs = LOOP.saved["recs"]; done = LOOP.saved["done"]
loop_state = lambda: dict(rng=rng.bit_generator.state, agg={p: dict(v) for p, v in agg.items()}, recs=recs, done=done)
for k_ev in range(LOOP.start, len(ITEMS)):
    LOOP.at(k_ev, loop_state)
    tid, sp = ITEMS[k_ev]
    L = ev[tid]; champ = [m for m in L if m["rnd"] == "F"][0]["a"]
    P = list(sp["st"]); n = len(P); idx = {p: i for i, p in enumerate(P)}
    N = 1 << (n - 1).bit_length(); M = np.full((N, N), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            v = pwin(sp["st"][P[i]], sp["st"][P[j]], sp["mu"], sp["bo"]); M[i, j] = v; M[j, i] = 1 - v
    M[:n, n:] = 1.0; M[n:, :n] = 0.0
    ns = min((32 if sp["when"].year >= 2001 else 16) if sp["slam"] else max(2, N // 4), N // 2)
    ranked = sorted([p for p in P if p in sp["rank"]], key=lambda p: sp["rank"][p])[:ns]
    sl = seed_slots(N, len(ranked)); seeds = [idx[p] for p in ranked]
    byes = list(range(n, N)); nb = len(byes); seedset = set(sl)
    bye_slots = []
    for s0 in sl:                                             # byes next to the top seeds first
        if len(bye_slots) == nb: break
        if (s0 ^ 1) not in seedset: bye_slots.append(s0 ^ 1)
    rest = [s0 for s0 in range(N) if s0 not in seedset and s0 not in bye_slots]
    if len(bye_slots) < nb:
        extra = list(rng.choice(rest, nb - len(bye_slots), replace=False)); bye_slots += extra; rest = [s0 for s0 in rest if s0 not in extra]
    free = np.array(rest); others = np.array([i for i in range(n) if i not in set(seeds)])
    assert len(free) == len(others) and len(set(sl) | set(bye_slots)) == len(sl) + nb, (n, N, len(free), len(others))
    NS = 3000 if sp["slam"] else 1200
    D = np.empty((NS, N), int)
    for t in range(NS):
        s_perm = list(sl); k0 = 2
        while k0 < len(sl):
            k1 = min(2 * k0, len(sl)); seg = s_perm[k0:k1]; rng.shuffle(seg); s_perm[k0:k1] = seg; k0 = k1
        D[t, s_perm] = seeds; D[t, bye_slots] = byes; D[t, free] = rng.permutation(others)
    titles = []
    for rr_ in RS:                                   # retirement / walkover hazard: with prob r the match is a coin flip
        rr_, bb_ = (RF["r_slam"], max(0.0, RF["b_slam"])) if sp["slam"] else (RF["r_other"], max(0.0, RF["b_other"]))
        Zl = np.log(np.clip(M, 1e-9, 1 - 1e-9) / np.clip(1 - M, 1e-9, 1))
        Mr = (1 - rr_) * M + rr_ / (1 + np.exp(-bb_ * Zl)); Mr[:n, n:] = 1.0; Mr[n:, :n] = 0.0
        B = D
        while B.shape[1] > 1:
            x, y = B[:, 0::2], B[:, 1::2]; B = np.where(rng.random(x.shape) < Mr[x, y], x, y)
        titles.append(np.bincount(B[:, 0], minlength=N)[:n] / NS)
    title = titles[0]
    for p in P:
        a = agg[p]; tp = float(title[idx[p]]); w = int(p == champ)
        a["e"] += 1; a["t"] += w; a["x"] += tp
        if sp["slam"]: a["se"] += 1; a["st"] += w; a["sx"] += tp
        recs.append((sp['when'].year, int(sp['slam']), n, w) + tuple(round(float(tt[idx[p]]), 5) for tt in titles))
    done += 1
    if done % 500 == 0: print(done, "events", sp["when"], flush=True)
LOOP.at(len(ITEMS), loop_state)
json.dump({p: dict(v) for p, v in agg.items()}, open(f"{OUT}/titles_expected_{T}.json", "w")); json.dump(recs, open(f"{OUT}/title_chances_{T}.json", "w"))
print(T, done, "events done")
for p in sorted(agg, key=lambda p: -agg[p]["t"])[:10]:
    a = agg[p]; print(f"  {p:22s} Slams {a['st']:2d} exp {a['sx']:5.1f} ({a['se']} entered) | titles {a['t']:3d} exp {a['x']:6.1f} ({a['e']} events)")
