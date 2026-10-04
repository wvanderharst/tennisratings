"""Retirements/walkovers: replay with update-only rows (sr_clone_fit.json), then estimate the retirement risk r and
slope beta used in the odds (direct estimate on the training years).   python3 ret_odds_fit.py atp|wta"""
import paths as P
import sys, json, datetime, numpy as np
from scipy.optimize import minimize
import confounder_harness as H, score_driven as S, clone_engine as CE
from margin_model import parse_games
T = sys.argv[1]; SCR = P.ROOT
if T == "wta": H.DATA_PATHS = [f"{SCR}/wta_data/wta_tml.csv"]; H.birth = json.load(open(f"{SCR}/wta_data/wta_birth_ordinal_v2.json"))
CFE = json.load(open(P.OUT + "/confounder_fit_v21e.json")); CFG = json.load(open(P.OUT + "/sr_clone_fit.json"))[T]
allrows = H.load(); ext = []; rated_idx = []; kind = []
for r in allrows:
    if S.parse_sets(r["score"]) and parse_games(r["score"]) and not (r["ret"] or r["wo"] or r["dq"]):
        rated_idx.append(len(ext)); ext.append(r); kind.append(0)
    elif (r["ret"] or r["wo"]) and not r["dq"]:
        q = dict(r); q["upd_only"] = True; q["pts"] = r["pts"] if (r["ret"] and not r["wo"]) else None; ext.append(q); kind.append(1 if r["ret"] else 2)
rated_idx = np.array(rated_idx); kind = np.array(kind)
wext = np.array([r["when"].toordinal() for r in ext]); lvl = np.array([r["lvl"] == "G" for r in ext]); bo = np.array([r["bo"] for r in ext])
C = np.load(f"cache_{T}.npz"); corr, when = C["corr"], C["when"]; fm, vm, tr, te = C["fm"], C["vm"], C["tr"], C["te"]
ALL = when >= datetime.date(2005, 1, 1).toordinal(); L = lambda z: np.logaddexp(0, -z)
def fitz(F, m):
    def g(w):
        zz = F[m] @ w; p = 1 / (1 + np.exp(-zz))
        return np.logaddexp(0, -zz).sum() + 5e-5 * (w[1:] @ w[1:]), -(F[m].T @ (1 - p)) + np.r_[0, 1e-4 * w[1:]]
    w0 = np.zeros(F.shape[1]); w0[0] = 1; return minimize(g, w0, jac=True, method="L-BFGS-B").x
split = when[te].min(); vcut = when[vm].min()
best = None
for k, side in ((CFG.get("k_ret", 1.0), CFG.get("ret_side", "winner")),):     # chosen setting (sr_clone_fit.json)
    rows = ext
    z, _ = CE.replay_ckpt(f"rating_{T}", allrows, rows, H.age_at, CFE["init_rank"], CFE["init_prior"], **dict(CFG, k_ret=k, ret_side=side))
    zr = z[rated_idx]; F = np.column_stack([zr, corr]); wv = fitz(F, fm); wt = fitz(F, tr); wa = fitz(F, ALL)
    sel = L(F[vm] @ wv).mean()
    print(f"  partial points weight {k:4.2f} ({side:6s}): selection {sel:.5f} | held-out {L((F @ wt)[te]).mean():.5f} | 2005+ {L((F @ wa)[ALL]).mean():.5f}", flush=True)
    if best is None or sel < best[0]: best = (sel, k, side, z, wt)
sel, k, side, z, wt = best
print(f"chosen on selection years: weight {k}, {side}")
# ---- who advances: completed + retirements + walkovers; main logit z (rating chain) times fitted slope ----
zc = z.copy(); zc[rated_idx] = np.column_stack([z[rated_idx], corr]) @ wt     # full model where corrections exist
trX = wext < split; teX = ~trX
def nll(par, m, grp):
    r = 1 / (1 + np.exp(-par[grp])); bb = par[-1]
    p = (1 - r) / (1 + np.exp(-zc[m])) + r / (1 + np.exp(-bb * zc[m]))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))
grp = np.where(lvl, 0, 1)    # 0 = Slam, 1 = other
def fitmix(m):
    f = lambda par: nll(np.r_[par[:2][grp[m]], par[2]] if False else par, m, grp[m])
    res = minimize(lambda par: -np.mean(np.log(np.clip((1 - 1/(1+np.exp(-par[grp[m]]))) / (1 + np.exp(-zc[m])) + 1/(1+np.exp(-par[grp[m]])) / (1 + np.exp(-par[2] * zc[m])), 1e-9, 1))),
                   np.array([-3.5, -3.5, 0.2]), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-9))
    return res.x
par = fitmix(trX)
r_s, r_o, bb = 1 / (1 + np.exp(-par[0])), 1 / (1 + np.exp(-par[1])), par[2]
print(f"retirement/walkover risk per match: Slams {r_s*100:.2f}%, other {r_o*100:.2f}% | strength still counts with slope {bb:.3f} when it happens")
def lls(m, mix):
    if mix:
        rr = np.where(lvl[m], r_s, r_o); p = (1 - rr) / (1 + np.exp(-zc[m])) + rr / (1 + np.exp(-bb * zc[m]))
    else: p = 1 / (1 + np.exp(-zc[m]))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))
for lab, m in (("held-out, all matches", teX), ("held-out, completed", teX & (kind == 0)), ("held-out, retirements+walkovers", teX & (kind > 0))):
    print(f"  {lab:34s} n={int(m.sum()):6d} | odds ignoring retirements {lls(m, False):.5f} | with retirement risk {lls(m, True):.5f}")
json.dump(dict(k_ret=k, ret_side=side, r_slam=float(r_s), r_other=float(r_o), b_ret=float(bb)), open(P.OUT + f"/ret_fit_{T}.json", "w"))

# ---- direct estimate: r = share of matches ending early (training years), b = logistic slope of who advances in those matches ----
print("direct estimate (training years):")
out = {}
for lab, g in (("Slam", lvl), ("other", ~lvl)):
    m = trX & g
    rr = float(np.mean(kind[m] > 0)); mm = m & (kind > 0)
    from scipy.optimize import minimize_scalar
    bb2 = minimize_scalar(lambda b_: np.mean(np.logaddexp(0, -b_ * zc[mm])), bounds=(-2, 2), method="bounded").x
    out[lab] = (rr, bb2); print(f"  {lab}: early-ending share {rr*100:.2f}% of {int(m.sum())} matches; in those, favourite advances with slope {bb2:.3f} (vs 1 in completed)")
def lls2(m, mix):
    if mix:
        rr = np.where(lvl[m], out['Slam'][0], out['other'][0]); b2 = np.where(lvl[m], out['Slam'][1], out['other'][1])
        p = (1 - rr) / (1 + np.exp(-zc[m])) + rr / (1 + np.exp(-b2 * zc[m]))
    else: p = 1 / (1 + np.exp(-zc[m]))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))
for lab, m in (("held-out, all matches", teX), ("held-out, completed", teX & (kind == 0)), ("held-out, retirements+walkovers", teX & (kind > 0))):
    print(f"  {lab:34s} n={int(m.sum()):6d} | ignoring retirements {lls2(m, False):.5f} | with retirement risk {lls2(m, True):.5f}")
d = json.load(open(P.OUT + f"/ret_fit_{T}.json")); d.update(r_slam=out["Slam"][0], b_slam=out["Slam"][1], r_other=out["other"][0], b_other=out["other"][1])
json.dump(d, open(P.OUT + f"/ret_fit_{T}.json", "w"))
