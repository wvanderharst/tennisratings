"""Charts + CSV from slam_eve_{atp,wta}.json (slam_analysis.py)."""
import paths as P
import json, csv, numpy as np, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
OUT = P.OUT
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#c9c7c0", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
D = {T: json.load(open(f"{OUT}/slam_eve_{T}.json")) for T in ("atp", "wta")}
BG = "#fcfcfb"
# ---------- chart 1: where champions ranked + title chance over time ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw=dict(width_ratios=[1, 1.25])); fig.patch.set_facecolor(BG)
groups = [("Men 1968–89", "atp", 1968, 1989), ("Men 1990–2004", "atp", 1990, 2004), ("Men 2005–26", "atp", 2005, 2026),
          ("Women 1992–2004", "wta", 1992, 2004), ("Women 2005–15", "wta", 2005, 2015), ("Women 2016–26", "wta", 2016, 2026)]
cats = [("No. 1", 1, 1), ("No. 2", 2, 2), ("No. 3–4", 3, 4), ("No. 5–8", 5, 8), ("No. 9–16", 9, 16), ("No. 17+", 17, 999)]
ramp = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]; y = np.arange(len(groups))[::-1]
for gi, (lab, T, lo, hi) in enumerate(groups):
    E = [r["lvl_rank"] for r in D[T] if lo <= r["year"] <= hi]; left = 0
    for ci, (cl, l, h) in enumerate(cats):
        share = np.mean([(l <= x <= h) for x in E]) * 100
        a1.barh(y[gi], share, left=left, color=ramp[ci], edgecolor=BG, linewidth=1.5, height=0.62)
        if share >= 7: a1.text(left + share / 2, y[gi], f"{share:.0f}%", ha="center", va="center", fontsize=8.3, color="white" if ci < 3 else "#0b0b0b")
        left += share
    a1.text(101, y[gi], f"n={len(E)}", va="center", fontsize=8, color="#52514e")
a1.set_yticks(y); a1.set_yticklabels([g[0] for g in groups]); a1.set_xlim(0, 108); a1.set_xticks([0, 25, 50, 75, 100]); a1.set_xticklabels(["0", "25", "50", "75", "100%"])
a1.set_title("Where Slam champions ranked on the eve of the event\n(model rating on that surface, among the entrants)", loc="left", fontsize=10.5, color="#0b0b0b", pad=10)
a1.legend([Patch(color=c) for c in ramp], [c[0] for c in cats], ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.08), frameon=False, fontsize=8.3, handlelength=1.2, columnspacing=1.2)
a1.set_facecolor(BG)
for T, col, lab in (("atp", "#2a78d6", "Men"), ("wta", "#eb6834", "Women")):
    R = D[T]; yr = np.array([r["year"] + {"Australian Open": 0.05, "Roland Garros": 0.4, "Wimbledon": 0.52, "US Open": 0.7}.get(r["slam"], 0.5) for r in R]); tp = np.array([r["title_p"] * 100 for r in R])
    a2.scatter(yr, tp, s=16, color=col, alpha=0.55, linewidths=0, label=f"{lab}: each champion")
    o = np.argsort(yr); med = [np.median(tp[o][max(0, i - 6):i + 6]) for i in range(len(o))]
    a2.plot(yr[o], med, color=col, lw=2, label=f"{lab}: median of 12 Slams")
def find(T, nm, yr, sl): return next((r for r in D[T] if nm in r["champ"] and r["year"] == yr and r["slam"] == sl), None)
for T, nm, yr0, sl, txt, tx, ty in [("wta", "Raducanu", 2021, "US Open", "Raducanu 2021", 2005, 3), ("atp", "Edmondson", 1975, "Australian Open", "Edmondson 1975", 1966, 5),
                                    ("atp", "Borg", 1980, "Roland Garros", "Borg, Roland Garros 1980", 1984, 84), ("wta", "Seles", 1992, "Australian Open", "Seles, Australian Open 1992", 1994, 70)]:
    r = find(T, nm, yr0, sl)
    if r: a2.annotate(f"{txt} ({r['title_p']*100:.0f}%)" if r["title_p"] >= 0.01 else f"{txt} ({r['title_p']*100:.2f}%)", (yr0 + 0.4, max(r["title_p"] * 100, 0.6)), xytext=(tx, ty), fontsize=7.8, color="#52514e", arrowprops=dict(arrowstyle="-", color="#9a988f", lw=0.7))
a2.set_ylim(0, 92); a2.set_ylabel("champion's title chance before the event (%)")
a2.set_title("How likely the eventual champion was beforehand\n(model title chance from 4,000 seeded random draws)", loc="left", fontsize=10.5, color="#0b0b0b", pad=10)
a2.grid(axis="y", color="#ebe9e3", lw=0.8); a2.set_axisbelow(True); a2.set_facecolor(BG); a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8.3)
plt.tight_layout(); plt.savefig(f"{OUT}/slam_champions_pre_event.png", dpi=170, facecolor=BG); plt.close()
# ---------- chart 2: rating distribution ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 8.2), gridspec_kw=dict(height_ratios=[1.25, 1])); fig.patch.set_facecolor(BG)
cols = {"atp": "#2a78d6", "wta": "#eb6834"}; names = {"atp": f"Men, {len(D['atp'])} Slams 1968–2026", "wta": f"Women, {len(D['wta'])} Slams 1992–2026"}
notes = {"atp": [("Edmondson", 1975, "Australian Open", "Edmondson, AO 1975", 0.42), ("Djokovic", 2011, "Wimbledon", "Djokovic, Wimbledon 2011", 0.62), ("Borg", 1980, "Roland Garros", "Borg, RG 1980", 0.42)],
         "wta": [("Raducanu", 2021, "US Open", "Raducanu, US Open 2021", 0.42), ("Rybakina", 2026, "Australian Open", "Rybakina, AO 2026", 0.62), ("Graf", 1995, "US Open", "Graf, US Open 1995", 0.42)]}
for j, T in enumerate(("atp", "wta")):
    R = D[T]; ch = np.array([r["champ_lvl"] for r in R]); fld = np.concatenate([r["field_lvl"] for r in R]); c = cols[T]
    a = ax[0, j]; a.set_facecolor(BG); edges = np.arange(-0.8, 2.41, 0.1)
    a.hist(fld, bins=edges, density=True, color="#d8d6cf", edgecolor=BG, linewidth=1, label=f"all entrants ({len(fld):,} player-Slams)")
    a.hist(ch, bins=edges, density=True, color=c, alpha=0.85, edgecolor=BG, linewidth=1, label=f"champions ({len(ch)})")
    med = np.median(ch); a.axvline(med, color="#0b0b0b", lw=1, ls=(0, (3, 3))); ymax = a.get_ylim()[1]
    a.text(med + 0.03, ymax * 0.97, f"median champion {med:.2f}", fontsize=8, color="#0b0b0b", va="top")
    for nm, yr0, sl, txt, fy in notes[T]:
        r = find(T, nm, yr0, sl)
        if r: a.annotate(f"{txt}\n{r['champ_lvl']:.2f}", (r["champ_lvl"], 0.02), xytext=(r["champ_lvl"], ymax * fy), fontsize=7.6, color="#52514e", ha="center", arrowprops=dict(arrowstyle="-", color="#9a988f", lw=0.6))
    a.set_title(f"{names[T]}: rating on the eve of the Slam", loc="left", fontsize=10.5, color="#0b0b0b")
    a.set_xlabel("pre-event rating on that surface (model level, 0 = average tour player)"); a.set_ylabel("share of players (density)")
    a.legend(frameon=False, fontsize=8.3, loc="upper left"); a.set_xlim(-0.8, 2.4)
    b = ax[1, j]; b.set_facecolor(BG)
    e2 = np.array([-1, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 3])
    lab = ["< 0.4", "0.4–0.6", "0.6–0.8", "0.8–0.9", "0.9–1.0", "1.0–1.1", "1.1–1.2", "1.2–1.3", "1.3–1.4", "1.4–1.6", "≥ 1.6"]
    nf = np.histogram(fld, e2)[0]; nc = np.histogram(ch, e2)[0]; rate = nc / np.maximum(nf, 1) * 100
    x = np.arange(len(lab)); b.bar(x, rate, color=c, width=0.72); top = max(rate) * 1.18 + 2
    for xi, (rt, k1, k0) in enumerate(zip(rate, nc, nf)):
        b.text(xi, rt + top * 0.012, f"{rt:.0f}%" if rt >= 1 else f"{rt:.1f}%", ha="center", fontsize=7.8, color="#0b0b0b")
        b.text(xi, -top * 0.07, f"{k1}/{k0}", ha="center", fontsize=7, color="#52514e")
    b.set_xticks(x); b.set_xticklabels(lab, fontsize=8); b.set_ylim(-top * 0.1, top); b.set_yticks([v for v in b.get_yticks() if 0 <= v <= top])
    b.set_title("Chance that an entrant at this rating wins the Slam (titles / entrants)", loc="left", fontsize=10, color="#0b0b0b")
    b.set_ylabel("won the title (%)"); b.axhline(0, color="#c9c7c0", lw=0.8); b.spines["bottom"].set_visible(False); b.tick_params(axis="x", length=0)
plt.tight_layout(); plt.savefig(f"{OUT}/slam_champions_rating_distribution.png", dpi=170, facecolor=BG); plt.close()
# ---------- CSV ----------
with open(f"{OUT}/slam_champions_pre_event.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tour", "year", "slam", "surface", "champion", "entrants", "pre_event_rating", "pre_event_rating_rank", "pre_event_title_chance", "title_chance_rank",
                                    "favourite", "favourite_title_chance", "rating_gap_to_best_entrant", "opponents", "opponents_pre_event_rank", "chance_to_beat_this_path", "top10_opponents_beaten"])
    for T in ("atp", "wta"):
        for r in D[T]:
            w.writerow([T.upper(), r["year"], r["slam"], r["surf"], r["champ"], r["n"], round(r["champ_lvl"], 3), r["lvl_rank"], round(r["title_p"], 4), r["title_rank"], r["fav"], round(r["fav_p"], 4),
                        round(r["gap_to_best"], 3), " | ".join(r["opp"]), " ".join(map(str, r["opp_lvl_rank"])), round(r["path_p"], 4), r["n_top10"]])
print("charts + csv written")
