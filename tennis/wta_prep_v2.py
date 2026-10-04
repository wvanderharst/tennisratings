"""WTA data v2: TennisMyLife main-tour files 1990-2026 (with serve stats, ~2003+), plus the ongoing file.

  A = TML main tour only                       -> wta_data/wta_tml.csv
  B = A + the earlier upload's ITF and qualifying matches (to Sept 2017) -> wta_data/wta_tml_itf.csv

Level coding for the harness (same idea as wta_prep.py): 1000-level events (Tier I, Premier Mandatory,
Premier 5, WTA 1000; NOT 2009-20 "I" = International) -> 'M'; season finals -> 'F'; Slams 'G'; BJK/Fed Cup 'D'; ITF 'C'; qualifying 'Q'.
Birthdates by name: the earlier players.csv where it knows the player, else derived from the age column.
"""
import paths as P
import csv, datetime, glob, json, os, collections
Z = os.environ.get("TML_DIR", P.ROOT + "/tml_zip")
D = P.ROOT + "/wta_data"
COLS = ["tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "indoor", "tourney_date", "match_num",
        "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
        "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
        "score", "best_of", "round", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"]
BIG = {"T1", "PM", "P5", "1000"}   # NB: "I" (2009-2020) = International, the LOWEST tier -- not Tier I
def level(r):
    lv = (r.get("tourney_level") or "").strip(); nm = r.get("tourney_name") or ""
    if (r.get("round") or "").strip() in ("Q1", "Q2", "Q3", "Q4"): return "Q"
    if lv in BIG: return "M"
    if lv in ("F", "YEC") or (lv == "W" and ("Championships" in nm or "Finals" in nm)): return "F"
    return lv or "A"
tml, seen = [], set()
files = sorted(glob.glob(f"{Z}/[12][0-9][0-9][0-9]_wta.csv")) + [f"{Z}/wta_ongoing_tourneys.csv"]
for f in files:
    for r in csv.DictReader(open(f, encoding="utf-8-sig", errors="replace")):
        d = (r.get("tourney_date") or "").strip()
        if not (len(d) >= 8 and d[:8].isdigit()): continue
        k = (d[:8], r["winner_name"].strip(), r["loser_name"].strip(), (r.get("score") or "").strip())
        if k in seen: continue
        seen.add(k)
        rr = {c: (r.get(c) or "") for c in COLS}; rr["tourney_level"] = level(r); rr["tourney_date"] = d[:8]
        tml.append(rr)
# ---- fix scores written from the loser's side (78 rows, mostly 2026: Rome, Berlin, ...). In every case the listed
# winner played the next round, so the names are right and only the score is reversed: flip each set, and drop the
# serve stats of those rows (their orientation can't be verified), so the update uses the corrected games.
import re
def _sets(sc):
    w = l = 0
    for t in sc.split():
        m = re.match(r"^(\d+)-(\d+)", t)
        if not m: return None
        a_, b_ = int(m.group(1)), int(m.group(2)); w += a_ > b_; l += b_ > a_
    return w, l
def _flip(sc):
    return " ".join(re.sub(r"^(\d+)-(\d+)", lambda m: f"{m.group(2)}-{m.group(1)}", t) for t in sc.split())
nfix = 0
for rr in tml:
    sc = rr["score"].strip()
    if not sc or any(x in sc.upper() for x in ("RET", "W/O", "DEF", "ABD")): continue
    st = _sets(sc)
    if st and st[0] < st[1]:
        rr["score"] = _flip(sc); nfix += 1
        for c in COLS:
            if c.startswith(("w_", "l_")): rr[c] = ""
print(f"reversed scores fixed: {nfix}")
old = [r for r in csv.DictReader(open(f"{D}/wta_all.csv")) if r["tourney_level"] in ("C", "Q")] if os.path.exists(f"{D}/wta_all.csv") else []
for nm, rows in (("wta_tml.csv", tml), ("wta_tml_itf.csv", tml + old))[:2 if old else 1]:     # the ITF variant is optional (not used by the model)
    with open(f"{D}/{nm}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"{nm}: {len(rows)} rows; levels {collections.Counter(r['tourney_level'] for r in rows).most_common()}")
births = json.load(open(f"{D}/wta_birth_ordinal.json" if os.path.exists(f"{D}/wta_birth_ordinal.json") else f"{D}/wta_birth_ordinal_v2.json"))
added = 0
for rr in tml:
    for s in ("winner", "loser"):
        nm = rr[s + "_name"].strip()
        if nm and nm not in births and rr[s + "_age"]:
            try:
                dd = rr["tourney_date"]; t = datetime.date(int(dd[:4]), int(dd[4:6]), int(dd[6:8])).toordinal()
                births[nm] = int(t - float(rr[s + "_age"]) * 365.25); added += 1
            except ValueError: pass
json.dump(births, open(f"{D}/wta_birth_ordinal_v2.json", "w"))
names_tml = {r[s + "_name"].strip() for r in tml for s in ("winner", "loser")}
names_old = {r[s + "_name"].strip() for r in old for s in ("winner", "loser")}
print(f"birthdates: {len(births)} ({added} new from TML ages); TML players {len(names_tml)}, of whom {len(names_tml & names_old)} also appear in the old ITF/qualifying rows")
print("last TML match date:", max(r["tourney_date"] for r in tml), "| with serve stats:", sum(1 for r in tml if r["w_svpt"]) , "of", len(tml))
