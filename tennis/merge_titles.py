"""Add actual vs predicted Slams/titles (from expect_titles_haz.py) to the profile index.   python3 merge_titles.py atp|wta
Index row becomes [name, bucket, rated matches, last date, level, slam entries, slams won, slams predicted,
                   events, titles, titles predicted]."""
import paths as P
import json, sys
OUT = P.OUT; SCR = P.ROOT
T = sys.argv[1]
te = json.load(open(f"{OUT}/titles_expected_{T}.json"))
fn = f"{SCR}/site_v8/prof_{T}_index.json"; idx = json.load(open(fn))
for row in idx["players"]:
    d = te.get(row[0], dict(se=0, st=0, sx=0.0, e=0, t=0, x=0.0))
    row[5:] = [d["se"], d["st"], round(d["sx"], 2), d["e"], d["t"], round(d["x"], 2)]
json.dump(idx, open(fn, "w"), separators=(",", ":"))
print(T, "index updated:", len(idx["players"]), "players")
