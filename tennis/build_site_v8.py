"""Assemble site_v8: template + app_tabs, traj (WTA from 1992) with model constants, odds."""
import paths as P
import json
O = P.OUT; S = P.ROOT; D = f"{S}/site_v8"
for t in ("atp", "wta"):
    od = json.load(open(f"{O}/odds_data_{t}_sr.json")); tr = json.load(open(f"{O}/traj_data_{t}_sr.json"))
    tr["cal"].update(hm=od["const"]["hm"], ptm=od["const"]["pt_m"], mu=od["const"]["mu"], bcl=od["const"]["beta"]["deciding_set_clutch"])
    if t == "wta":
        for p in tr["players"]:
            p["pts"] = [x for x in p["pts"] if x[0] >= "1992-01-01"]; p["ptsFine"] = [x for x in p["ptsFine"] if x[0] >= "1992-01-01"]
        tr["players"] = [p for p in tr["players"] if p["pts"]]
        tr["slam_markers"] = [m for m in tr["slam_markers"] if m["date"] >= "1992-01-01"]
        tr["thr_month"] = {k: v for k, v in tr["thr_month"].items() if k >= "1992-01"}; tr["thr_week"] = {k: v for k, v in tr["thr_week"].items() if k >= "1992-W01"}
    for p in tr["players"]:     # 3 decimals is plenty for the chart and keeps the file under the size limit
        for k in ("pts", "ptsFine"): p[k] = [[round(v, 3) if isinstance(v, float) else v for v in x] for x in p[k]]
    json.dump(tr, open(f"{D}/traj_{t}.json", "w"), separators=(",", ":")); json.dump(od, open(f"{D}/odds_{t}.json", "w"), separators=(",", ":"))
h = open(f"{S}/tennis/rating_history_artifact_v8.html").read(); j = open(f"{S}/tennis/app_tabs_v8.js").read()
open(f"{D}/tennis_ratings.html", "w").write('<meta charset="utf-8">\n' + h + "\n<script>\n" + j + "\n</script>\n")
print("site built")
