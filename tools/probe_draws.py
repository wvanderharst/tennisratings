"""One-off: where can we get the current draws? (1) the TennisMyLife download: unplayed rows / draw files?
(2) Tennis Abstract's current-event pages: what do they contain? A handful of requests, no crawling."""
import collections, csv, io, re, sys, urllib.request, zipfile
UA = {"User-Agent": "tennis-ratings-updater (personal, non-commercial)"}
def get(u, binary=False):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120) as r:
        b = r.read(); return b if binary else b.decode("utf-8", "replace")

print("==== TennisMyLife download")
zf = zipfile.ZipFile(io.BytesIO(get("https://stats.tennismylife.org/api/download-all", True)))
names = [i.filename for i in zf.infolist()]
print("files:", [n for n in names if not re.search(r"/?(19|20)\d\d", n)][:60])
for n in names:
    if "ongoing" in n.lower() or "draw" in n.lower() or "schedul" in n.lower():
        rows = list(csv.DictReader(io.StringIO(zf.read(n).decode("utf-8-sig", "replace"))))
        c = collections.Counter((r.get("tourney_name"), r.get("round"), bool((r.get("score") or "").strip())) for r in rows)
        print(f"\n-- {n}: {len(rows)} rows; (event, round, has score): n")
        for k, v in sorted(c.items(), key=str): print("  ", k, v)
        unplayed = [r for r in rows if not (r.get("score") or "").strip()]
        if unplayed: print("   example unplayed row:", dict(list(unplayed[0].items())[:32]))

print("\n==== Tennis Abstract home: links to current events")
home = get("https://www.tennisabstract.com/")
links = sorted(set(re.findall(r'href="([^"]*current/[^"]+)"', home)))
for l in links: print("  ", l)
for l in [l for l in links if "Forecast" not in l][:1] + [l for l in links if "Forecast" in l][:1] + [l for l in links if "WTA" in l][:1]:
    u = l if l.startswith("http") else "https://www.tennisabstract.com/" + l.lstrip("/")
    try: h = get(u)
    except Exception as e: print("  ", u, e); continue
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    print(f"\n-- {u}: {len(h)} chars")
    for kw in ("Forecast", "forecast", "QF", "SF", "R32", "W", "Draw"):
        print(f"  {kw!r}: {h.count(kw)}")
    i = h.find("orecast"); print("  forecast context:", re.sub(r"\s+", " ", h[max(0, i - 300): i + 2500]))
    vars_ = re.findall(r"var\s+(\w+)\s*=", h); print("  JS vars:", vars_[:40])
