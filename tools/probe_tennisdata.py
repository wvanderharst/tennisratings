"""One-off look at one results.tennisdata.com match page: is the match data inside the plain HTML (Next.js payload)?
Fetches that single page once; no browser, no crawling."""
import json, re, sys, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "https://results.tennisdata.com/en/results/ecef0cbb-a99e-49ad-88a3-e529b18065ee"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": UA}), timeout=30) as r:
    print("status", r.status); html = r.read().decode("utf-8", "replace")
print("html chars", len(html), "| blocked page:", "you have been blocked" in html)
# Next.js app router streams its data as self.__next_f.push([1,"..."]) chunks
chunks = re.findall(r'self\.__next_f\.push\(\[\d+,\s*"((?:[^"\\]|\\.)*)"\]\)', html)
payload = "".join(json.loads('"' + c + '"') for c in chunks)
print("next_f chunks", len(chunks), "| payload chars", len(payload))
txt = payload or html
for kw in ("shotQuality", "shot_quality", "Shot Quality", "serveSpeed", "serve_speed", "forehand", "Forehand", "backhand", "Backhand",
           "rally", "pointByPoint", "point_by_point", "points", "aces", "Aces", "firstServe", "winners", "unforced", "Hawk", "spin", "depth"):
    n = txt.count(kw)
    if n: print(f"  {kw!r}: {n}")
for kw in ("orehand", "uality", "point"):
    print(f"\n==== contexts for {kw!r}")
    for m in list(re.finditer(kw, txt))[:6]:
        print("  ..." + txt[max(0, m.start() - 250): m.end() + 250].replace("\n", " ") + "...")
# largest JSON-looking objects in the payload
objs = sorted(re.findall(r'\{"[^{}]{200,}\}', txt), key=len, reverse=True)[:3]
print("\n==== largest flat objects")
for o in objs: print("  ", o[:1500], "\n")
