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
# keys that carry numbers (the actual match values, not the translated labels)
from collections import Counter
num = Counter(m.group(1) for m in re.finditer(r'"([A-Za-z_]+)":\s*-?\d+(?:\.\d+)?[,}\]]', txt))
print("==== numeric keys (key: count)\n  " + ", ".join(f"{k}:{n}" for k, n in num.most_common(80)))
for kw in ("shotQualityForehand", "forehand", "rallyLength", "speed", "Sinner", "Alcaraz", "playerName", "lastName"):
    hits = [m for m in re.finditer(re.escape(kw), txt) if re.match(r'"\s*:\s*[-\d\[{]', txt[m.end():m.end() + 4])]
    print(f"\n==== value contexts for {kw!r}: {len(hits)}")
    for m in hits[:3]: print("  ..." + txt[max(0, m.start() - 300): m.end() + 400] + "...")
# where are the player names? (first person-like "firstName"/"name" values)
for m in list(re.finditer(r'"(?:firstName|lastName|fullName|name)":"([^"]{3,40})"', txt))[:15]: print("  name:", m.group(1))
