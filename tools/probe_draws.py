"""One-off: the format of a Tennis Abstract current-event page (draw order, completed/upcoming matches). Two requests."""
import re, urllib.request
UA = {"User-Agent": "tennis-ratings-updater (personal, non-commercial)"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read().decode("utf-8", "replace")
for u in ("https://www.tennisabstract.com/current/2026ATPShanghai.html", "https://www.tennisabstract.com/current/2026WTAWuhan.html"):
    h = get(u); print(f"\n######## {u}")
    m = re.search(r"<title>(.*?)</title>", h); print("title:", m and m.group(1))
    for v in ("proj128", "proj64", "proj32", "projCurrent", "prob8", "upcomingSingles", "completedSingles"):
        m = re.search(r"var\s+%s\s*=\s*'(.*?)';\s*\n" % v, h, re.S) or re.search(r"var\s+%s\s*=\s*'(.*?)';" % v, h, re.S)
        if not m: print(f"== {v}: -"); continue
        s = m.group(1)
        rows = re.findall(r"<tr>(.*?)</tr>", s, re.S)
        clean = lambda x: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x.replace("&nbsp;", " "))).strip()
        print(f"== {v}: {len(s)} chars, {len(rows)} rows")
        if v.startswith("proj") and v != "projCurrent":
            print("   draw order:", [clean(r).split("  ")[0] for r in rows][:40])
        else:
            for r in rows[:12]: print("   ", clean(r))
            if not rows: print("   raw:", s[:2500])
    i = h.find("completedSingles"); print("\nraw completed:", h[i:i + 2500])
