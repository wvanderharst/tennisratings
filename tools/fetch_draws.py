"""Draws of the ongoing ATP / WTA tour events (incl. Grand Slams) from Tennis Abstract's current-event pages, for the
'Draws' tab: the site computes each player's chance to reach every round with the model's match odds.

    python3 tools/fetch_draws.py [OUT.json]        (default: site_v8/draws.json)

Per event: the draw in bracket order (seeds, WC/Q/LL, byes), the completed matches (winner, loser, score), and the
conditions (surface, indoor, level, best of) from the most recent edition in the match data. Knockout draws only:
round-robin events (Tour Finals, Next Gen) and team events (United Cup, Davis/BJK Cup) are skipped, as are Challengers and WTA 125s (most of their players are not in the odds data). A handful of requests per run; Tennis Abstract pages by
Jeff Sackmann (tennisabstract.com)."""
import csv, datetime, glob, html, json, os, re, sys, time, unicodedata, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "site_v8", "draws.json")
BASE = "https://www.tennisabstract.com/"
UA = {"User-Agent": "tennis-ratings site (personal, non-commercial; github.com/wvanderharst/tennisratings)"}
SLAMS = {"australian open": "Australian Open", "roland garros": "Roland Garros", "french open": "Roland Garros",
         "wimbledon": "Wimbledon", "us open": "US Open"}


def get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if getattr(e, "code", None) == 404 or k == 2: raise
            time.sleep(5 * (k + 1))


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def jsvar(page, name):
    m = re.search(r"var\s+%s\s*=\s*'(.*?)';" % name, page, re.S)
    return m.group(1).replace("\\'", "'") if m else None


PLAYER = re.compile(r'player\.cgi\?p=\d+/[^"&]*">([^<]+)</a>')


def parse_draw(page):
    """The largest projection table: one row per draw slot, in bracket order."""
    for n in (256, 128, 64, 32, 16, 8):
        t = jsvar(page, f"proj{n}")
        if t: break
    else:
        return None
    slots = []
    for row in re.findall(r"<tr>(.*?)</tr>", t, re.S):
        cell = re.search(r"<td>(.*?)</td>", row, re.S)
        if not cell: continue
        c = cell.group(1)
        if c.strip() == "Player": continue
        m = PLAYER.search(c)
        if m:
            pre = re.match(r"\s*\(([^)]+)\)", c)            # (1) seed, or (WC)/(Q)/(LL)/(PR)/(SE)
            slots.append([html.unescape(m.group(1)).strip(), pre.group(1) if pre else ""])
        elif re.sub(r"<[^>]+>|&nbsp;|\s", "", c).lower() == "bye":
            slots.append(["Bye", ""])
        elif re.sub(r"<[^>]+>|&nbsp;|\s", "", c):           # "Qualifier" or another placeholder
            slots.append([re.sub(r"<[^>]+>|&nbsp;", " ", c).strip(), "?"])
    if len(slots) != n:
        print(f"    draw has {len(slots)} slots, expected {n}: skipped", flush=True); return None
    return slots


def parse_results(page):
    """Completed singles: [round, winner, loser, score]."""
    out = []
    for line in (jsvar(page, "completedSingles") or "").split("<br/>"):
        m = re.match(r"\s*([A-Z0-9]+):", line); names = PLAYER.findall(line)
        if not m or len(names) < 2: continue
        tail = re.sub(r"<[^>]+>", "", line.rsplit("</a>", 1)[1])         # after the loser's name: " (CTY) 6-4 7-6(3)"
        sc = re.sub(r"^\s*\([A-Z]{3}\)\s*", "", html.unescape(tail)).strip()
        out.append([m.group(1), html.unescape(names[0]).strip(), html.unescape(names[1]).strip(), sc])
    return out


def conditions():
    """Most recent surface / indoor / level per tournament name, from the match data in the repository."""
    info = {}
    def scan(files, tour):
        for f in files:
            try:
                with open(f, encoding="utf-8-sig", errors="replace") as fh:
                    for r in csv.DictReader(fh):
                        nm, d = norm(r.get("tourney_name") or ""), (r.get("tourney_date") or "")[:8]
                        if not nm or not d: continue
                        k = (tour, nm)
                        if k not in info or d >= info[k]["d"]:
                            info[k] = dict(d=d, surf=(r.get("surface") or "Hard").strip() or "Hard",
                                           indoor=(r.get("indoor") or "").strip().upper() == "I", lvl=(r.get("tourney_level") or "").strip())
            except FileNotFoundError:
                pass
    scan(sorted(glob.glob(os.path.join(ROOT, "tennis", "tml-db", "*.csv")))[-6:] +
         [f for f in glob.glob(os.path.join(ROOT, "atp_uploads", "*.csv")) if "challenger" not in f and "ATP_Database" not in f], "atp")
    scan([os.path.join(ROOT, "wta_data", "wta_tml.csv")], "wta")
    return info


def main():
    home = get(BASE)
    yr = str(datetime.date.today().year)
    links = sorted(set(re.findall(r'href="(?:https?://www\.tennisabstract\.com/)?/?(current/[^"]+\.html)"', home)))
    info = conditions()
    events = []
    for l in links:
        name = l.split("/")[-1][:-5]
        if not name.startswith(yr) or "Challenger" in name or re.search(r"125$", name) or "Doubles" in name: continue
        try:
            page = get(BASE + l)
        except Exception as e:
            print(f"  {l}: {e}", flush=True); continue
        title = re.search(r"<title>Tennis Abstract:\s*(.*?)\s*Results and Forecast", page)
        title = title.group(1) if title else name
        draw = parse_draw(page)
        if re.search(r"\bWTA\b|Women", title): tour = "wta"
        elif re.search(r"\bATP\b|Men", title): tour = "atp"
        else:     # the draw's own player links (the site menu links to both tours)
            t = jsvar(page, "projCurrent") or ""
            tour = "wta" if t.count("wplayer.cgi") > t.count("/player.cgi") else "atp"
        if not draw: continue
        res = parse_results(page)
        up = jsvar(page, "upcomingSingles") or ""
        if any(r[0] == "RR" for r in res) or re.search(r"(^|>)\s*RR:", up) or re.search(r"Finals|Cup\b|Next Gen", title):
            print(f"  {title}: round robin / team event, skipped", flush=True); continue
        if any(r[0] == "F" for r in res):
            print(f"  {title}: final played, skipped", flush=True); continue
        ev = re.sub(r"^\d{4}\s+(ATP|WTA)?\s*", "", title).strip()
        ev = re.sub(r"\s+(Men|Women)$", "", ev).strip()
        slam = next((v for k, v in SLAMS.items() if k in norm(ev)), None)
        c = info.get((tour, norm(slam or ev)))
        if not c:     # loose match: one name contained in the other (e.g. 'Shanghai' vs 'Shanghai Masters')
            cand = [(k, v) for k, v in info.items() if k[0] == tour and (norm(ev) in k[1] or k[1] in norm(ev))]
            c = max(cand, key=lambda kv: kv[1]["d"])[1] if cand else None
        lvl = "G" if slam else ("M" if c and c["lvl"] == "M" and tour == "atp" else "A")
        events.append(dict(tour=tour, name=slam or ev, title=title, url=BASE + l,
                           surf=c["surf"] if c else "Hard", indoor=bool(c and c["indoor"]), lvl=lvl,
                           bo=5 if slam and tour == "atp" else 3, draw=draw, results=res))
        print(f"  {tour} {title}: {len(draw)} slots, {len(res)} results, {events[-1]['surf']}{' indoor' if events[-1]['indoor'] else ''}, "
              f"level {lvl}, conditions from {'data ' + c['d'] if c else 'default'}", flush=True)
        time.sleep(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(asof=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), source="Tennis Abstract (tennisabstract.com)",
                   events=events), open(OUT, "w"), separators=(",", ":"))
    print(f"{len(events)} events -> {OUT}")


if __name__ == "__main__":
    main()
