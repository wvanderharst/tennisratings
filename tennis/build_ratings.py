"""
build_ratings.py — turn Jeff Sackmann's Match Charting Project into a time-tracked
player rating dataset for the tennis simulators.

    python3 build_ratings.py [--outdir .] [--no-download]

Outputs
    player-history.csv       one row per player per charted match: raw observed rates,
                             opponent-adjusted targets, and the running EWMA after that match
    players-atp.csv          snapshot of the field on the last charted date, importable
                             straight into tournament-simulator.html
    series.json              the same ratings, compacted, for embedding in the HTML

Source data: https://github.com/JeffSackmann/tennis_MatchChartingProject
Licensed CC BY-NC-SA 4.0 — attribution required, non-commercial use only.
"""

import argparse
import csv
import json
import os
import glob
import urllib.request
from collections import defaultdict
from datetime import date

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

LAM = 0.07          # weight on the newest match
KEEP = 0.93         # weight on the running value
INIT_N = 3          # a rating is born as the average of the first three charted matches
ACTIVE_DAYS = 365   # used for the snapshot only

KEYS = ["f1", "w1", "f2", "w2", "r1", "r2"]

# The two sides of the same coin. The simulator computes a serve point as
#     p = (server's own rate + (1 - returner's matching rate)) / 2
# so w1 is answered by r1, w2 by r2, and vice versa.
OPP = {"w1": "r1", "w2": "r2", "r1": "w1", "r2": "w2"}

# Serve-in percentages have no opponent counterpart — the returner cannot make you
# miss a first serve — so these take the raw match figure with no adjustment.
SOLO = ("f1", "f2")

# Guard rails, so one freak match can't push a rating somewhere no player has ever been.
BOUND = {"f1": (.45, .80), "w1": (.50, .88), "f2": (.75, .99),
         "w2": (.30, .75), "r1": (.10, .48), "r2": (.25, .65)}

ELO_START = 1500.0


def elo_k(n):
    """Move new players quickly and settled ones slowly — the usual tennis variant."""
    return 250.0 / (n + 5) ** 0.4

MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"
MCP_FILES = {"mcp-matches.csv": "charting-m-matches.csv",
             "mcp-overview.csv": "charting-m-stats-Overview.csv"}

# Tennismylife's TML-Database: the tennis_atp schema, live-updated, one file per year.
# Tour-level main draw only, but roughly thirteen times as many matches as the charting
# project, which matters most for how often an opponent is already rated.
TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/"
TML_YEARS = range(1991, 2027)


def clip(key, x):
    lo, hi = BOUND[key]
    return min(hi, max(lo, x))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def download(outdir, source):
    if source == "mcp":
        jobs = [(os.path.join(outdir, local), MCP_BASE + remote)
                for local, remote in MCP_FILES.items()]
    else:
        os.makedirs(os.path.join(outdir, "tml"), exist_ok=True)
        jobs = [(os.path.join(outdir, "tml", f"{y}.csv"), TML_BASE + f"{y}.csv")
                for y in TML_YEARS]
    for path, url in jobs:
        if os.path.exists(path):
            continue
        print("fetching", os.path.basename(path))
        urllib.request.urlretrieve(url, path)


def parse_date(s):
    s = (s or "").strip()
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None


def to_int(row, key):
    try:
        return int(row[key])
    except (ValueError, TypeError, KeyError):
        return None


def rates(server, returner):
    """The six rates for one player in one match, from their own row and their opponent's.

    Note that MCP's `second_in` column counts second-serve *points*, not second serves
    landed, so double faults have to be subtracted to get a true in-percentage.
    """
    v = {k: to_int(server, k) for k in ("serve_pts", "first_in", "first_won", "second_won", "dfs")}
    w = {k: to_int(returner, k) for k in ("serve_pts", "first_in", "first_won", "second_won", "dfs")}
    if any(x is None for x in list(v.values()) + list(w.values())):
        return None

    s2 = v["serve_pts"] - v["first_in"]      # my second-serve points
    s2in = s2 - v["dfs"]                     # ... of which the serve actually landed
    o2 = w["serve_pts"] - w["first_in"]      # their second-serve points
    o2in = o2 - w["dfs"]
    if min(v["first_in"], s2, s2in, w["first_in"], o2, o2in) <= 0:
        return None

    return {
        "f1": v["first_in"] / v["serve_pts"],
        "w1": v["first_won"] / v["first_in"],
        "f2": s2in / s2,
        "w2": v["second_won"] / s2in,
        "r1": (w["first_in"] - w["first_won"]) / w["first_in"],
        "r2": (o2in - w["second_won"]) / o2in,
    }


def side_from_tml(row, me, them):
    """The six rates for one side of a TML row. `me`/`them` are the "w" / "l" prefixes."""
    def n(prefix, field):
        try:
            return int(row[prefix + "_" + field])
        except (ValueError, TypeError, KeyError):
            return None

    svpt, f_in, f_won, s_won, dfs = (n(me, x) for x in
                                     ("svpt", "1stIn", "1stWon", "2ndWon", "df"))
    o_svpt, o_in, o_won, o_swon, o_df = (n(them, x) for x in
                                         ("svpt", "1stIn", "1stWon", "2ndWon", "df"))
    if any(x is None for x in (svpt, f_in, f_won, s_won, dfs,
                               o_svpt, o_in, o_won, o_swon, o_df)):
        return None

    s2, o2 = svpt - f_in, o_svpt - o_in          # second-serve points
    s2in, o2in = s2 - dfs, o2 - o_df             # ... of which the serve landed
    if min(f_in, s2, s2in, o_in, o2, o2in) <= 0:
        return None
    if f_won > f_in or s_won > s2in or o_won > o_in or o_swon > o2in:
        return None                               # occasional bad row in the source

    return {"f1": f_in / svpt, "w1": f_won / f_in, "f2": s2in / s2, "w2": s_won / s2in,
            "r1": (o_in - o_won) / o_in, "r2": (o2in - o_swon) / o2in}


def load_matches_tml(outdir):
    """Every tour-level match with serve stats, in date order, plus each player's country."""
    out, country = [], {}
    for path in sorted(glob.glob(os.path.join(outdir, "tml", "*.csv"))):
        for row in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            when = parse_date(row.get("tourney_date"))
            if not when:
                continue
            a, b = row.get("winner_name", "").strip(), row.get("loser_name", "").strip()
            if not a or not b or a == b:
                continue
            ra, rb = side_from_tml(row, "w", "l"), side_from_tml(row, "l", "w")
            if not ra or not rb:
                continue

            surface = row.get("surface", "")
            if surface not in ("Hard", "Clay", "Grass", "Carpet"):
                surface = ""
            for name, key in ((a, "winner_ioc"), (b, "loser_ioc")):
                ioc = (row.get(key) or "").strip()
                if ioc:
                    country[name] = ioc

            match_id = f"{row.get('tourney_id','')}-{row.get('match_num','')}"
            out.append((when, match_id, a, ra, b, rb, surface, row.get("tourney_name", "")))

    # tourney_date is the start of the event, so order within it by the id we built
    out.sort(key=lambda x: (x[0], x[1]))
    return out, country


def load_matches_mcp(outdir):
    """Every usable charted match, in date order."""
    meta = {x["match_id"]: x for x in csv.DictReader(open(os.path.join(outdir, "mcp-matches.csv")))}

    overview = defaultdict(dict)
    for row in csv.DictReader(open(os.path.join(outdir, "mcp-overview.csv"))):
        if row["set"] == "Total":                      # skip the per-set breakdowns
            overview[row["match_id"]][row["player"]] = row

    out = []
    for match_id, players in overview.items():
        m = meta.get(match_id)
        if not m or len(players) != 2:
            continue
        when = parse_date(m.get("Date"))
        if not when:
            continue

        a, b = list(players)
        ra, rb = rates(players[a], players[b]), rates(players[b], players[a])
        if not ra or not rb:
            continue

        surface = m.get("Surface", "")
        if surface not in ("Hard", "Clay", "Grass", "Carpet"):
            surface = ""
        out.append((when, match_id, a, ra, b, rb, surface, m.get("Tournament", "")))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


# ---------------------------------------------------------------------------
# Rate the field
# ---------------------------------------------------------------------------

def estimate_novice(matches, pool):
    """What a not-yet-rated player is actually worth.

    Falling back on the tour average for anyone without a record flatters whoever plays
    them: beat up a qualifier's second serve and the model reads it as beating up an
    average tour second serve. So measure the gap directly. Take players who do have a
    record and compare, within each player, what they did against unrated opponents with
    what they did against rated ones. Their own level cancels out of that difference.

    An observed serve rate is (my w1 + (1 - their r1)) / 2, so a lift of d against unrated
    opponents means their return rate sits 2d below the rated average. Nothing in this
    estimate depends on the ratings it feeds, so unlike inverting each match it cannot
    spiral: every quantity is a raw observed rate.
    """
    played = defaultdict(int)
    seen = defaultdict(lambda: defaultdict(list))   # rated player -> key -> (opp_rated, obs)
    unrated_solo = defaultdict(list)

    for _, _, a, ra, b, rb, _, _ in matches:
        rated = {a: played[a] >= INIT_N, b: played[b] >= INIT_N}
        for me, obs, opp in ((a, ra, b), (b, rb, a)):
            if rated[me]:
                for k in KEYS:
                    seen[me][k].append((rated[opp], obs[k]))
            else:
                for k in SOLO:
                    unrated_solo[k].append(obs[k])
            played[me] += 1

    novice = {k: sum(unrated_solo[k]) / len(unrated_solo[k]) for k in SOLO}
    for k in KEYS:
        if k in SOLO:
            continue
        diffs, weights = [], []
        for d in seen.values():
            soft = [x for r, x in d[OPP[k]] if not r]
            hard = [x for r, x in d[OPP[k]] if r]
            if len(soft) < 3 or len(hard) < 3:
                continue
            diffs.append(sum(soft) / len(soft) - sum(hard) / len(hard))
            weights.append(min(len(soft), len(hard)))
        lift = sum(x * w for x, w in zip(diffs, weights)) / sum(weights)
        novice[k] = pool[k] - 2 * lift
    return novice


def build_history(matches):
    """Replay every match in order, updating both players after each one."""
    pool = {k: sum(r[k] for _, _, _, ra, _, rb, _, _ in matches for r in (ra, rb)) / (2 * len(matches))
            for k in KEYS}
    novice = estimate_novice(matches, pool)
    print("novice prior:", {k: round(100 * v, 1) for k, v in novice.items()})

    rating, seed_buf, played, history = {}, defaultdict(list), defaultdict(int), []
    elo = defaultdict(lambda: ELO_START)

    def current(player, key):
        # anyone without a settled rating is treated as a novice, not as an average pro
        return rating[player][key] if player in rating else novice[key]

    for when, match_id, a, ra, b, rb, surface, tourn in matches:
        # Elo runs alongside as a plain, well-understood benchmark. The winner is
        # always the first player in these files, so this update needs no result flag.
        expected = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
        ea, eb = elo[a], elo[b]
        elo[a] = ea + elo_k(played[a]) * (1 - expected)
        elo[b] = eb - elo_k(played[b]) * (1 - expected)

        # Both updates read the ratings as they stood *before* this match, so neither
        # player's new number contaminates the other's adjustment.
        pre = {a: {k: current(a, k) for k in KEYS},
               b: {k: current(b, k) for k in KEYS}}

        for me, obs, opp in ((a, ra, b), (b, rb, a)):
            played[me] += 1
            n = played[me]
            mine, theirs = pre[me], pre[opp]

            target = {}
            for k in KEYS:
                if k in SOLO:
                    target[k] = clip(k, obs[k])
                    continue

                # What the simulator would have predicted for this pairing...
                pred = (mine[k] + (1 - theirs[OPP[k]])) / 2
                err = obs[k] - pred

                # ...and how the surprise is shared out. A settled player takes half,
                # leaving the other half for the opponent, whose own row this match
                # moves by the same amount in the opposite direction. That antisymmetry
                # matters: p is unchanged if you add a constant to both w1 and r1, so a
                # rule that pushed both the same way would let elite-vs-elite matches
                # inflate the whole scale with nothing to pin it down.
                # A player inside their first three matches has no history to defend
                # and takes the whole error.
                share = 2 * err if n <= INIT_N else err
                target[k] = clip(k, mine[k] + share)

            if n <= INIT_N:
                seed_buf[me].append(target)
                if n == INIT_N:
                    rating[me] = {k: sum(x[k] for x in seed_buf[me]) / INIT_N for k in KEYS}
                settled = rating.get(me)
            else:
                prev = rating[me]
                rating[me] = {k: clip(k, KEEP * prev[k] + LAM * target[k]) for k in KEYS}
                settled = rating[me]

            history.append({
                "player": me, "date": when.isoformat(), "match_no": n, "match_id": match_id,
                "tournament": tourn, "surface": surface, "opponent": opp,
                "settled": 1 if settled else 0, "elo": round(elo[me], 1),
                **{"obs_" + k: round(100 * obs[k], 2) for k in KEYS},
                **{"adj_" + k: round(100 * target[k], 2) for k in KEYS},
                **{"ewma_" + k: (round(100 * settled[k], 2) if settled else "") for k in KEYS},
            })

    return history, pool, novice, rating


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_history(history, path):
    cols = (["player", "date", "match_no", "match_id", "tournament", "surface", "opponent", "settled", "elo"]
            + ["obs_" + k for k in KEYS] + ["adj_" + k for k in KEYS] + ["ewma_" + k for k in KEYS])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(history)


def build_series(history, pool, novice, country=None, epoch=date(1990, 1, 1)):
    """Compact form for the browser: [dayIndex, matchNo, f1, w1, f2, w2, r1, r2],
    every rate in tenths of a percent."""
    series = defaultdict(list)
    for h in history:
        if not h["settled"]:
            continue
        y, m, d = map(int, h["date"].split("-"))
        series[h["player"]].append(
            [(date(y, m, d) - epoch).days, h["match_no"]]
            + [int(round(10 * h["ewma_" + k])) for k in KEYS] + [int(round(h["elo"]))])
    country = country or {}
    return {"epoch": epoch.isoformat(), "keys": KEYS,
            "pool": {k: round(100 * v, 2) for k, v in pool.items()},
            "novice": {k: round(100 * v, 2) for k, v in novice.items()},
            "players": [[p, country.get(p, ""), rows] for p, rows in sorted(series.items())]}


def thin(series):
    """Keep only each player's last rating in any calendar month. The app reads a rating
    as of a chosen day, and month resolution is plenty for that — it cuts the file by
    roughly an order of magnitude on a full tour-level dataset."""
    epoch = date.fromisoformat(series["epoch"])
    for entry in series["players"]:
        rows, kept = entry[2], []
        for row in rows:
            d = date.fromordinal(epoch.toordinal() + row[0])
            if kept and (d.year, d.month) == kept[-1][0]:
                kept[-1] = ((d.year, d.month), row)
            else:
                kept.append(((d.year, d.month), row))
        entry[2] = [r for _, r in kept]


def pack(series):
    """Flatten each player's rating points into one delta-encoded array.

    A point is [dayIndex, matchNo, f1, w1, f2, w2, r1, r2]; after the first, every field
    is stored as its change from the point before. Ratings move slowly, so most of those
    deltas are single digits, which cuts the file to about half the size of the nested
    form with no loss at all — it is exact, not lossy.
    """
    series["packed"] = True
    series["width"] = 9
    for entry in series["players"]:
        rows, flat, prev = entry[2], [], None
        for row in rows:
            flat.extend(row if prev is None else [row[i] - prev[i] for i in range(len(row))])
            prev = row
        entry[2] = flat


def strength(vals):
    """Hold expectation plus average return rate — used only for seeding."""
    f1, w1, f2, w2, r1, r2 = (vals[k] / 100 for k in KEYS)
    return f1 * w1 + (1 - f1) * f2 * w2 + (r1 + r2) / 2


def unpack(flat, width=9):
    """Inverse of pack(): rebuild the list of rating points."""
    rows, prev = [], None
    for i in range(0, len(flat), width):
        chunk = flat[i:i + width]
        row = chunk if prev is None else [prev[j] + chunk[j] for j in range(width)]
        rows.append(row)
        prev = row
    return rows


def snapshot(series, on, min_matches=10, window=ACTIVE_DAYS):
    """The field as it stood on a given date, newest rating per player."""
    epoch = date.fromisoformat(series["epoch"])
    limit = (on - epoch).days
    floor = limit - window

    field = []
    for name, ioc, rows in series["players"]:
        if series.get("packed"):
            rows = unpack(rows)
        last = None
        for row in rows:
            if row[0] <= limit:
                last = row
            else:
                break
        if not last:                       # no settled rating yet on that date
            continue
        if last[0] < floor:                # a year without a charted match — out
            continue
        if last[1] < min_matches:          # too thin a record to trust
            continue
        vals = {k: last[2 + i] / 10 for i, k in enumerate(KEYS)}
        vals["elo"] = last[8]
        field.append({"name": name, "country": ioc, "matches": last[1],
                      "last": (epoch.toordinal() + last[0]),
                      **vals})

    field.sort(key=strength, reverse=True)
    return field


def write_snapshot(field, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "country", "first_serve_in_pct", "first_serve_won_pct",
                    "second_serve_in_pct", "second_serve_won_pct",
                    "return_won_vs_first_pct", "return_won_vs_second_pct",
                    "elo", "matches_played", "last_played"])
        for i, p in enumerate(field):
            w.writerow([i + 1, p["name"], p["country"]] + [p[k] for k in KEYS]
                       + [p["elo"], p["matches"], date.fromordinal(p["last"]).isoformat()])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("tml", "mcp"), default="tml",
                    help="tml: tour-level ATP matches 1991- (default). mcp: Sackmann's charting project.")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--monthly", action="store_true",
                    help="keep one rating per player per month in series.json instead of one per match")
    args = ap.parse_args()
    out = lambda name: os.path.join(args.outdir, name)

    os.makedirs(args.outdir, exist_ok=True)
    if not args.no_download:
        download(args.outdir, args.source)

    if args.source == "tml":
        matches, country = load_matches_tml(args.outdir)
    else:
        matches, country = load_matches_mcp(args.outdir), {}
    print(f"usable matches: {len(matches)}  ({matches[0][0]} to {matches[-1][0]})")

    history, pool, novice, rating = build_history(matches)
    print("pool prior: ", {k: round(100 * v, 1) for k, v in pool.items()})
    print(f"players seen: {len(set(h['player'] for h in history))}  "
          f"with a settled rating: {len(rating)}  rows: {len(history)}")

    write_history(history, out("player-history.csv"))

    series = build_series(history, pool, novice, country)
    if args.monthly:
        thin(series)
    pack(series)
    with open(out("series.json"), "w") as f:
        json.dump(series, f, separators=(",", ":"))
    print(f"series.json: {os.path.getsize(out('series.json'))/1e6:.2f} MB, "
          f"{sum(len(r) for _, _, r in series['players']) // 9} rating points")

    latest = max(h["date"] for h in history)
    field = snapshot(series, date.fromisoformat(latest), args.min_matches)
    write_snapshot(field, out("players-atp.csv"))
    print(f"snapshot on {latest}: {len(field)} players, top seed {field[0]['name']}")


if __name__ == "__main__":
    main()
