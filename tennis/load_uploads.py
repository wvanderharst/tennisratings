"""Direct loader for the uploaded 2024-2026 tour + challenger CSVs (the fuller
1991-2023 TML-Database history is not present in this sandbox any more, so this
analysis runs on 2024-01 through 2026-08 only). Reuses the parsing helpers in
score_driven.py so counts/scores are extracted identically to the main pipeline.
"""
import paths as P
import csv
import glob
import os
import score_driven as S

UPLOAD_DIR = P.ROOT + "/atp_uploads"   # 2024-2026 tour, challenger and qualifying files (2026 up to 2026-09-29)
HIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tml-db")


def load_counts_uploads(levels=("tour", "chall", "qual"), history=True):
    """2024-2026 tour+challenger uploads, optionally extended back to 1968
    with the ATP-tour-only GitHub mirror of TML-Database (no challenger
    coverage there, but it gives every player years of properly warmed-up
    rating history instead of a cold start in 2024)."""
    out, seen = [], set()
    paths = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.csv")))
    if history and os.path.isdir(HIST_DIR):
        # keep only the pre-2024 years from the mirror; 2024-2026 come from
        # the fuller uploaded files (tour+challenger) instead
        for p in sorted(glob.glob(os.path.join(HIST_DIR, "*.csv"))):
            name = os.path.basename(p)
            if name[:4].isdigit() and int(name[:4]) < 2024:
                paths.append(p)
        paths.sort()
    for path in paths:
        if "ATP_Database" in path:
            continue
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(f):
                when = S.parse_date(row.get("tourney_date"))
                if not when:
                    continue
                a = (row.get("winner_name") or "").strip()
                b = (row.get("loser_name") or "").strip()
                if not a or not b or a == b:
                    continue
                level = S.LEVEL_OF.get((row.get("tourney_level") or "").strip().upper(), "tour")
                if level not in levels:
                    continue
                key = (when, a, b, (row.get("score") or "").strip())
                if key in seen:
                    continue
                ca, cb = S._counts(row, "w", "l"), S._counts(row, "l", "w")
                if not ca or not cb:
                    continue
                seen.add(key)
                surface = row.get("surface", "")
                if surface not in ("Hard", "Clay", "Grass", "Carpet"):
                    surface = ""
                try:
                    bo = int(row["best_of"])
                except (ValueError, TypeError, KeyError):
                    bo = 3
                mid = f"{level[:2]}{row.get('tourney_id','')}"
                out.append((when, mid, a, ca, b, cb, surface, level, bo,
                            S.parse_sets(row.get("score")),
                            S.parse_close(row.get("score"), bo)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


if __name__ == "__main__":
    m = load_counts_uploads()
    print("matches:", len(m))
    print("date range:", m[0][0], "to", m[-1][0])
    players = set()
    for row in m:
        players.add(row[2]); players.add(row[4])
    print("players:", len(players))
