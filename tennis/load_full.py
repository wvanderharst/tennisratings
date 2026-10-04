"""Same match universe as load_uploads.load_counts_uploads, but also keeps
minutes and the raw score string -- needed for scheduling/fatigue (minutes,
date) and injury-proxy (RET/W/O in the score) features. Everything else
(surface, level, best_of, sets) is identical."""
import csv
import glob
import os
import score_driven as S
from load_uploads import UPLOAD_DIR, HIST_DIR


def load_full(levels=("tour", "chall", "qual"), history=True):
    out, seen = [], set()
    paths = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.csv")))
    if history and os.path.isdir(HIST_DIR):
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
                score = (row.get("score") or "").strip()
                key = (when, a, b, score)
                if key in seen:
                    continue
                seen.add(key)
                surface = row.get("surface", "")
                if surface not in ("Hard", "Clay", "Grass", "Carpet"):
                    surface = ""
                try:
                    bo = int(row["best_of"])
                except (ValueError, TypeError, KeyError):
                    bo = 3
                try:
                    minutes = int(row.get("minutes") or 0) or None
                except (ValueError, TypeError):
                    minutes = None
                upper = score.upper()
                retired = ("RET" in upper) or ("W/O" in upper) or ("DEF" in upper) or ("ABD" in upper)
                mid = f"{level[:2]}{row.get('tourney_id','')}"
                out.append(dict(
                    when=when, mid=mid, a=a, b=b, surface=surface, level=level, bo=bo,
                    sets=S.parse_sets(score), minutes=minutes, retired=retired, score=score,
                ))
    out.sort(key=lambda r: (r["when"], r["mid"]))
    return out


if __name__ == "__main__":
    m = load_full()
    print("matches:", len(m))
    print("with minutes:", sum(1 for r in m if r["minutes"]))
    print("retirements/walkovers:", sum(1 for r in m if r["retired"]))
