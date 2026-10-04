"""Download the latest TennisMyLife data and append it to the model's input files.

    python3 update_data.py                 # download https://stats.tennismylife.org/api/download-all, then update
    python3 update_data.py --zip FILE.zip  # use a zip you downloaded yourself
    python3 update_data.py --dir FOLDER    # use an already extracted folder
    add --rebuild to run the full rebuild (run_all.sh) afterwards

What it updates (old years before 2024 never change and stay in tennis/tml-db/):
  atp_uploads/   {year}.csv + {year}_challenger.csv for every year >= 2024 in the download (new seasons are picked up
                 automatically), plus ongoing_tourneys.csv and challenger_ongoing_tourneys.csv (matches of events still
                 running, minus any match already in a season file)
  wta_data/      wta_tml.csv + birthdates, rebuilt from all {year}_wta.csv + wta_ongoing_tourneys.csv (wta_prep_v2.py)
  tml_zip/       the extracted download (kept for reference)
Qualifying files are not used by the model and are skipped. Data: TennisMyLife, MIT license."""
import argparse, csv, datetime, io, os, re, shutil, subprocess, sys, urllib.request, zipfile
import paths as P

URL = "https://stats.tennismylife.org/api/download-all"
FIRST_UPLOAD_YEAR = 2024
ap = argparse.ArgumentParser()
ap.add_argument("--zip"); ap.add_argument("--dir"); ap.add_argument("--rebuild", action="store_true")
args = ap.parse_args()
Z = os.path.join(P.ROOT, "tml_zip"); UP = os.path.join(P.ROOT, "atp_uploads")
KEY = lambda r: (r["tourney_date"].strip()[:8], r["winner_name"].strip(), r["loser_name"].strip())


def read(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        rd = csv.DictReader(f); return rd.fieldnames, list(rd)


def snapshot():
    """(matches, last date) per tour, as the model sees them."""
    out = {}
    for tour, files in (("ATP", [os.path.join(UP, f) for f in os.listdir(UP) if f.endswith(".csv") and "ATP_Database" not in f] if os.path.isdir(UP) else []),
                        ("WTA", [os.path.join(P.ROOT, "wta_data", "wta_tml.csv")])):
        keys = set()
        for f in files:
            if os.path.exists(f): keys |= {KEY(r) for r in read(f)[1] if r.get("tourney_date")}
        out[tour] = (len(keys), max((k[0] for k in keys), default="-"))
    return out


before = snapshot()
# ---- 1. get the files ----
if args.dir:
    src = args.dir
else:
    if args.zip:
        data = open(args.zip, "rb").read()
    else:
        print("downloading", URL, "...", flush=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "tennis-ratings-updater (personal, non-commercial)"})
        data = urllib.request.urlopen(req, timeout=300).read()
        print(f"  {len(data) / 1e6:.1f} MB")
    if os.path.isdir(Z): shutil.rmtree(Z)
    os.makedirs(Z)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".csv"): continue
            rel = info.filename.split("/")
            sub = "atp_quali" if "atp_quali" in rel[:-1] else ""          # flatten any top-level folder in the zip
            os.makedirs(os.path.join(Z, sub), exist_ok=True)
            with zf.open(info) as fi, open(os.path.join(Z, sub, rel[-1]), "wb") as fo: shutil.copyfileobj(fi, fo)
    src = Z
files = set(os.listdir(src))
need = {"ongoing_tourneys.csv", "wta_ongoing_tourneys.csv"}
assert need <= files and any(re.fullmatch(r"\d{4}_wta\.csv", f) for f in files), f"download looks incomplete: {sorted(files)[:20]}"

# ---- 2. ATP: season files from 2024 on + ongoing events ----
os.makedirs(UP, exist_ok=True)
seasons = sorted(f for f in files if re.fullmatch(r"(\d{4})(_challenger)?\.csv", f) and int(f[:4]) >= FIRST_UPLOAD_YEAR)
for f in seasons:
    shutil.copy(os.path.join(src, f), os.path.join(UP, f))
season_keys = {KEY(r) for f in seasons for r in read(os.path.join(src, f))[1] if r.get("tourney_date")}
for f in ("ongoing_tourneys.csv", "challenger_ongoing_tourneys.csv"):
    if f not in files: continue
    cols, rows = read(os.path.join(src, f))
    keep = [r for r in rows if r.get("tourney_date") and (r.get("score") or "").strip() and KEY(r) not in season_keys]
    with open(os.path.join(UP, f), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(keep)
    print(f"  {f}: {len(keep)} matches from events still running")
print("ATP files:", ", ".join(seasons))

# ---- 3. WTA: rebuild wta_tml.csv ----
env = dict(os.environ, TML_DIR=src)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wta_prep_v2.py")], check=True, env=env)

after = snapshot()
print("\nmatches (last date):")
for t in ("ATP", "WTA"):
    print(f"  {t}: {before[t][0]} ({before[t][1]})  ->  {after[t][0]} ({after[t][1]})   +{after[t][0] - before[t][0]}")
with open(os.path.join(P.ROOT, "last_update.txt"), "w") as fh:
    fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  ATP {after['ATP'][0]} matches to {after['ATP'][1]}, WTA {after['WTA'][0]} to {after['WTA'][1]}\n")

if args.rebuild:
    subprocess.run(["bash", os.path.join(P.ROOT, "run_all.sh")], check=True)
