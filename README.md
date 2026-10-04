# Tennis serve/return ratings — code and website

Serve/return rating model for ATP and WTA (1968–Sep 2026), with surface gaps, closing, clutch, a corrections layer and
retirement risk in the odds. The full model specification is in `tennis_model_documentation.pdf`.

## Quick start

```bash
git clone https://github.com/wvanderharst/tennisratings.git
cd tennisratings

# view the prebuilt site (site_v8/ is committed, so no Python packages are needed for this)
cd site_v8 && python3 -m http.server 8000      # open http://localhost:8000/tennis_ratings.html
cd ..

# rebuild everything from the bundled data
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh                                  # ~10-20 min, rewrites work/ and site_v8/

# or: download the latest TennisMyLife data first, then rebuild
bash update.sh
```

To publish automatically every day, enable GitHub Pages with Settings → Pages → Source: **GitHub Actions**
(see section 3, option A). The daily schedule only runs from the repository's default branch.

## 1. Hosting the website (no Python needed)

The website is a set of static files: `tennis_ratings.html` plus JSON data files (`traj_*`, `odds_*`, `prof_*`), ~106 MB in total.
The page loads its data with `fetch()`, so it must be served over **http(s)**. Double-clicking the HTML file (`file://`) will not work.

**On your own computer**

```bash
cd site_v8                    # the folder that contains tennis_ratings.html
python3 -m http.server 8000
# open http://localhost:8000/tennis_ratings.html
```

**Public, free: GitHub Pages**

1. Create a new public repository and upload the contents of `site_v8/`. Every file is under GitHub's 100 MB per-file limit.
   For an upload this large, `git` is easier than the web uploader:
   ```bash
   cd site_v8
   git init && git add . && git commit -m "site"
   git branch -M main
   git remote add origin https://github.com/<you>/tennis-ratings.git
   git push -u origin main
   ```
2. In the repository, go to Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
3. After a minute the site is at `https://<you>.github.io/tennis-ratings/tennis_ratings.html`.
   To get the plain address `.../tennis-ratings/`, rename `tennis_ratings.html` to `index.html`.

**Alternatives**

- **Netlify:** drag the `site_v8` folder onto app.netlify.com/drop.
- **Cloudflare Pages:** create a project and upload the folder (per-file limit 25 MB; the largest file, `traj_atp.json`, is ~14 MB).
- **Any web host or S3 bucket:** upload the files as they are. No server code or database is needed.

To update the site, replace the files and push or upload again.

## 2. Rebuilding the ratings from the data

**Requirements:** Python 3.10+ and the packages below.

```bash
pip install -r requirements.txt
```

`statsmodels` is only used by a couple of analysis helpers; `reportlab` only by the PDF.

**Run everything**

```bash
bash run_all.sh          # ~10–20 minutes; writes everything into work/ and site_v8/
```

`run_all.sh` lists every step with its output. In order:

| Step | Script | What it does |
|---|---|---|
| 1 | `prod_replay_v21e.py`, `wta_run_v2.py wta_tml.csv` | Long replay that builds the corrections-layer features (rest, home, qualifier, age, layoff, ...) for every match |
| 2 | `fast_prep.py atp\|wta` | Caches the match rows and correction features |
| 3 | `ret_odds_fit.py` | Retirement/walkover risk `r` and slope `β` used in the odds |
| 4 | `odds_export_sr.py` | **The ratings**: replays the serve/return engine (`clone_engine.py`), fits the corrections layer, writes current ratings and odds constants |
| 5 | `build_traj_sr.py` | Rating-history chart data |
| 6 | `build_profiles.py` | Player profiles: every match with win chance and rating change |
| 7 | `expect_titles_haz.py` | Pre-event title chances (simulated draws) → predicted Slams/titles |
| 8 | `slam_analysis.py`, `slam_charts.py` | Slam champions' pre-event ratings and path quality |
| 9 | `build_site_v8.py`, `merge_titles.py` | Assembles `site_v8/` |

All paths are set in `tennis/paths.py` (`ROOT` = this folder, `OUT` = `work/`). Change them there if needed.

## 3. Updating with new matches (automatic)

```bash
bash update.sh          # download the latest data, append it, rebuild everything (~10–20 min)
```

`update.sh` runs `tennis/update_data.py --rebuild`, which:

1. Downloads the full TennisMyLife zip (https://stats.tennismylife.org/api/download-all; updated daily, MIT license).
2. Copies the ATP season files from 2024 onward (tour + challenger) into `atp_uploads/`. A new season such as 2027 is picked up automatically.
3. Adds this week's matches from events still running (`ongoing_tourneys.csv`, `challenger_ongoing_tourneys.csv`), skipping any match that is already in a season file.
4. Rebuilds `wta_data/wta_tml.csv` from all `*_wta.csv` files plus the ongoing WTA events.
5. Prints how many matches were added, then runs `run_all.sh`.

Other ways to run it:

- **Without the rebuild:** `python3 tennis/update_data.py`.
- **From a zip you downloaded yourself:** `python3 tennis/update_data.py --zip tml.zip --rebuild`.
- **Old years:** history before 2024 never changes and stays in `tennis/tml-db/`.
- **Parameters:** nothing is refitted; the model parameters stay fixed in `work/*.json`.

**Run it every day, option A: GitHub (recommended, no computer needed)**

This repository includes `.github/workflows/update-site.yml`.

1. Push this repository to GitHub (already done if you cloned it). Public repositories get free Actions minutes and free Pages.
2. Go to Settings → Pages → Source: **GitHub Actions**.
3. Every day at 05:17 UTC the workflow downloads the data, rebuilds and publishes the site at `https://<you>.github.io/<repo>/`.
4. Actions → "Update tennis ratings" → **Run workflow** starts it by hand.

**Run it every day, option B: on your own computer**

- **macOS/Linux:** add a cron job with `crontab -e`:
  `17 7 * * * /path/to/tennisratings/update.sh >> /path/to/tennisratings/update.log 2>&1`
- **Windows:** in Task Scheduler, create a daily task that runs `python tennis\update_data.py --rebuild` with "Start in" set to the repository folder.
- **Publishing:** upload `site_v8/` afterwards (section 1).

## 4. Where things are

```
tennisratings/
  requirements.txt           Python packages for the rebuild
  update.sh                  download latest data + full rebuild
  run_all.sh                 full rebuild
  .github/workflows/         daily GitHub Actions job (update + publish to Pages)
  tennis/                    Python code + page template (rating_history_artifact_v8.html, app_tabs_v8.js)
    clone_engine.py          the rating engine (serve/return update, surface gaps, closing, retirements)
    confounder_harness.py    data loading, set/match probability chain, calibration constants
    odds_export_sr.py        current ratings + corrections layer + odds constants for the page
    fast_grid.py             parameter search harness used to tune sr_clone_fit.json (not needed to rebuild)
    tml-db/                  ATP history 1968–2023 (TennisMyLife)
  atp_uploads/               ATP 2024 onward: tour + challenger seasons and this week's running events
  wta_data/                  WTA matches 1990 onward (TennisMyLife) + birthdates
  tml_zip/                   the last download, extracted (created by update_data.py)
  work/                      fitted parameters (sr_clone_fit.json = engine settings, ret_fit_*.json,
                             confounder_fit_*.json, ...) and all generated outputs
  site_v8/                   the website after a build
```

**Data:** TennisMyLife (tennismylife.org) ATP and WTA match files, building on Jeff Sackmann's tennis_atp / tennis_wta datasets.
