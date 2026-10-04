#!/usr/bin/env bash
# Full rebuild: ratings, odds, charts, profiles, title expectations -> site_v8/ (ready to host).
# Takes about 10-20 minutes.
set -euo pipefail
cd "$(dirname "$0")/tennis"
py=${PYTHON:-python3}

# 1. Corrections-layer feature tables (one long replay per tour)
$py prod_replay_v21e.py                 # -> work/cf_FULL_v21e_reference.npz   (ATP)
$py wta_run_v2.py wta_tml.csv           # -> work/wta2_wta_tml.npz            (WTA)

for T in atp wta; do
  # 2. Cached rows + correction features for this tour
  $py fast_prep.py $T                   # -> tennis/cache_$T.npz, cache_${T}_rows.pkl
  # 3. Retirement risk for the odds (r, beta)
  $py ret_odds_fit.py $T                # -> work/ret_fit_$T.json
  # 4. Exports for the website
  $py odds_export_sr.py $T              # -> work/odds_data_${T}_sr.json   (current ratings, corrections, odds constants)
  $py build_traj_sr.py $T               # -> work/traj_data_${T}_sr.json   (rating history charts)
  $py build_profiles.py $T              # -> site_v8/prof_${T}_*.json     (player profile pages)
  $py expect_titles_haz.py $T           # -> work/titles_expected_$T.json, title_chances_$T.json
  $py slam_analysis.py $T               # -> work/slam_eve_$T.json
done

# 5. Assemble the site, then add actual-vs-predicted titles to the profile index
$py build_site_v8.py                    # -> site_v8/tennis_ratings.html, traj_*.json, odds_*.json
$py merge_titles.py atp
$py merge_titles.py wta

# 6. Optional extras
$py slam_charts.py || true              # -> work/slam_champions_*.png/.csv
$py make_doc.py || true                 # -> work/tennis_model_documentation.pdf (needs reportlab)
echo "Done. Serve the site with:  cd site_v8 && python3 -m http.server 8000"
