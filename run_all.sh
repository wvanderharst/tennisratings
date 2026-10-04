#!/usr/bin/env bash
# Rebuild: ratings, odds, charts, profiles, title expectations -> site_v8/ (ready to host).
# Incremental: every replay resumes from its checkpoint in state/ (ckpt.py) and only replays the last weeks; a checkpoint
# that does not match the data exactly is ignored. First run / TENNIS_FULL=1: full replay from 1968 (~12 min, ~6 min with
# the two tours in parallel).
set -euo pipefail
cd "$(dirname "$0")/tennis"
py=${PYTHON:-python3}
export PYTHONHASHSEED=0                 # fixed set/frozenset order -> identical output files run to run

tour() {
  T=$1
  # 1. Corrections-layer feature tables (one long replay per tour)
  if [ "$T" = atp ]; then $py prod_replay_v21e.py      # -> work/cf_FULL_v21e_reference.npz
  else $py wta_run_v2.py wta_tml.csv; fi              # -> work/wta2_wta_tml.npz
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
}

# ATP and WTA are independent until the site is assembled: run them side by side
(set -o pipefail; tour atp 2>&1 | sed -u 's/^/[atp] /') & ATP=$!
(set -o pipefail; tour wta 2>&1 | sed -u 's/^/[wta] /') & WTA=$!
ok=1; wait $ATP || ok=0; wait $WTA || ok=0
[ $ok = 1 ] || { echo "rebuild failed (see the [atp]/[wta] lines above)"; exit 1; }

# 5. Assemble the site, then add actual-vs-predicted titles to the profile index
$py build_site_v8.py                    # -> site_v8/tennis_ratings.html, traj_*.json, odds_*.json
$py merge_titles.py atp
$py merge_titles.py wta

# 6. Optional extras
$py slam_charts.py || true              # -> work/slam_champions_*.png/.csv
$py make_doc.py || true                 # -> work/tennis_model_documentation.pdf (needs reportlab)
echo "Done. Serve the site with:  cd site_v8 && python3 -m http.server 8000"
