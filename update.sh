#!/usr/bin/env bash
# Download the latest TennisMyLife data and rebuild everything (for cron / Task Scheduler / by hand).
set -euo pipefail
cd "$(dirname "$0")/tennis"
${PYTHON:-python3} update_data.py --rebuild "$@"
