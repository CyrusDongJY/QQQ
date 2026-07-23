#!/usr/bin/env zsh
set -eu

echo "started_at=$(date)"
echo "cwd=$(pwd)"

for year in 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
  echo "=== convert ${year} $(date) ==="
  PYTHONPATH=src .venv/bin/python -m qqq_options_system convert --year "${year}" --skip-existing
done

echo "=== run full regime study $(date) ==="
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-regime-study --regime-config configs/regime_vix_walk_forward.json
echo "finished_at=$(date)"
