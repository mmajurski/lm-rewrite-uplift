#!/bin/bash
# Run all plotting scripts to regenerate paper figures.
# Must be run from the project root directory.

source .venv/bin/activate

python analysis/scatterplot_per_Q_acc.py
# python analysis/scatterplot_giveaway.py  # only for binned giveaway vs reformat scatterplots
python analysis/scatterplot_acc_vs_giveaway.py
python analysis/scatterplot_acc_vs_embedding.py

python analysis/scatterplot_hle_results.py
