

source .venv/bin/activate

python scatterplot_per_Q_acc.py
# python scatterplot_giveaway.py  # only do this if you want the binned giveaway vs reformat scatterplots
python scatterplot_acc_vs_giveaway.py
python scatterplot_acc_vs_embedding.py

python scatterplot_hle_results.py