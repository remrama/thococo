"""
Test the relationship between self-reported mind-wandering and semantic coherence.
Data comes from Mills et al., 2021, Emotion.

Exports 3 files:
    - correlation statistics in a tsv file
    - correlation plot in a png file
    - correlation plot in a pdf file
"""
from pathlib import Path
import pandas as pd
import utils

import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns

sns.set_style("white")

# Declare filepaths.
deriv_dir = Path(utils.config["derivatives_directory"])
import_path = deriv_dir / "corp-thoughtpings_scores.tsv"
export_path_stat = deriv_dir / "corp-thoughtpings_mw-stat.tsv"
export_path_plot = deriv_dir / "corp-thoughtpings_mw-plot.png"

# Load data.
df = pd.read_csv(import_path, sep="\t")
_, _, attr = utils.load_zipped_corpus_info("thoughtpings")
attr = attr.dropna(subset="wandering")
df = df.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")

# Run repeated-measures correlation.
stat = pg.rm_corr(data=df, x="wandering", y="CoherenceMean", subject="author_id")
r, p = stat.loc["rm_corr", ["r", "pval"]]

# Plot repeated-measures correlation.
g = pg.plot_rm_corr(data=df, x="wandering", y="CoherenceMean", subject="author_id",
    kwargs_facetgrid=dict(height=3, aspect=1, palette="cubehelix"))
g.ax.set_xticks(range(1, 7))
g.ax.set_xlabel("Self-reported Mind-wandering")
g.ax.set_ylabel("Semantic coherence")

# Draw resulting statistics on the plot.
txt = f"r = {r:.2f}\np = {p:.3f}"
g.ax.text(.99, .99, txt, transform=g.ax.transAxes, ha="right", va="top")

# Export.
stat.to_csv(export_path_stat, index_label="method", sep="\t")
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()