"""
Test the relationship between self-reported mind-wandering and semantic coherence.

Exports 3 files:
    - correlation statistics in a tsv file
    - correlation plot in a png file
    - correlation plot in a pdf file
"""
from pathlib import Path
import pandas as pd
import utils

import colorcet as cc
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns

utils.load_matplotlib_settings()

# Declare filepaths.
deriv_dir = Path(utils.config["derivatives_directory"])
import_path = deriv_dir / "corp-thoughtpings_scores.tsv"
export_path_stat = deriv_dir / "corp-thoughtpings_corr.tsv"
export_path_plot = deriv_dir / "corp-thoughtpings_corr.png"

# Load data.
df = pd.read_csv(import_path, sep="\t")
_, _, attr = utils.load_zipped_corpus_info("thoughtpings")
attr = attr.dropna(subset="wandering")
df = df.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")
df = df.dropna(subset=["author_id", "wandering", "IncoherenceMean"])

# Run repeated-measures correlation.
stat = pg.rm_corr(data=df, x="wandering", y="IncoherenceMean", subject="author_id")
stat["n"] = df["author_id"].nunique()

# Plot repeated-measures correlation.
palette = cc.cm.glasbey_dark(range(df["author_id"].nunique()))
# palette = cc.cm.glasbey_cool(range(df["author_id"].nunique()))
g = pg.plot_rm_corr(
    data=df, x="wandering", y="IncoherenceMean", subject="author_id",
    kwargs_facetgrid=dict(palette=palette, height=2, aspect=1, despine=False),
    kwargs_line=dict(lw=1, alpha=.7),
    kwargs_scatter=dict(marker="o", s=20, alpha=.7),
)
g.ax.set_xticks(range(1, 7))
# g.ax.set_xlabel(r"More$\leftarrow$Mind-wandering$\rightarrow$Less")
g.ax.set_xlabel("Mind-wandering\n" + r"More $\longleftrightarrow$ Less", labelpad=-4)
g.ax.set_ylabel("Thought variability")
g.ax.yaxis.set(major_locator=plt.MultipleLocator(.5),
               minor_locator=plt.MultipleLocator(.1))
g.ax.xaxis.set(major_locator=plt.FixedLocator([1, 6]),
               minor_locator=plt.MultipleLocator(1))
# g.ax.tick_params(top=False, bottom=False)
g.ax.margins(.1)
g.ax.grid(False)
g.ax.invert_xaxis()

# Draw resulting statistics on the plot.
r, p = stat.loc["rm_corr", ["r", "pval"]]
asterisks = "*" * sum( p<cutoff for cutoff in [.05, .01, .001] )
stat_txt = asterisks + fr"$r$ = {r:.2f}".replace("0.", ".")
g.ax.text(.5, .95, stat_txt, va="top", ha="center", transform=g.ax.transAxes)

# Export.
stat.to_csv(export_path_stat, index_label="method", sep="\t")
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()