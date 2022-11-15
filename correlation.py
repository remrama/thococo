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

# Run repeated-measures correlation.
stat = pg.rm_corr(data=df, x="wandering", y="CoherenceMean", subject="author_id")

# Plot repeated-measures correlation.
g = pg.plot_rm_corr(
    data=df, x="wandering", y="CoherenceMean", subject="author_id",
    kwargs_facetgrid=dict(
        height=2.5, aspect=1,
        palette="cubehelix",
        despine=False,
    )
)
g.ax.set_xticks(range(1, 7))
g.ax.set_xlabel("Self-reported Mind-wandering")
g.ax.set_ylabel("Semantic coherence")
g.ax.yaxis.set(major_locator=plt.MultipleLocator(.2))
g.ax.margins(.1)

# Draw resulting statistics on the plot.
r, p = stat.loc["rm_corr", ["r", "pval"]]
asterisks = "*" * sum( p<cutoff for cutoff in [.05, .01, .001] )
stat_txt = asterisks + fr"$r$ = {r:.2f}".replace("0.", ".")
g.ax.text(.95, .05, stat_txt, va="bottom", ha="right", transform=g.ax.transAxes)

# Export.
stat.to_csv(export_path_stat, index_label="method", sep="\t")
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()