"""
Test the relationship between semantic coherence
and other cognitive stuff from different datasets.

Two-way test is paired if dataset allows it.

Exports 3 files:
    - two-way statistics in a tsv file
    - two-way plot in a png file
    - two-way plot in a pdf file
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import sem

import utils


utils.load_matplotlib_settings()


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", type=str, required=True, choices=utils.config["corpora"])
args = parser.parse_args()

corpus_id = args.corpus

metric = "IncoherenceMean"


if corpus_id == "thoughtpings":
    xlabel = "Waking"
    column_of_interest = "off_task"
    label_order = ["off-task", "on-task"]
    method = "Wilcoxon"
    major_yloc = .05
    minor_yloc = .01
elif corpus_id == "dreamviews":
    xlabel = "Dreaming"
    column_of_interest = "lucidity"
    label_order = ["non-lucid", "lucid"]
    method = "Wilcoxon"
    major_yloc = .005
    minor_yloc = .001
elif corpus_id == "hippocorpus":
    xlabel = "Writing"
    column_of_interest = "memType"
    label_order = ["imagined", "recalled"]
    method = "MWU"
    major_yloc = .01
    minor_yloc = .002

deriv_dir = Path(utils.config["derivatives_directory"])
export_path_stat = deriv_dir / f"corp-{corpus_id}_2way.tsv"
export_path_plot = deriv_dir / f"corp-{corpus_id}_2way.png"

df = pd.read_csv(deriv_dir / f"corp-{corpus_id}_scores.tsv", sep="\t")
_, _, attr = utils.load_zipped_corpus_info(corpus_id)
if corpus_id == "thoughtpings":
    attr[column_of_interest] = attr[column_of_interest].replace({"Y": "off-task", "N": "on-task"})
attr = attr.loc[attr[column_of_interest].isin(label_order), :]
df = df.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")

# Average coherence over authors, within each variable of interest.
avg = df.groupby(["author_id", column_of_interest])[metric].mean()
avg = avg.reorder_levels([column_of_interest, "author_id"]).sort_index()
table = avg.unstack(0)


## Run stats.
if method == "Wilcoxon":
    table = table.dropna()
    a, b = table[label_order].T.values
    stat = pg.wilcoxon(a, b)
elif method == "MWU":
    a, b = table[label_order].T.values
    a = a[np.isfinite(a)] 
    b = b[np.isfinite(b)] 
    stat = pg.mwu(a, b)
stat["n(a)"] = a.size
stat["n(b)"] = b.size
stat["mean(a)"] = a.mean()
stat["mean(b)"] = b.mean()
stat["median(a)"] = np.quantile(a, .5)
stat["median(b)"] = np.quantile(b, .5)
stat["std(a)"] = a.std(ddof=1)
stat["std(b)"] = b.std(ddof=1)
stat["sem(a)"] = sem(a)
stat["sem(b)"] = sem(b)


FIGSIZE = (1.5, 2)
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

def significance_bars(ax, x1, x2, y, p, height=.1, linewidth=1, caplength=None):
    """x1, x2 in data coords; y in axes coords"""
    stars = "*" * sum( p < cutoff for cutoff in [.05, .01, .001] )
    color = "black" if stars else "gainsboro"
    x_coords = [x1, x1, x2, x2]
    y_coords = [y, y+height, y+height, y]
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, transform=ax.get_xaxis_transform())
    if caplength is not None:
        for x in [x1, x2]:
            cap_x = [x-caplength/2, x+caplength/2]
            cap_y = [y, y]
            ax.plot(cap_x, cap_y, color=color, linewidth=linewidth, transform=ax.get_xaxis_transform())
    if stars:
        x_txt = (x1+x2) / 2
        ax.text(x_txt, y, stars, fontsize=10, color=color,
            ha="center", va="bottom", transform=ax.get_xaxis_transform())

# Extract relevant data.
xvals = range(len(label_order))
yvals = stat.loc[method, ["mean(a)", "mean(b)"]]
yerrs = stat.loc[method, ["sem(a)", "sem(b)"]]

# Plot.
ax.errorbar(xvals, yvals, fmt="-s", yerr=yerrs, color="#091A60", linewidth=.5, ms=6)
ax.set_xticks(xvals)
ax.set_xticklabels(label_order)
ax.set_xlabel(xlabel)
ax.set_ylabel("Thought variability")

# Significance bars.
p = stat.loc[method, "p-val"]
significance_bars(ax, x1=0, x2=1, y=.9, p=p, height=.02, caplength=None, linewidth=1)

# More aesthetics.
ax.margins(x=.5, y=.5)
ax.tick_params(axis="x", which="both", direction="out", top=False)
# ax.tick_params(axis="both", which="both", top=False, right=False, direction="out")
# ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set(major_locator=plt.MultipleLocator(major_yloc),
             minor_locator=plt.MultipleLocator(minor_yloc))

# Export.
stat.to_csv(export_path_stat, index_label="method", sep="\t")
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()