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
import pandas as pd
import utils

import matplotlib.pyplot as plt
import pingouin as pg


utils.load_matplotlib_settings()


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", type=str, required=True, choices=utils.config["corpora"])
args = parser.parse_args()

corpus_id = args.corpus

metric = "CoherenceMean"


if corpus_id == "thoughtpings":
    column_of_interest = "off_task"
    label_order = ["off-task", "on-task"]
    repeated_measures = True
elif corpus_id == "dreamviews":
    column_of_interest = "lucidity"
    label_order = ["non-lucid", "lucid"]
    repeated_measures = True
elif corpus_id == "hippocorpus":
    column_of_interest = "memType"
    label_order = ["imagined", "recalled"]
    repeated_measures = False


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
if repeated_measures:
    table = table.dropna()
    a, b = table[label_order].T.values
    stat = pg.wilcoxon(a, b)
    p = stat.loc["Wilcoxon", "p-val"]
else:
    a, b = table[label_order].T.values
    stat = pg.mwu(a, b)
    p = stat.loc["MWU", "p-val"]


labels = utils.load_corpus_labels()

# corpus_types = utils.load_corpus_types()
# types_palette = utils.load_types_palette()
# palette = { c: types_palette[corpus_types[c]] for c in corpus_order }
palette = utils.load_corpus_palette()

# df["label"] = df["corpus_id"].map(labels_map)
# df["ctype"] = df["corpus_id"].map(types_map)
# df["color"] = df["ctype"].map(palette_map)



FIGSIZE = (2, 2)
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
descr = table[label_order].agg(["count", "mean", "sem"]).T
descr["xval"] = range(len(label_order))
color = palette[corpus_id]
ax.errorbar("xval", "mean", fmt="-", yerr="sem", linewidth=.5, color=color, data=descr.loc[label_order])
ax.scatter("xval", "mean", marker="s", s=20, color=color, data=descr.loc[label_order])

ax.set_xticks(range(len(label_order)))
ax.set_xticklabels(label_order)
ax.set_xlabel(labels[corpus_id])
ax.set_ylabel("Semantic coherence")

significance_bars(ax, x1=0, x2=1, y=.9, p=p, height=.02, caplength=None, linewidth=1)

ax.margins(x=.5, y=.5)
# ax.grid(False)
ax.tick_params(axis="both", which="both", top=False, right=False, direction="out")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set(major_locator=plt.MultipleLocator(.04),
             minor_locator=plt.MultipleLocator(.01))


# Export.
stat.to_csv(export_path_stat, index_label="method", sep="\t")
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()