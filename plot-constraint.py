"""
"""
import argparse
from pathlib import Path
import pandas as pd
import utils

import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", type=str, required=True, choices=set(utils.config["corpora"]))
args = parser.parse_args()

corpus_id = args.corpus

metric = "SequentialCoherenceMean"

deriv_dir = Path(utils.config["derivatives_directory"])
export_path = deriv_dir / f"corp-{corpus_id}_constraint.png"

df = pd.read_csv(deriv_dir / f"corp-{corpus_id}_scores.tsv", sep="\t")

# luc = luc.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")
# # df = df.set_index(ids).join(attr.set_index(ids)["wandering"], how="inner")

_, _, attr = utils.load_zipped_corpus_info(corpus_id)
if corpus_id == "thoughtpings":
    col = "off_task"
    attr = attr.query("off_task.isin(['Y', 'N'])")
    # attr = attr.dropna(subset="off_task")
    order = ["off-task", "on-task"]
elif corpus_id == "dreamviews":
    col = "lucidity"
    attr = attr.query("lucidity.isin(['lucid', 'non-lucid'])")
    order = ["non-lucid", "lucid"]
elif corpus_id == "dreamviewsann":
    col = "label"
    attr = attr.query("label.isin(['lucid', 'nonlucid'])")
    order = ["nonlucid", "lucid"]
elif corpus_id == "hippocorpus":
    col = "memType"
    attr = attr.query("memType.isin(['recalled', 'imagined'])")
    order = ["imagined", "recalled"]

df = df.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")

if corpus_id == "thoughtpings":
    df[col] = df[col].replace({"Y": "off-task", "N": "on-task"})


# df = df.query("SequentialCoherenceN.ge(3)")
# df = df.query("Nwords.ge(10)")

# Average authors within each corpus.
avg = df.groupby(["author_id", col])[metric].mean()
    # ).groupby(["corpus_id", "constraint"]).agg(["mean", "sem"])
avg = avg.reorder_levels([col, "author_id"]).sort_index()

if corpus_id == "hippocorpus":
    a, b = avg.loc[order].unstack(col).T.values
    stat = pg.mwu(a, b)
    p = stat.loc["MWU", "p-val"]
else: # paired test
    tab = avg.unstack(0)
    a, b = tab.dropna()[order].T.values
    stat = pg.wilcoxon(a, b)
    p = stat.loc["Wilcoxon", "p-val"]


labels = utils.load_corpus_labels()
# labels = {
#     "dreamviews": "Dreaming",
#     "thoughtpings": "Experience Sampling",
#     "hippocorpus": "Memories",
# }

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
    """
    x1, x2 in data coords; y in axes coords
    """
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

# subset = avg.loc[corpus_id].loc[(slice(None), slice(None), "lucid"), :]
# subset = avg.loc[corpus_id].loc[(slice(None), slice(None), "lucid"), :]
# Extract relevant data.
descr = avg.groupby(col).agg(["count", "mean", "sem"]).loc[order, :]
a, b = avg.groupby(col).apply(list).loc[order]
# descr["xval"] = [i-.2, i+.2]
descr["xval"] = range(len(order))
# ax.boxplot([a, b], positions=xvals)
color = palette[corpus_id]
# palette = {constraints[0]: "gray", constraints[1]: "black"}
ax.errorbar("xval", "mean", fmt="-", yerr="sem", linewidth=.5, color=color, data=descr.loc[order])
ax.scatter("xval", "mean", marker="s", s=20, color=color, data=descr.loc[order])
# ax.errorbar("xval", "mean", fmt="-o", yerr="sem", color=palette[constraints[1]], data=descr.loc[constraints[1]])
# ax.plot("xval", "mean", "-", color="gainsboro", data=descr)
# handles = [ plt.Line2D([0], [0], marker="o", color=c, label=l, markersize=5, linewidth=0)
#     for l, c in palette.items() ]
# leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1))
    # bbox_to_anchor=[(i+1)/(n_corpora+1), 1])
# ax.add_artist(leg)

# ax.set_xlim(-.5, 1.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_xlabel(labels[corpus_id])
ax.set_ylabel("Semantic coherence")

# handles = [ plt.Line2D([0], [0], marker=m, label=l, color="black", markersize=5, linewidth=0)
#     for l, m in {"Low": "s", "High": "D"}.items() ]
# legend = ax.legend(handles=handles, title="Cognitive control",
#     loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
# legend._legend_box.align = "left"


# caplen = BARPLOT_KWARGS["width"]
# fliph = caplen
significance_bars(ax, x1=0, x2=1, y=.9, p=p,
    height=.02, caplength=None, linewidth=1)

ax.margins(x=.5, y=.5)
ax.grid(False)
ax.tick_params(top=False, right=False, direction="out", axis="both")
ax.spines[["top", "right"]].set_visible(False)
# ax.yaxis.set(major_locator=plt.MultipleLocator(.01))


# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()